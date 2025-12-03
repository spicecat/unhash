# %% [markdown]
# <a href="https://colab.research.google.com/github/spicecat/unhash/blob/main/unhash.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# [https://scratch.mit.edu/projects/164028530](https://scratch.mit.edu/projects/164028530)

# %%
# @title Import libraries

import numpy as np
from numpy.typing import NDArray
import numba as nb

np.set_printoptions(formatter={"int": hex})

# %% [markdown]
# # Unhash

# %% [markdown]
# ## Hash

# %%
# @title Define Encode

Char = np.uint8
Enc = np.uint64
NEnc = nb.types.uint64

C = Char(16)
N = Char(16)
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 "[:C]


@nb.jit(NEnc(nb.types.string))
def encode(s: str) -> Enc:
    enc = np.array([+CHARS.index(c) for c in s], dtype=Char)
    return (enc << Enc(4) * np.arange(N, dtype=Char)).sum(dtype=Enc)


@nb.jit(nb.types.string(NEnc))
def decode(e: Enc) -> str:
    enc = e >> Enc(4) * np.arange(N, dtype=Char)
    return "".join([CHARS[c & 0xF] for c in enc])


assert decode(encode("abcdefghijklmnop")) == "abcdefghijklmnop"

# %%
# @title Define htable

M = np.uint64(100_000)
S = Char(4)


def scale_htable(Con):
    assert np.issubdtype(Con, np.integer)

    MQ = np.float64(np.iinfo(Con).max) + 1.0
    H = Char(2)

    P0 = np.array([11, 17, 7, 5], np.uint32)
    P1 = np.array([29, 31, 17, 13], np.uint32)
    P2 = np.array([53, 67, 103, 47], np.uint32)
    P3 = np.array([52, 12, 24, 30], np.uint32)
    P4 = np.array([0, 90, 0, 90], np.uint32)
    idx = np.arange(N, dtype=np.uint32)
    E = np.outer(np.arange(C, dtype=Char), np.ones(N, dtype=Char))[:, :, np.newaxis]
    sin = 5 * np.sin(np.radians((np.outer(idx + 1, P0) % P1 + P2) * (E + P3) + P4))
    CTABLE = np.round((sin - np.floor(sin)) * MQ).astype(Con)

    HTABLE = (
        CTABLE.reshape(C, N // H, H, S)
        .transpose(2, 0, 1, 3)[
            np.arange(H, dtype=Char).reshape((H,) + (1,) * H),
            np.indices((C.tolist(),) * H)[::-1],
        ]
        .sum(axis=0, dtype=Con)
        .reshape(-1, N // H, S)
    )

    return HTABLE

# %% [markdown]
# ## Search

# %%
# @title Define target

# cbaaaaaadcbaaaaa: 43122, 62816, 54790, 81950
# gfnagjfbjjpnikhp: 24295, 70808, 28723, 42152
# fohdbbjbokjcfiha: 1218, 19828, 86010, 42474
# faaaaaaafaaaaaaa: 45247, 46284, 11706, 55672
# faaaaaaaaaaaaaaa: 83678, 89866, 80596, 35871
# aaaaaaaaaaaaaaaa: 11801, 55301, 96631, 16025
# 92227, 32143, 23135, 72362

Enc = np.uint64
NEnc = nb.types.uint64

TARGET = np.array([45247, 46284, 11706, 55672], dtype=np.uint64)
# TARGET = np.array([83678, 89866, 80596, 35871], dtype=np.uint64)
# TARGET = np.array([92227, 32143, 23135, 72362], dtype=np.uint64)


def scale_target(Con):
    assert np.issubdtype(Con, np.integer)

    MQ = np.float64(np.iinfo(Con).max) + 1.0
    Q = MQ / np.float64(M)
    return (TARGET * Q).astype(Con)

# %%
# @title Define verify

VCon = np.uint32
NVCon = nb.types.uint32
VCons = NDArray[VCon]

V_HTABLE = scale_htable(VCon)
V_TARGET = scale_target(VCon)
VQ = VCon((np.iinfo(VCon).max + 1) // M)


@nb.jit(nb.types.bool(NEnc))
def verify(b: Enc):
    for s in range(S):
        con = VCon(-V_TARGET[s])
        for i in range(8):
            con += V_HTABLE[Char(b >> (8 * i)), i, s]
        if VCon(con) > VQ:
            return False
    return True


del VCon, NVCon, VCons, V_HTABLE, V_TARGET, VQ

# %%
# @title Define cu_search
from numba import cuda

cuda.config.CUDA_ENABLE_PYNVJITLINK = True

Con = np.uint16
NCon = nb.types.uint16
Cons = NDArray[Con]


@cuda.jit([(NEnc, NEnc, NCon[:, :, :], NCon[:], NEnc[:], NEnc[:])], max_registers=40)
def cu_search(
    start: Enc,
    n_keys: Enc,
    htable: Cons,
    target: Cons,
    result: NDArray[Enc],
    tick: NDArray[Enc],
):
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x

    ht = cuda.shared.array((256, 8, 4), dtype=Con)
    for i in range(tid, 8192, bdim):
        seg = i & 3
        temp = i >> 2
        pos = temp & 7
        val = temp >> 3
        ht[val, pos, seg] = htable[val, pos, seg]
    cuda.syncthreads()

    start_chunk = cuda.grid(1)
    n_chunks = n_keys // 256
    stride = cuda.gridDim.x * cuda.blockDim.x

    WINDOW = Con(16)
    OFFSET = Con(8)
    D0 = Con(OFFSET - target[0])
    D1 = Con(OFFSET - target[1])
    D2 = Con(OFFSET - target[2])
    D3 = Con(OFFSET - target[3])

    for chunk_idx in range(start_chunk, n_chunks, stride):
        if (chunk_idx & 0xFFFFF) == 0:
            cuda.atomic.add(tick, 0, 1)

        b = start + chunk_idx * Enc(256)
        b1 = Char(b >> 8)
        b2 = Char(b >> 16)
        b3 = Char(b >> 24)
        b4 = Char(b >> 32)
        b5 = Char(b >> 40)
        b6 = Char(b >> 48)
        b7 = Char(b >> 56)

        ps0 = (
            D0
            + ht[b1, 1, 0]
            + ht[b2, 2, 0]
            + ht[b3, 3, 0]
            + ht[b4, 4, 0]
            + ht[b5, 5, 0]
            + ht[b6, 6, 0]
            + ht[b7, 7, 0]
        )
        ps1 = (
            D1
            + ht[b1, 1, 1]
            + ht[b2, 2, 1]
            + ht[b3, 3, 1]
            + ht[b4, 4, 1]
            + ht[b5, 5, 1]
            + ht[b6, 6, 1]
            + ht[b7, 7, 1]
        )
        ps2 = (
            D2
            + ht[b1, 1, 2]
            + ht[b2, 2, 2]
            + ht[b3, 3, 2]
            + ht[b4, 4, 2]
            + ht[b5, 5, 2]
            + ht[b6, 6, 2]
            + ht[b7, 7, 2]
        )
        ps3 = (
            D3
            + ht[b1, 1, 3]
            + ht[b2, 2, 3]
            + ht[b3, 3, 3]
            + ht[b4, 4, 3]
            + ht[b5, 5, 3]
            + ht[b6, 6, 3]
            + ht[b7, 7, 3]
        )

        for b0 in range(0, 256, 8):
            p0 = ht[b0, 0, 0]
            p1 = ht[b0 + 1, 0, 0]
            p2 = ht[b0 + 2, 0, 0]
            p3 = ht[b0 + 3, 0, 0]
            p4 = ht[b0 + 4, 0, 0]
            p5 = ht[b0 + 5, 0, 0]
            p6 = ht[b0 + 6, 0, 0]
            p7 = ht[b0 + 7, 0, 0]

            if Con(p0 + ps0) <= WINDOW:
                if Con(ht[b0, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0

            if Con(p1 + ps0) <= WINDOW:
                if Con(ht[b0 + 1, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 1, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 1, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 1

            if Con(p2 + ps0) <= WINDOW:
                if Con(ht[b0 + 2, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 2, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 2, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 2

            if Con(p3 + ps0) <= WINDOW:
                if Con(ht[b0 + 3, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 3, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 3, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 3

            if Con(p4 + ps0) <= WINDOW:
                if Con(ht[b0 + 4, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 4, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 4, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 4

            if Con(p5 + ps0) <= WINDOW:
                if Con(ht[b0 + 5, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 5, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 5, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 5

            if Con(p6 + ps0) <= WINDOW:
                if Con(ht[b0 + 6, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 6, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 6, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 6

            if Con(p7 + ps0) <= WINDOW:
                if Con(ht[b0 + 7, 0, 1] + ps1) <= WINDOW:
                    if Con(ht[b0 + 7, 0, 2] + ps2) <= WINDOW:
                        if Con(ht[b0 + 7, 0, 3] + ps3) <= WINDOW:
                            idx = cuda.atomic.add(result, 0, 1)
                            result[idx + 1] = b | b0 + 7

# %% [markdown]
# ## Run Search

# %%
# @title Define Drive
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


SCOPES = ["https://www.googleapis.com/auth/drive.file"]

creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
    with open("token.json", "w") as token:
        token.write(creds.to_json())

SERVICE = build("drive", "v3", credentials=creds)

# %%
# @title Define Log
import sys
import io
import json

FOLDER_ID = "1savlD9Gk4rPuV4ibw4Zm8jv8fcReA4qH"
LOG_MAP = [
    {"file": "unhash_log.json", "id": "1r-SDElZjQQdtqVXgXE2CyHtseMiL5TgQ"},
    {"file": "unhash_log2.json", "id": "1XfF4iZlcZ6dnCv6GHSH8Bd-CZi4-qkXO"},
]
LOG_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
LOG_FILE = LOG_MAP[LOG_INDEX]["file"]
FILE_ID = LOG_MAP[LOG_INDEX]["id"]

if not FILE_ID:
    LOG = {}
else:
    request = SERVICE.files().get_media(fileId=FILE_ID)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    data = fh.read().decode("utf-8")
    LOG = json.loads(data)

print(LOG)


def update_log(data):
    global FILE_ID
    try:
        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(data).encode("utf-8")),
            mimetype="application/json",
            resumable=True,
        )

        if FILE_ID:
            SERVICE.files().update(fileId=FILE_ID, media_body=media).execute()
        else:
            file_metadata = {"name": LOG_FILE, "parents": [FOLDER_ID]}
            file = (
                SERVICE.files()
                .create(body=file_metadata, media_body=media, fields="id")
                .execute()
            )
            FILE_ID = file.get("id")
    except Exception as e:
        print(f"\n[Log Error] Drive upload failed: {e}")


update_log({"start": 5065052563052631040, "n_keys": 1000000000000000000, "found": [], "elapsed": 7802.577965736389, "progress": 0.14752314854363063, "speed": 18906.975255250236, "ETA": "12:31:27"})
print(FILE_ID)

# %%
# @title Define search params

start = 2_509_344_931_948_638_208 # @param {"type":"integer","placeholder":"0"}
n_keys = 1_000_000_000_000_000_000  # @param {"type":"integer","placeholder":"100_000_000"}

start = LOG.get("start", start)
n_keys = LOG.get("n_keys", n_keys)

threads = 256
blocks = 4096

progress = LOG.get("progress", 0.0)
print(f"{start=:_} {n_keys=:_} {progress=}")
start = int(start + progress * n_keys) - threads * blocks * 256
print(f"{start=:_}")
print(f"{(start + n_keys) / (1 << 64):.6}")

# %%
# @title Run search

CU_HTABLE = cuda.to_device(scale_htable(Con))
CU_TARGET = cuda.to_device(scale_target(Con))

result = cuda.mapped_array(4, dtype=Enc)
tick = cuda.mapped_array(1, dtype=Enc)
result[:] = tick[:] = 0


cu_search[blocks, threads](Enc(start), Enc(n_keys), CU_HTABLE, CU_TARGET, result, tick)

# %%
# @title Track progress

import time
import sys
import datetime

LOG["start"] = start
LOG["n_keys"] = n_keys
LOG["found"] = LOG.get("found", [])

start_time = time.time()
ticks = max(1, n_keys // 256 // 0x100000)

while tick[0] < ticks - 1:
    count = result[0]
    if count:
        result[0] = 0
        res = result[1 : count + 1]
        print(f"Check: {res}")
        for r in res:
            if verify(r):
                print(f"Found: {r}")
                LOG["found"].append(r)

    t = tick[0]
    elapsed = time.time() - start_time
    progress = t / ticks
    speed = t * 0x100000 * 256 / elapsed / 1e9
    eta = elapsed / t * (ticks - t) if t else 0

    LOG["elapsed"] = elapsed
    LOG["progress"] = progress
    LOG["speed"] = speed
    LOG["ETA"] = str(datetime.timedelta(seconds=int(eta)))
    sys.stdout.write(f"\r")
    sys.stdout.write(json.dumps(LOG))
    sys.stdout.flush()

    update_log(LOG)

    time.sleep(30)

cuda.synchronize()


