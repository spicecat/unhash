# %% [markdown]
# <a href="https://colab.research.google.com/github/spicecat/unhash/blob/main/unhash_0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# [https://scratch.mit.edu/projects/164028530](https://scratch.mit.edu/projects/164028530)

# %% [markdown]
# ```js
# function ClickConnect(){
#   console.log("Connnect Clicked - Start");
#   document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
#   console.log("Connnect Clicked - End");
# };
# setInterval(ClickConnect, 10000)
# ```

# %% [markdown]
# # Unhash

# %%
# @title Import libraries

import numpy as np
from numpy.typing import NDArray
import numba as nb

np.set_printoptions(formatter={"int": hex})

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

Enc = np.uint64
NEnc = nb.types.uint64

TARGET = np.array([92227, 32143, 23135, 72362], dtype=np.uint64)


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


@nb.jit(nb.types.boolean(NEnc))
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
# ## Log

# %%
# @title Define drive

import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


if os.path.exists("client_secrets.json"):
    with open("client_secrets.json", "r") as f:
        client_secrets = f.read()
else:
    client_secrets = input("client_secrets.json: ")
if client_secrets:
    with open("client_secrets.json", "w") as f:
        f.write(client_secrets)
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
else:
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)

# %%
# @title Define log
import json

FOLDER_ID = "1savlD9Gk4rPuV4ibw4Zm8jv8fcReA4qH"

LOGS = sorted(
    drive.ListFile(
        {
            "q": f"'{FOLDER_ID}' in parents and mimeType='application/json' and trashed=false"
        }
    ).GetList(),
    key=lambda f: f["title"],
)
for f in LOGS:
    print(f"{f['title']} {f['id']}")

# %%
# @title Define update_log

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)


def update_log(log_file, data):
    try:
        if gauth.access_token_expired:
            if client_secrets:
                gauth.Refresh()
            else:
                from google.colab import auth
                from oauth2client.client import GoogleCredentials
                auth.authenticate_user()
                gauth.credentials = GoogleCredentials.get_application_default()
        log_file.SetContentString(json.dumps(data))
        log_file.Upload()
    except Exception as e:
        logging.error(f"Drive upload failed: {e}")

# %% [markdown]
# 
# ## Run

# %%
# @title Define load_log


def load_log(log_index):
    log_file = LOGS[log_index]
    log = json.loads(log_file.GetContentString())
    logging.info(f"{log_file['title']}, {log_file['id']}")
    logging.info(json.dumps(log))
    return log_file, log

# %%
# @title Define run_search


def run_search(log):
    threads = 256
    blocks = 4096

    log["start"] = (
        int(log["start"] + log["progress"] * log["n_keys"]) - threads * blocks * 256
    )
    logging.info(f"{log['start']=:_}")

    cu_htable = cuda.to_device(scale_htable(Con))
    cu_target = cuda.to_device(scale_target(Con))

    result = cuda.mapped_array(4, dtype=Enc)
    tick = cuda.mapped_array(1, dtype=Enc)
    result[:] = tick[:] = 0

    cu_search[blocks, threads](
        Enc(log["start"]), Enc(log["n_keys"]), cu_htable, cu_target, result, tick
    )

    return result, tick

# %%
# @title Define track_progress

import time
import datetime


def track_progress(log_file, log, result, tick):
    start_time = time.time()
    ticks = max(1, log["n_keys"] // 256 // 0x100000)

    while tick[0] < ticks - 1:
        count = int(result[0])
        if count:
            result[0] = 0
            res = result[1 : count + 1]
            logging.info(f"Check: {res}")
            for r in res:
                if verify(r):
                    logging.info(f"Found: {r}")
                    log["found"].append(r)

        t = tick[0]
        elapsed = time.time() - start_time
        progress = t / ticks
        speed = t * 0x100000 * 256 / elapsed / 1e9
        eta = elapsed / t * (ticks - t) if t else 0

        log["elapsed"] = elapsed
        log["progress"] = progress
        log["speed"] = speed
        log["ETA"] = str(datetime.timedelta(seconds=int(eta)))
        logging.info(json.dumps(log))
        update_log(log_file, log)

        time.sleep(30)

    cuda.synchronize()

# %% [markdown]
# ## Main

# %%
# @title Run main

import threading


def worker(log_index=0, device_id=0):
    cuda.select_device(device_id)
    log_file, log = load_log(log_index)
    result, tick = run_search(log)
    track_progress(log_file, log, result, tick)


cuda.detect()

for i in range(len(cuda.gpus)):
    log_index = int(input("log_index: ") or 0)
    threading.Thread(target=worker, args=(log_index, i,), name=f"GPU-{i}").start()


