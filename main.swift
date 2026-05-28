import Foundation
import Metal
import simd

// ==========================================
// ERROR HANDLING
// ==========================================

enum UnhashError: Error, LocalizedError {
  case metalUnavailable
  case metalSourceNotFound
  case libraryCompileError(Error)

  var errorDescription: String? {
    switch self {
    case .metalUnavailable:
      return "Metal GPU compute unavailable"
    case .metalSourceNotFound:
      return "Could not read Main.metal"
    case .libraryCompileError(let error):
      return "Failed to compile Metal library: \(error.localizedDescription)"
    }
  }
}

// ==========================================
// CONSTANTS & CONFIGURATION
// ==========================================

/// Application configuration constants
struct Constants {
  /// Number of possible 5-character prefixes (32^5 = 33.5M)
  static let numWords: UInt32 = 33_554_432

  /// Number of threads used to build the hash map (numWords / 8)
  static let buildThreads: UInt32 = 4_194_304

  /// Search space dimensions for Side B
  static let bSideSize = 1024
  static let bInnerSize = 1024

  /// Output buffer sizing
  static let candidatesCapacity = 1000

  /// Character encoding dimensions
  static let tablePosCount = 16  // 16 character positions
  static let tableCharCount = 32  // 32 characters (5-bit encoding)
}

// ==========================================
// CHARACTER SET
// ==========================================

/// Lowercase alphabet with digits (32 characters = 5-bit encoding)
let CHARS = Array("abcdefghijklmnopqrstuvwxyz012345")

// ==========================================
// FILE I/O UTILITIES
// ==========================================

/// Gets the output file path for storing solutions
/// @returns URL to solutions.txt in current directory
func outputURL() -> URL {
  URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("solutions.txt")
}

/// Appends a line to the solutions output file
/// Creates the file if it doesn't exist, appends if it does
/// @param url Target file URL
/// @param line Line to append (newline added automatically)
func appendLine(to url: URL, _ line: String) {
  guard let data = (line + "\n").data(using: .utf8) else { return }

  if FileManager.default.fileExists(atPath: url.path) {
    do {
      let fh = try FileHandle(forWritingTo: url)
      try fh.seekToEnd()
      try fh.write(contentsOf: data)
      try fh.close()
    } catch {
      print("⚠️ Warning: failed to append to \(url.path): \(error)")
    }
  } else {
    do {
      try data.write(to: url, options: .atomic)
    } catch {
      print("⚠️ Warning: write failed: \(error)")
    }
  }
}

// ==========================================
// HASH TABLE GENERATION
// ==========================================

/// Generates the character hash lookup table for all positions and characters
/// Implements sinusoidal mixing for distribution across the hash space
/// @returns Hash table with 16 positions × 32 characters = 512 SIMD4 vectors
func generateHTables() -> [SIMD4<UInt32>] {
  let M = 100_000.0
  let MQ = 4_294_967_296.0

  let P0: [Double] = [11, 17, 7, 5]
  let P1: [Double] = [29, 31, 17, 13]
  let P2: [Double] = [53, 67, 103, 47]
  let P3: [Double] = [52, 12, 24, 30]
  let P4: [Double] = [0, 90, 0, 90]

  var htable = [SIMD4<UInt32>](
    repeating: .zero,
    count: Constants.tablePosCount * Constants.tableCharCount)

  for pos in 0..<Constants.tablePosCount {
    for c in 0..<Constants.tableCharCount {
      var row = SIMD4<UInt32>()

      for s in 0..<4 {
        let term1 =
          (Double(pos + 1) * P0[s])
          .truncatingRemainder(dividingBy: P1[s]) + P2[s]
        let term2 = Double(c) + P3[s]
        let rad = (term1 * term2 + P4[s]) * .pi / 180.0
        let sin_val = 5.0 * sin(rad)
        let frac = sin_val - floor(sin_val)

        var rounded = round(frac * MQ)
        if rounded >= MQ { rounded = 0 }
        row[s] = UInt32(rounded)
      }

      htable[pos * Constants.tableCharCount + c] = row
    }
  }

  return htable
}

/// Pre-computes 2D combinations of hash values from 1D table
/// Reduces GPU computation by pre-computing common combinations
/// @param htable Base character hash table (16 pos × 32 chars)
/// @returns Three tables with pre-combined hash values for GPU search space
func generateTableB(from htable: [SIMD4<UInt32>]) -> (
  outer: [SIMD4<UInt32>],
  mid: [SIMD4<UInt32>],
  inner: [SIMD4<UInt32>]
) {
  var outer = [SIMD4<UInt32>](repeating: .zero, count: Constants.bSideSize)
  var mid = [SIMD4<UInt32>](repeating: .zero, count: Constants.bSideSize)
  var inner = [SIMD4<UInt32>](repeating: .zero, count: Constants.bSideSize)

  // Precompute all 32×32 combinations for 3 dimension pairs
  for i in 0..<Constants.tableCharCount {
    for j in 0..<Constants.tableCharCount {
      let idx = i | (j << 5)
      outer[idx] = htable[5 * 32 + i] &+ htable[6 * 32 + j]
      mid[idx] = htable[7 * 32 + i] &+ htable[8 * 32 + j]
      inner[idx] = htable[9 * 32 + i] &+ htable[10 * 32 + j]
    }
  }

  return (outer, mid, inner)
}

// ==========================================
// GPU SETUP
// ==========================================

/// Loads and compiles Metal library from disk
/// @param device Metal device to create library on
/// @param metalPath Path to Metal source file
/// @throws UnhashError if compilation fails
/// @returns Compiled Metal library
func makeLibrary(device: MTLDevice, metalPath: URL) throws -> MTLLibrary {
  let source: String
  do {
    source = try String(contentsOf: metalPath, encoding: .utf8)
  } catch {
    throw UnhashError.metalSourceNotFound
  }

  do {
    return try device.makeLibrary(source: source, options: nil)
  } catch {
    throw UnhashError.libraryCompileError(error)
  }
}

// ==========================================
// PHRASE RECONSTRUCTION
// ==========================================

/// Reconstructs original 16-character phrase from packed hash encoding
/// Reverses the bit-packing used during GPU search
/// @param a_val Packed 5-character prefix (32-bit ID)
/// @param b_val Packed 11-character suffix (32-bit ID encoding 3 character pairs)
/// @param charIndices Suffix ID unpacked into individual character indices
/// @returns Complete 16-character phrase
func reconstructPhrase(a_val: Int, b_val: Int, charIndices: [Int]) -> String {
  // Unpack Side A: 5 characters × 5 bits each = 25 bits used
  let c0 = a_val & 31
  let c1 = (a_val >> 5) & 31
  let c2 = (a_val >> 10) & 31
  let c3 = (a_val >> 15) & 31
  let c4 = (a_val >> 20) & 31

  // Unpack Side B: 3 pairs of characters
  let outer = (b_val >> 20) & 1023
  let mid = (b_val >> 10) & 1023
  let b_inner = b_val & 1023

  let c5 = outer & 31
  let c6 = (outer >> 5) & 31
  let c7 = mid & 31
  let c8 = (mid >> 5) & 31
  let c9 = b_inner & 31
  let c10 = (b_inner >> 5) & 31

  let allCharIndices = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10] + charIndices
  return allCharIndices.map { String(CHARS[$0]) }.joined()
}

// ==========================================
// PROGRAM ENTRY POINT
// ==========================================

/// Main execution function
/// - Builds hash map from all 5-character prefixes
/// - Sweeps suffix space to find matching 16-character phrases
/// - Optional: accepts starting suffix ID for resuming interrupted searches
func main() throws {
  // Parse optional starting suffix ID from command line
  var startSuffixID = 0
  if CommandLine.arguments.count > 1, let parsed = Int(CommandLine.arguments[1]) {
    startSuffixID = parsed
    print("ℹ️ Starting search from user-defined Suffix ID: \(startSuffixID)")
  }

  let outURL = outputURL()

  // ==========================================
  // Phase 1: Hash Table Generation
  // ==========================================
  print("\n[Phase 1] Generating hash tables...")
  let htable = generateHTables()
  print("  ✓ Primary hash table generated (512 entries)")

  let tableB = generateTableB(from: htable)
  print("  ✓ Pre-computed B-side tables generated (3×1024 entries)")

  // ==========================================
  // Phase 2: GPU Setup
  // ==========================================
  print("\n[Phase 2] Setting up GPU environment...")
  guard let device = MTLCreateSystemDefaultDevice(),
    let commandQueue = device.makeCommandQueue()
  else {
    throw UnhashError.metalUnavailable
  }
  print("  ✓ Metal device initialized")

  let metalFileURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("Main.metal")
  let library = try makeLibrary(device: device, metalPath: metalFileURL)
  print("  ✓ Metal library compiled")

  guard let initFunc = library.makeFunction(name: "init_map"),
    let buildFunc = library.makeFunction(name: "build_map"),
    let sieveFunc = library.makeFunction(name: "sieve")
  else {
    fatalError("❌ Missing required kernel functions in Main.metal")
  }

  let initPipeline = try device.makeComputePipelineState(function: initFunc)
  let buildPipeline = try device.makeComputePipelineState(function: buildFunc)
  let sievePipeline = try device.makeComputePipelineState(function: sieveFunc)
  print("  ✓ Compute pipelines created")

  // ==========================================
  // Phase 3: Buffer Allocation
  // ==========================================
  print("\n[Phase 3] Allocating GPU memory...")
  let mapBufferSize = Int(Constants.numWords) * MemoryLayout<UInt32>.stride
  let htableBufferSize = htable.count * MemoryLayout<SIMD4<UInt32>>.stride
  let tableBSize = Constants.bSideSize * MemoryLayout<SIMD4<UInt32>>.stride
  let candidatesSize = Constants.candidatesCapacity * MemoryLayout<UInt32>.stride
  let counterSize = MemoryLayout<UInt32>.stride

  guard let mapBuffer = device.makeBuffer(length: mapBufferSize, options: .storageModeShared),
    let htableBuffer = device.makeBuffer(
      bytes: htable, length: htableBufferSize,
      options: .storageModeShared),
    let tOuterBuffer = device.makeBuffer(
      bytes: tableB.outer, length: tableBSize,
      options: .storageModeShared),
    let tMidBuffer = device.makeBuffer(
      bytes: tableB.mid, length: tableBSize,
      options: .storageModeShared),
    let tInnerBuffer = device.makeBuffer(
      bytes: tableB.inner, length: tableBSize,
      options: .storageModeShared),
    let candidatesABuffer = device.makeBuffer(
      length: candidatesSize,
      options: .storageModeShared),
    let candidatesBBuffer = device.makeBuffer(
      length: candidatesSize,
      options: .storageModeShared),
    let counterBuffer = device.makeBuffer(length: counterSize, options: .storageModeShared)
  else {
    fatalError("❌ Failed to allocate GPU memory")
  }
  print("  ✓ GPU buffers allocated (\(Int(Constants.numWords)/1_000_000)GB map)")

  // ==========================================
  // Phase 4: Hash Map Initialization
  // ==========================================
  print("\n[Phase 4] Initializing hash map...")
  do {
    let cb0 = commandQueue.makeCommandBuffer()!
    let enc0 = cb0.makeComputeCommandEncoder()!
    enc0.setComputePipelineState(initPipeline)
    enc0.setBuffer(mapBuffer, offset: 0, index: 0)
    let threadsPerGroup = MTLSize(
      width: min(initPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
    enc0.dispatchThreads(
      MTLSize(width: Int(Constants.numWords), height: 1, depth: 1),
      threadsPerThreadgroup: threadsPerGroup)
    enc0.endEncoding()
    cb0.commit()
    cb0.waitUntilCompleted()
  }
  print("  ✓ Hash map initialized")

  // ==========================================
  // Phase 5: Hash Map Building
  // ==========================================
  print("[Phase 5] Building hash map from prefixes...")
  do {
    let cb1 = commandQueue.makeCommandBuffer()!
    let enc1 = cb1.makeComputeCommandEncoder()!
    enc1.setComputePipelineState(buildPipeline)
    enc1.setBuffer(mapBuffer, offset: 0, index: 0)
    enc1.setBuffer(htableBuffer, offset: 0, index: 1)
    let threadsPerGroup = MTLSize(
      width: min(buildPipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
    enc1.dispatchThreads(
      MTLSize(width: Int(Constants.buildThreads), height: 1, depth: 1),
      threadsPerThreadgroup: threadsPerGroup)
    enc1.endEncoding()
    cb1.commit()
    cb1.waitUntilCompleted()
  }
  print("  ✓ Hash map built")

  // ==========================================
  // Phase 6: Suffix Search
  // ==========================================
  print("\n[Phase 6] Sweeping suffix space...\n")
  print("Saving solutions to: \(outURL.path)\n")

  let threadgroupSize = MTLSize(width: 32, height: 32, depth: 1)
  let gridSize = MTLSize(width: 1024, height: 1024, depth: 1)

  // Precompute target hash
  let M = 100_000.0
  let MQ = 4_294_967_296.0
  let targetArray: [Double] = [92227, 32143, 23135, 72362]
  // let targetArray: [Double] = [86679, 57610, 20255, 93418]
  var exactTarget = SIMD4<UInt32>.zero
  for s in 0..<4 {
    exactTarget[s] = UInt32((targetArray[s] * MQ) / M)
  }
  var VQ = UInt32(MQ / M)

  let startTime = Date()
  var suffixID = startSuffixID

  while suffixID < Int(Constants.buildThreads) {
    // Progress reporting every 100 iterations
    if suffixID > startSuffixID && suffixID % 100 == 0 {
      let elapsed = Date().timeIntervalSince(startTime)
      let loopsDone = Double(suffixID - startSuffixID)
      let loopsPerSec = loopsDone / max(1e-6, elapsed)
      let expectedTimeMins = max(
        0, (1000.0 - (loopsDone.truncatingRemainder(dividingBy: 1000.0))) / loopsPerSec / 60.0)
      print(
        String(
          format: "  -> Sweeping offset %d... Speed: %.1f loops/sec. Next stat match in ~%.1f mins",
          suffixID, loopsPerSec, expectedTimeMins))
    }

    // Unpack suffix ID into individual character indices
    let c11 = suffixID & 31
    let c12 = (suffixID >> 5) & 31
    let c13 = (suffixID >> 10) & 31
    let c14 = (suffixID >> 15) & 31
    let c15 = (suffixID >> 20) & 31

    // Compute suffix hash
    var suffixHash = htable[11 * 32 + c11] &+ htable[12 * 32 + c12]
    suffixHash =
      suffixHash &+ htable[13 * 32 + c13] &+ htable[14 * 32 + c14] &+ htable[15 * 32 + c15]

    // Compute target for this suffix
    var offsetTarget = exactTarget &- suffixHash
    counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

    // Execute sieve kernel
    let cb2 = commandQueue.makeCommandBuffer()!
    let enc2 = cb2.makeComputeCommandEncoder()!
    enc2.setComputePipelineState(sievePipeline)

    enc2.setBuffer(mapBuffer, offset: 0, index: 0)
    enc2.setBytes(&offsetTarget, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)
    enc2.setBuffer(htableBuffer, offset: 0, index: 2)
    enc2.setBuffer(tOuterBuffer, offset: 0, index: 3)
    enc2.setBuffer(tMidBuffer, offset: 0, index: 4)
    enc2.setBuffer(tInnerBuffer, offset: 0, index: 5)
    enc2.setBuffer(candidatesABuffer, offset: 0, index: 6)
    enc2.setBuffer(candidatesBBuffer, offset: 0, index: 7)
    enc2.setBuffer(counterBuffer, offset: 0, index: 8)
    enc2.setBytes(&VQ, length: MemoryLayout<UInt32>.stride, index: 9)

    enc2.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    enc2.endEncoding()
    cb2.commit()
    cb2.waitUntilCompleted()

    // Process results
    let candidateCount = counterBuffer.contents()
      .bindMemory(to: UInt32.self, capacity: 1).pointee
    if candidateCount > 0 {
      let a_vals = candidatesABuffer.contents()
        .bindMemory(to: UInt32.self, capacity: Int(candidateCount))
      let b_vals = candidatesBBuffer.contents()
        .bindMemory(to: UInt32.self, capacity: Int(candidateCount))

      for i in 0..<Int(candidateCount) {
        let charIndices = [c11, c12, c13, c14, c15]
        let finalString = reconstructPhrase(
          a_val: Int(a_vals[i]), b_val: Int(b_vals[i]),
          charIndices: charIndices)
        print("\n🎉 SUCCESS! Solution Found at Suffix ID \(suffixID): \(finalString)")
        appendLine(to: outURL, "[\(Date())] Suffix ID: \(suffixID) | Solution: \(finalString)")
      }
    }

    suffixID += 1
  }

  print("\n✓ Search Complete.")
}

// ==========================================
// EXECUTION
// ==========================================

do {
  try main()
} catch {
  print("❌ Fatal error: \(error.localizedDescription)")
  exit(1)
}
