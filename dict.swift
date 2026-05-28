import Foundation
import Metal
import simd

// ==========================================
// ERROR HANDLING
// ==========================================

enum DictError: Error, LocalizedError {
  case dictNotFound
  case dictEmpty
  case metalUnavailable
  case metalSourceNotFound
  case missingKernelFunction
  case libraryCompileError(Error)

  var errorDescription: String? {
    switch self {
    case .dictNotFound:
      return "Could not find 'dict.txt'. Please create it in the same folder as this script."
    case .dictEmpty:
      return "dict.txt is empty or contained no valid words."
    case .metalUnavailable:
      return "Metal GPU compute unavailable"
    case .metalSourceNotFound:
      return "Could not read Dict.metal"
    case .missingKernelFunction:
      return "Missing required kernel functions in Dict.metal"
    case .libraryCompileError(let error):
      return "Failed to compile Metal library: \(error.localizedDescription)"
    }
  }
}

// ==========================================
// CONSTANTS & CONFIGURATION
// ==========================================

/// Mathematical constants for hash generation
enum MathConstants {
  static let M = 100_000.0  // Target value scale factor
  static let MQ = 4_294_967_296.0  // 2^32 for uint32 quantization
  static let characterCount = 37  // 26 letters + 10 digits + 1 space
  static let maxStringLength = 128  // Maximum string length in bytes
  static let hashTablePositions = 128  // Positions sampled for each character
  static let hashTableSize = 128 * 37  // Total entries: positions × characters
}

/// Target hash values for matching
enum TargetConstants {
  static let targetArray: [Double] = [92227, 32143, 23135, 72362]
  // static let targetArray: [Double] = [86679, 57610, 20255, 93418]
}

/// Hash function parameters (P0-P4 polynomials)
enum HashParameters {
  static let P0: [Double] = [11, 17, 7, 5]
  static let P1: [Double] = [29, 31, 17, 13]
  static let P2: [Double] = [53, 67, 103, 47]
  static let P3: [Double] = [52, 12, 24, 30]
  static let P4: [Double] = [0, 90, 0, 90]
}

/// GPU resource sizing
enum GPUConstants {
  static let mapElements = 268_435_456  // 1 GB hash map
  static let candidatesCapacity = 1000  // Max matches per dispatch
}

/// Search parameters
enum SearchConstants {
  static let minOffsetLength = 2
  static let maxOffsetLength = 50
  static let minTotalLength = 12
  static let maxTotalLength = 128
}

// ==========================================
// DICTIONARY SETUP
// ==========================================

/// Valid character set for the dictionary
let CHARS = Array("abcdefghijklmnopqrstuvwxyz0123456789 ")
var charToIndex: [Character: Int] = [:]

/// Initialize character-to-index mapping
func initializeCharacterMap() {
  for (i, c) in CHARS.enumerated() {
    charToIndex[c] = i
  }
}

/// Load and validate dictionary from file
/// @throws DictError if dict.txt cannot be found or is empty
/// @returns Array of valid words (lowercase, unique, valid characters only)
func loadDictionary() throws -> [String] {
  let dictURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("dict.txt")

  guard let fileContents = try? String(contentsOf: dictURL, encoding: .utf8) else {
    throw DictError.dictNotFound
  }

  let rawSplit = fileContents.components(separatedBy: .whitespacesAndNewlines)
  var words: [String] = []

  for w in rawSplit {
    let clean = w.lowercased().trimmingCharacters(in: .whitespaces)

    // Validate: not empty, valid characters only, not duplicate
    if !clean.isEmpty && clean.allSatisfy({ charToIndex[$0] != nil }) && !words.contains(clean) {
      words.append(clean)
    }
  }

  if words.isEmpty {
    throw DictError.dictEmpty
  }

  return words
}

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

/// Logs a discovered solution to solutions.txt with timestamp
/// @param phrase The matched phrase
func logSolution(phrase: String) {
  let entry = "[\(Date())] Solution: \(phrase)"
  appendLine(to: outputURL(), entry)
}

// ==========================================
// CHARACTER HASH PRECOMPUTATION
// ==========================================

/// Precomputes hash values for all (position, character) pairs
/// Uses sinusoidal function with modular arithmetic for distribution
/// @returns Hash table with 128 positions × 37 characters = 4,736 SIMD4 vectors
func generateHashTable() -> [SIMD4<UInt32>] {
  var htable = [SIMD4<UInt32>](
    repeating: .zero,
    count: MathConstants.hashTableSize)

  for pos in 0..<MathConstants.hashTablePositions {
    for c in 0..<MathConstants.characterCount {
      var row = SIMD4<UInt32>()

      for s in 0..<4 {
        // Polynomial mixing with modular reduction
        let term1 =
          (Double(pos + 1) * HashParameters.P0[s])
          .truncatingRemainder(dividingBy: HashParameters.P1[s]) + HashParameters.P2[s]

        // Sine-based hash: maps to [0, MQ)
        let rad = (term1 * (Double(c) + HashParameters.P3[s]) + HashParameters.P4[s]) * .pi / 180.0
        let sin_val = 5.0 * sin(rad)
        let frac = sin_val - floor(sin_val)

        var rounded = round((frac) * MathConstants.MQ)
        if rounded >= MathConstants.MQ { rounded = 0 }
        row[s] = UInt32(rounded)
      }

      htable[pos * MathConstants.characterCount + c] = row
    }
  }

  return htable
}

// ==========================================
// WORD HASH PRECOMPUTATION
// ==========================================

/// Precomputes hash values for all words at all positions
/// @param words Dictionary of valid words
/// @param htable Character hash lookup table
/// @returns Tuple of (lengths with space, lengths without space, hashes with space, hashes without space)
func precomputeWordHashes(words: [String], htable: [SIMD4<UInt32>])
  -> (lenS: [UInt32], lenN: [UInt32], hashesSpace: [SIMD4<UInt32>], hashesNoSpace: [SIMD4<UInt32>])
{

  let N = words.count
  var lenS = [UInt32](repeating: 0, count: N)
  var lenN = [UInt32](repeating: 0, count: N)
  var hashesSpace = [SIMD4<UInt32>](repeating: .zero, count: N * MathConstants.hashTablePositions)
  var hashesNoSpace = [SIMD4<UInt32>](repeating: .zero, count: N * MathConstants.hashTablePositions)

  for (id, word) in words.enumerated() {
    let withSpace = word + " "
    lenS[id] = UInt32(withSpace.count)
    lenN[id] = UInt32(word.count)

    // Precompute hashes for all starting positions
    for pos in 0..<MathConstants.hashTablePositions {
      // With trailing space
      if pos + withSpace.count <= MathConstants.maxStringLength {
        var h = SIMD4<UInt32>.zero
        for (i, c) in withSpace.enumerated() {
          h &+= htable[(pos + i) * MathConstants.characterCount + charToIndex[c]!]
        }
        hashesSpace[id * MathConstants.hashTablePositions + pos] = h
      }

      // Without trailing space
      if pos + word.count <= MathConstants.maxStringLength {
        var h = SIMD4<UInt32>.zero
        for (i, c) in word.enumerated() {
          h &+= htable[(pos + i) * MathConstants.characterCount + charToIndex[c]!]
        }
        hashesNoSpace[id * MathConstants.hashTablePositions + pos] = h
      }
    }
  }

  return (lenS, lenN, hashesSpace, hashesNoSpace)
}

// ==========================================
// GPU SETUP & INITIALIZATION
// ==========================================

/// Loads Metal library and creates compute pipelines
/// @throws Error if Metal is unavailable or compilation fails
/// @returns Tuple of (device, queue, pipelines)
func setupMetalEnvironment(words: [String]) throws
  -> (
    device: MTLDevice, queue: MTLCommandQueue,
    pipelines: (
      initMap: MTLComputePipelineState, build: MTLComputePipelineState,
      sieve: MTLComputePipelineState
    )
  )
{

  guard let device = MTLCreateSystemDefaultDevice(),
    let queue = device.makeCommandQueue()
  else {
    throw DictError.metalUnavailable
  }

  let metalURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    .appendingPathComponent("Dict.metal")

  guard let rawMetalSource = try? String(contentsOf: metalURL, encoding: .utf8) else {
    throw DictError.metalSourceNotFound
  }

  let finalMetalSource = rawMetalSource.replacingOccurrences(of: "CONST_N", with: "\(words.count)u")

  let library: MTLLibrary
  do {
    library = try device.makeLibrary(source: finalMetalSource, options: nil)
  } catch {
    throw DictError.libraryCompileError(error)
  }

  guard let initFunc = library.makeFunction(name: "init_map"),
    let buildFunc = library.makeFunction(name: "build_map"),
    let sieveFunc = library.makeFunction(name: "sieve_map")
  else {
    throw DictError.missingKernelFunction
  }

  let initPipe = try device.makeComputePipelineState(function: initFunc)
  let buildPipe = try device.makeComputePipelineState(function: buildFunc)
  let sievePipe = try device.makeComputePipelineState(function: sieveFunc)

  // Fixed: label updated to `initMap` to prevent Swift keyword collision
  return (device, queue, (initMap: initPipe, build: buildPipe, sieve: sievePipe))
}

// ==========================================
// SOLUTION RECONSTRUCTION
// ==========================================

/// Reconstructs a phrase from packed Side A and Side B IDs
/// @param a_val Packed 32-bit value with two 16-bit word IDs
/// @param b_val Packed 64-bit value with up to four 16-bit word IDs
/// @param wB Number of words on Side B (1-4)
/// @param words Dictionary of words
/// @returns Complete phrase as space-separated words
func reconstructPhrase(a_val: UInt32, b_val: UInt64, wB: Int, words: [String]) -> String {
  var strIds = [Int]()

  // Side A: always exactly 2 words
  strIds.append(Int(a_val & 0xFFFF))
  strIds.append(Int(a_val >> 16))

  // Side B: 1 to 4 words
  strIds.append(Int(b_val & 0xFFFF))
  if wB >= 2 { strIds.append(Int((b_val >> 16) & 0xFFFF)) }
  if wB >= 3 { strIds.append(Int((b_val >> 32) & 0xFFFF)) }
  if wB >= 4 { strIds.append(Int((b_val >> 48) & 0xFFFF)) }

  return strIds.map { words[$0] }.joined(separator: " ")
}

// ==========================================
// GPU EXECUTION ENGINE
// ==========================================

/// Initializes the GPU hash map to empty state
func initializeHashMap(
  queue: MTLCommandQueue, mapBuffer: MTLBuffer, initPipe: MTLComputePipelineState
) {
  let cb0 = queue.makeCommandBuffer()!
  let enc0 = cb0.makeComputeCommandEncoder()!
  enc0.setComputePipelineState(initPipe)
  enc0.setBuffer(mapBuffer, offset: 0, index: 0)
  enc0.dispatchThreads(
    MTLSize(width: GPUConstants.mapElements, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  enc0.endEncoding()
  cb0.commit()
  cb0.waitUntilCompleted()
}

/// Builds the hash map from 2-word combinations
func buildHashMap(
  queue: MTLCommandQueue, mapBuffer: MTLBuffer, buildPipe: MTLComputePipelineState,
  wordCount: Int, bufferLenS: MTLBuffer, bufferHashS: MTLBuffer
) {
  let cb1 = queue.makeCommandBuffer()!
  let enc1 = cb1.makeComputeCommandEncoder()!
  enc1.setComputePipelineState(buildPipe)
  enc1.setBuffer(mapBuffer, offset: 0, index: 0)
  enc1.setBuffer(bufferLenS, offset: 0, index: 1)
  enc1.setBuffer(bufferHashS, offset: 0, index: 2)
  enc1.dispatchThreads(
    MTLSize(width: wordCount, height: wordCount, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
  enc1.endEncoding()
  cb1.commit()
  cb1.waitUntilCompleted()
}

/// Executes the sieve search across all word combinations
func executeSieve(
  queue: MTLCommandQueue, mapBuffer: MTLBuffer, sievePipe: MTLComputePipelineState,
  wordCount: Int,
  buffers: (
    lenS: MTLBuffer, lenN: MTLBuffer, hashS: MTLBuffer,
    hashN: MTLBuffer, candsA: MTLBuffer, candsB: MTLBuffer, counter: MTLBuffer
  ),
  words: [String]
) -> Set<String> {

  let N = UInt32(wordCount)
  var VQ = UInt32(MathConstants.MQ / MathConstants.M)

  var REAL_TARGET = SIMD4<UInt32>.zero
  for s in 0..<4 {
    REAL_TARGET[s] = UInt32((TargetConstants.targetArray[s] * MathConstants.MQ) / MathConstants.M)
  }

  var foundPhrases = Set<String>()

  // Fixed: Mutability warning (Changed var to let)
  let sweepStartTime = Date()

  var totalDispatches = 0
  var completedDispatches = 0

  // Calculate total GPU dispatches needed
  for wB in 1...4 {
    let cpu_multiplier = (wB == 4) ? Int(N) : 1
    totalDispatches +=
      (SearchConstants.maxOffsetLength - SearchConstants.minOffsetLength + 1) * cpu_multiplier
  }
  print("Total GPU Dispatches: \(totalDispatches)\n")

  // Sweep through all word combinations
  for wB in 1...4 {
    print("\n=== Sweeping \(wB) Words on Side B (Max \(wB + 2) words total) ===")

    for off in SearchConstants.minOffsetLength...SearchConstants.maxOffsetLength {
      let max_cpu_loops = (wB == 4) ? Int(N) : 1

      for id3 in 0..<max_cpu_loops {
        autoreleasepool {
          var cWords = UInt32(wB)
          var uOff = UInt32(off)
          var uId3 = UInt32(id3)
          memset(buffers.counter.contents(), 0, 4)

          let cb2 = queue.makeCommandBuffer()!
          let enc2 = cb2.makeComputeCommandEncoder()!
          enc2.setComputePipelineState(sievePipe)

          enc2.setBuffer(mapBuffer, offset: 0, index: 0)
          enc2.setBuffer(buffers.lenS, offset: 0, index: 1)
          enc2.setBuffer(buffers.lenN, offset: 0, index: 2)
          enc2.setBuffer(buffers.hashS, offset: 0, index: 3)
          enc2.setBuffer(buffers.hashN, offset: 0, index: 4)
          enc2.setBytes(&cWords, length: 4, index: 5)
          enc2.setBytes(&uOff, length: 4, index: 6)
          enc2.setBytes(&REAL_TARGET, length: 16, index: 7)
          enc2.setBuffer(buffers.candsA, offset: 0, index: 8)
          enc2.setBuffer(buffers.candsB, offset: 0, index: 9)
          enc2.setBuffer(buffers.counter, offset: 0, index: 10)
          enc2.setBytes(&VQ, length: 4, index: 11)
          enc2.setBytes(&uId3, length: 4, index: 12)

          let gridX = Int(N)
          let gridY = (wB >= 2) ? Int(N) : 1
          let gridZ = (wB >= 3) ? Int(N) : 1

          enc2.dispatchThreads(
            MTLSize(width: gridX, height: gridY, depth: gridZ),
            threadsPerThreadgroup: MTLSize(
              width: 8, height: 8,
              depth: (gridZ > 1 ? 4 : 1)))
          enc2.endEncoding()
          cb2.commit()
          cb2.waitUntilCompleted()

          completedDispatches += 1

          // Progress reporting
          if completedDispatches % 100 == 0 || completedDispatches == totalDispatches {
            let elapsed = Date().timeIntervalSince(sweepStartTime)
            let dps = Double(completedDispatches) / max(1e-5, elapsed)
            let remaining = Double(totalDispatches - completedDispatches) / dps

            let hrs = Int(remaining) / 3600
            let mins = (Int(remaining) % 3600) / 60
            let secs = Int(remaining) % 60
            let pct = (Double(completedDispatches) / Double(totalDispatches)) * 100.0

            print(
              String(
                format:
                  "\r    -> [%.2f%% | Off: %d] Speed: %.1f disp/s | ETA: %02d:%02d:%02d \u{1B}[K",
                pct, off, dps, hrs, mins, secs), terminator: "")
            fflush(stdout)
          }

          // Process candidates
          let candCount = buffers.counter.contents()
            .bindMemory(to: UInt32.self, capacity: 1).pointee
          if candCount > 0 {
            let pA = buffers.candsA.contents()
              .bindMemory(to: UInt32.self, capacity: Int(candCount))
            let pB = buffers.candsB.contents()
              .bindMemory(to: UInt64.self, capacity: Int(candCount))

            for i in 0..<Int(candCount) {
              let phrase = reconstructPhrase(a_val: pA[i], b_val: pB[i], wB: wB, words: words)
              if !foundPhrases.contains(phrase) {
                foundPhrases.insert(phrase)
                print("\n\n    🎉 SUCCESS! MATCH FOUND: '\(phrase)'\n")
                logSolution(phrase: phrase)
              }
            }
          }
        }
      }
    }
    print("")
  }

  return foundPhrases
}

// ==========================================
// PROGRAM ENTRY POINT
// ==========================================

do {
  // Initialize character mapping
  initializeCharacterMap()

  // Load dictionary
  let words = try loadDictionary()
  print("✓ Loaded \(words.count) unique words from dictionary")

  // Precompute hashes
  print("\n[Phase 1] Precomputing character hash table...")
  let htable = generateHashTable()
  print("  ✓ Character hash table ready")

  print("[Phase 2] Precomputing word hashes...")
  let (lenS, lenN, hashesSpace, hashesNoSpace) = precomputeWordHashes(words: words, htable: htable)
  print("  ✓ Word hashes precomputed")

  // Setup Metal
  print("\n[Phase 3] Setting up GPU environment...")
  let (device, queue, pipelines) = try setupMetalEnvironment(words: words)
  print("  ✓ Metal device initialized")
  print("  ✓ Compute pipelines created")

  // Create GPU buffers
  let mapBuffer = device.makeBuffer(
    length: GPUConstants.mapElements * 4,
    options: .storageModeShared)!
  let candsA = device.makeBuffer(
    length: GPUConstants.candidatesCapacity * 4,
    options: .storageModeShared)!
  let candsB = device.makeBuffer(
    length: GPUConstants.candidatesCapacity * 8,
    options: .storageModeShared)!
  let counter = device.makeBuffer(length: 4, options: .storageModeShared)!

  let bufLenS = device.makeBuffer(
    bytes: lenS, length: words.count * 4,
    options: .storageModeShared)!
  let bufLenN = device.makeBuffer(
    bytes: lenN, length: words.count * 4,
    options: .storageModeShared)!
  let bufHashS = device.makeBuffer(
    bytes: hashesSpace,
    length: words.count * MathConstants.hashTablePositions * 16,
    options: .storageModeShared)!
  let bufHashN = device.makeBuffer(
    bytes: hashesNoSpace,
    length: words.count * MathConstants.hashTablePositions * 16,
    options: .storageModeShared)!

  // Initialize hash map
  print("\n[Phase 4] Initializing GPU hash map...")
  // Fixed: Access `pipelines.initMap` instead of `pipelines.init`
  initializeHashMap(queue: queue, mapBuffer: mapBuffer, initPipe: pipelines.initMap)
  print("  ✓ Hash map initialized")

  // Build hash map
  print("[Phase 5] Building hash map from 2-word combinations...")
  buildHashMap(
    queue: queue, mapBuffer: mapBuffer, buildPipe: pipelines.build,
    wordCount: words.count, bufferLenS: bufLenS, bufferHashS: bufHashS)
  print("  ✓ Hash map built")

  // Execute sieve
  print("\n[Phase 6] Executing sieve search...\n")
  let gpuBuffers = (
    lenS: bufLenS, lenN: bufLenN, hashS: bufHashS, hashN: bufHashN,
    candsA: candsA, candsB: candsB, counter: counter
  )
  let results = executeSieve(
    queue: queue, mapBuffer: mapBuffer, sievePipe: pipelines.sieve,
    wordCount: words.count, buffers: gpuBuffers, words: words)

  print("\n✓ Attack Finished. Found \(results.count) unique solution(s).")

} catch {
  print("❌ Fatal error: \(error.localizedDescription)")
  exit(1)
}
