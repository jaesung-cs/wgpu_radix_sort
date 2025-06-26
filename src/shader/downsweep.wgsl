// See https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md for subgroups.
enable subgroups;

const RADIX = 256u;
const WORKGROUP_SIZE = 256u;
const PARTITION_DIVISION = 8u;
const PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
const MAX_SUBGROUP_SIZE = 128u;

@group(0) @binding(0) var<storage, read> elementCounts: array<u32>;
// TODO: read_write for pipeline layout. change to read.
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<u32>;
@group(0) @binding(2) var<storage, read_write> partitionHistogram: array<u32>;

@group(1) @binding(0) var<storage, read> keysIn: array<u32>;
@group(1) @binding(1) var<storage, read_write> keysOut: array<u32>;
#ifdef KEY_VALUE
@group(1) @binding(2) var<storage, read> valuesIn: array<u32>;
@group(1) @binding(3) var<storage, read_write> valuesOut: array<u32>;
#endif  // KEY_VALUE

@group(2) @binding(0) var<uniform> sortPass: u32;

var<workgroup> localHistogram: array<atomic<u32>, PARTITION_SIZE>;  // (R, S=8)=2048, (P) for alias. take maximum.
var<workgroup> localHistogramSum: array<u32, RADIX>;

// returns 0b00000....11111, where msb is id-1.
fn GetExclusiveWaveMask(id: u32) -> vec4<u32> {
    // clamp bit-shift right operand between 0..31 to avoid undefined behavior.
    let shift = (1 << extractBits(id, 0, 5)) - 1;  //  (1 << (id % 32)) - 1
    // right shift operation on signed integer copies sign bit, use the trick for masking.
    // (negative)     >> 31 = 111...111
    // (non-negative) >> 31 = 000...000
    let x = i32(id) >> 5;
    return vec4<u32>(u32((shift & ((-1 - x) >> 31)) | ((0 - x) >> 31)),
        u32((shift & ((0 - x) >> 31)) | ((1 - x) >> 31)),
        u32((shift & ((1 - x) >> 31)) | ((2 - x) >> 31)),
        u32((shift & ((2 - x) >> 31)) | ((3 - x) >> 31)));
}

fn GetBitCount(value: vec4<u32>) -> u32 {
    let result = countOneBits(value);
    return result[0] + result[1] + result[2] + result[3];
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) groupId: vec3<u32>,               // 0..P-1
    @builtin(local_invocation_index) groupIndex: u32,        // 0..W-1
    @builtin(subgroup_invocation_id) laneIndex: u32,         // 0..31 or 0..63
    @builtin(subgroup_size) laneCount: u32,                  // 32 or 64
) {
    let elementCount = elementCounts[0];

    let waveIndex = groupIndex / laneCount;         // 0..7 or 0..3
    let waveCount = WORKGROUP_SIZE / laneCount;     // 8 or 4
    let index = waveIndex * laneCount + laneIndex;  // 0..255

    let waveMask: vec4<u32> = GetExclusiveWaveMask(laneIndex);

    let partitionIndex = groupId.x;
    let partitionStart = partitionIndex * PARTITION_SIZE;

    if partitionStart >= elementCount {
        return;
    }

    for (var i = 0u; i < waveCount; i++) {
        atomicStore(&localHistogram[waveCount * index + i], 0);
    }
    workgroupBarrier();

    // load from global memory, local histogram and offset
    var localKeys: array<u32, PARTITION_DIVISION>;
    var localRadix: array<u32, PARTITION_DIVISION>;
    var localOffsets: array<u32, PARTITION_DIVISION>;
    var waveHistogram: array<u32, PARTITION_DIVISION>;
#ifdef KEY_VALUE
    var localValues: array<u32, PARTITION_DIVISION>;
#endif  // KEY_VALUE

    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let keyIndex = partitionStart + (PARTITION_DIVISION * laneCount) * waveIndex + i * laneCount + laneIndex;
        let key = select(0xffffffffu, keysIn[keyIndex], keyIndex < elementCount);
        localKeys[i] = key;

#ifdef KEY_VALUE
        localValues[i] = select(0, valuesIn[keyIndex], keyIndex < elementCount);
#endif  // KEY_VALUE

        let radix = extractBits(key, sortPass * 8, 8);
        localRadix[i] = radix;
        
        // mask per digit
        var mask = subgroupBallot(true);
        for (var j = 0u; j < 8; j++) {
            let digit = (radix >> j) & 1;
            let ballot = subgroupBallot(digit == 1);
            // digit - 1 is 0 or 0xffffffff. xor to flip.
            mask &= vec4<u32>(digit - 1) ^ ballot;
        }

        // wave level offset for radix
        let waveOffset = GetBitCount(waveMask & mask);
        let radixCount = GetBitCount(mask);
        
        // elect a representative per radix, add to histogram
        if waveOffset == 0 {
            // accumulate to local histogram
            atomicAdd(&localHistogram[waveCount * radix + waveIndex], radixCount);
            waveHistogram[i] = radixCount;
        } else {
            waveHistogram[i] = 0;
        }

        localOffsets[i] = waveOffset;
    }
    workgroupBarrier();

    // local histogram reduce 2048 or 1024
    for (var i = 0u; i < waveCount; i++) {
        let id = index + i * WORKGROUP_SIZE;
        let v = atomicLoad(&localHistogram[id]);
        let sum = subgroupAdd(v);
        let excl = subgroupExclusiveAdd(v);
        atomicStore(&localHistogram[id], excl);
        if laneIndex == 0 {
            localHistogramSum[id / laneCount] = sum;
        }
    }
    workgroupBarrier();
    
    // local histogram reduce 64 or 16
    let intermediateOffset0 = RADIX * waveCount / laneCount;
        {
        let v = select(0, localHistogramSum[index], index < intermediateOffset0);
        let sum = subgroupAdd(v);
        let excl = subgroupExclusiveAdd(v);
        if index < intermediateOffset0 {
            localHistogramSum[index] = excl;
        }
        if index < intermediateOffset0 && laneIndex == 0 {
            localHistogramSum[intermediateOffset0 + index / laneCount] = sum;
        }
    }
    workgroupBarrier();
    
    // local histogram reduce 2 or 1
    let intermediateSize1 = max(RADIX * waveCount / laneCount / laneCount, 1);
        {
        let v = localHistogramSum[(intermediateOffset0 + index) % RADIX];
        let excl = subgroupExclusiveAdd(v);
        if index < intermediateSize1 {
            localHistogramSum[intermediateOffset0 + index] = excl;
        }
    }
    workgroupBarrier();
    
    // local histogram add 128 or 32
    if index < intermediateOffset0 {
        localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / laneCount];
    }
    workgroupBarrier();
    
    // local histogram add 2048 or 1024
    for (var i = index; i < RADIX * waveCount; i += WORKGROUP_SIZE) {
        atomicAdd(&localHistogram[i], localHistogramSum[i / laneCount]);
    }
    workgroupBarrier();

    // post-scan stage
    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let radix = localRadix[i];
        localOffsets[i] += atomicLoad(&localHistogram[waveCount * radix + waveIndex]);

        workgroupBarrier();
        if waveHistogram[i] > 0 {
            atomicAdd(&localHistogram[waveCount * radix + waveIndex], waveHistogram[i]);
        }
        workgroupBarrier();
    }
    
    // after atomicAdd, localHistogram contains inclusive sum
        {
        let v = select(atomicLoad(&localHistogram[waveCount * index - 1]), 0, index == 0);
        localHistogramSum[index] = globalHistogram[RADIX * sortPass + index] + partitionHistogram[RADIX * partitionIndex + index] - v;
    }
    workgroupBarrier();
    
    // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge
    // speed boost. now localHistogram is unused, so alias memory.
    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let offset = localOffsets[i];
        atomicStore(&localHistogram[offset], localKeys[i]);
    }
    workgroupBarrier();
    
    // binning
    for (var i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
        let key = atomicLoad(&localHistogram[i]);
        let radix = extractBits(key, sortPass * 8, 8);
        let dstOffset = localHistogramSum[radix] + i;
        if dstOffset < elementCount {
            keysOut[dstOffset] = key;
        }
        
#ifdef KEY_VALUE
        localKeys[i / WORKGROUP_SIZE] = dstOffset;
#endif  // KEY_VALUE
    }
    
#ifdef KEY_VALUE
    workgroupBarrier();

    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let offset = localOffsets[i];
        atomicStore(&localHistogram[offset], localValues[i]);
    }
    workgroupBarrier();

    for (var i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
        let value = atomicLoad(&localHistogram[i]);
        let dstOffset = localKeys[i / WORKGROUP_SIZE];
        if dstOffset < elementCount {
            valuesOut[dstOffset] = value;
        }
    }
#endif  // KEY_VALUE
}
