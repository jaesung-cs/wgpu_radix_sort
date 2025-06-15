// See https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md for subgroups.
enable subgroups;

const RADIX = 256u;
const WORKGROUP_SIZE = 256u;
const PARTITION_DIVISION = 8u;
const PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
const MAX_SUBGROUP_SIZE = 128u;

@binding(0) @group(0) var<storage, read> elementCounts: array<u32>;
// TODO: read_write for pipeline layout. change to read.
@binding(1) @group(0) var<storage, read_write> globalHistogram: array<u32>;
@binding(2) @group(0) var<storage, read_write> partitionHistogram: array<u32>;
@binding(3) @group(0) var<storage, read> keysIn: array<u32>;
@binding(4) @group(0) var<storage, read_write> keysOut: array<u32>;

@binding(0) @group(1) var<uniform> sortPass: u32;

var<workgroup> localHistogram: array<atomic<u32>, PARTITION_SIZE>;  // (R, S=16)=4096, (P) for alias. take maximum.
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
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) groupThreadID: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) groupIndex: u32,
    @builtin(subgroup_invocation_id) laneIndex: u32,  // 0..31 or 0..63
    @builtin(subgroup_size) laneCount: u32,           // 32 or 64
) {
    let elementCount = elementCounts[0];

    let waveIndex = groupIndex / laneCount;         // 0..15 or 0..7
    let waveCount = WORKGROUP_SIZE / laneCount;     // 16 or 8
    let index = waveIndex * laneCount + laneIndex;  // 0..255

    let waveMask: vec4<u32> = GetExclusiveWaveMask(laneIndex);

    let partitionIndex = groupId.x;
    let partitionStart = partitionIndex * PARTITION_SIZE;

    if partitionStart >= elementCount {
        return;
    }

    if index < RADIX {
        for (var i = 0u; i < waveCount; i++) {
            atomicStore(&localHistogram[waveCount * index + i], 0);
        }
    }
    workgroupBarrier();

    // load from global memory, local histogram and offset
    var localKeys: array<u32, PARTITION_DIVISION>;
    var localRadix: array<u32, PARTITION_DIVISION>;
    var localOffsets: array<u32, PARTITION_DIVISION>;
    var waveHistogram: array<u32, PARTITION_DIVISION>;

    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let keyIndex = partitionStart + (PARTITION_DIVISION * laneCount) * waveIndex + i * laneCount + laneIndex;
        let key = select(0xffffffffu, keysIn[keyIndex], keyIndex < elementCount);
        localKeys[i] = key;

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

    // local histogram reduce 4096 or 2048
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
    
    // local histogram reduce 128 or 32
    let intermediateOffset0 = RADIX * waveCount / laneCount;
        {
        let v = localHistogramSum[index % intermediateOffset0];
        let sum = subgroupAdd(v);
        let excl = subgroupExclusiveAdd(v);
        if index < intermediateOffset0 {
            localHistogramSum[index] = excl;
        }
        if laneIndex == 0 {
            localHistogramSum[intermediateOffset0 + index / laneCount] = sum;
        }
    }
    workgroupBarrier();
    
    // local histogram reduce 4 or 1
    let intermediateSize1 = max(RADIX * waveCount / laneCount / laneCount, 1);
        {
        let v = localHistogramSum[(intermediateOffset0 + index) % RADIX];
        let excl = subgroupExclusiveAdd(v);
        if index < intermediateSize1 {
            localHistogramSum[intermediateOffset0 + index] = excl;
        }
    }
    workgroupBarrier();
    
    // local histogram add 128
    if index < intermediateOffset0 {
        localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / laneCount];
    }
    workgroupBarrier();
    
    // local histogram add 4096
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
    if index < RADIX {
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
    }
}
