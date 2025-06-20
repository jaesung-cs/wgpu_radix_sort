// See https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md for subgroups.
enable subgroups;

const RADIX = 256;
const WORKGROUP_SIZE = 256;
const PARTITION_DIVISION = 8;
const PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
const MAX_SUBGROUP_SIZE = 128;

@group(0) @binding(0) var<storage, read> elementCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHistogram: array<u32>;
@group(0) @binding(2) var<storage, read_write> partitionHistogram: array<u32>;

@group(2) @binding(0) var<uniform> sortPass: u32;

var<workgroup> reduction: u32;
var<workgroup> intermediate: array<u32, MAX_SUBGROUP_SIZE>;

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) groupThreadID: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
    @builtin(local_invocation_index) groupIndex: u32,
    @builtin(subgroup_invocation_id) laneIndex: u32,  // 0..31
    @builtin(subgroup_size) laneCount: u32,           // 32
) {
    let elementCount = elementCounts[0];

    let waveIndex = groupIndex / laneCount;         // 0..7
    let waveCount = WORKGROUP_SIZE / laneCount;     // 8
    let index = waveIndex * laneCount + laneIndex;  // 0..255

    let radix = groupId.x;

    let partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

    if index == 0 {
        reduction = 0;
    }
    workgroupBarrier();

    for (var i = 0u; WORKGROUP_SIZE * i < partitionCount; i++) {
        let partitionIndex = WORKGROUP_SIZE * i + index;
        let value = select(0, partitionHistogram[RADIX * (partitionIndex % partitionCount) + radix], partitionIndex < partitionCount);
        var excl = subgroupExclusiveAdd(value) + reduction;
        let sum = subgroupAdd(value);

        if subgroupElect() {
            intermediate[waveIndex] = sum;
        }
        workgroupBarrier();

            {
            let value = select(0, intermediate[index % MAX_SUBGROUP_SIZE], index < waveCount);
            let excl = subgroupExclusiveAdd(value);
            let sum = subgroupAdd(value);
            if index < waveCount {
                intermediate[index] = excl;

                if index == 0 {
                    reduction += sum;
                }
            }
        }
        workgroupBarrier();

        if partitionIndex < partitionCount {
            excl += intermediate[waveIndex];
            partitionHistogram[RADIX * partitionIndex + radix] = excl;
        }
        workgroupBarrier();
    }

    if radix == 0 {
        // one workgroup is responsible for global histogram prefix sum
        let value = globalHistogram[RADIX * sortPass + index];
        var excl = subgroupExclusiveAdd(value);
        let sum = subgroupAdd(value);

        if subgroupElect() {
            intermediate[waveIndex] = sum;
        }
        workgroupBarrier();

            {
            let excl = subgroupExclusiveAdd(intermediate[index % MAX_SUBGROUP_SIZE]);
            if index < RADIX / laneCount {
                intermediate[index] = excl;
            }
        }
        workgroupBarrier();

        excl += intermediate[waveIndex];
        globalHistogram[RADIX * sortPass + index] = excl;
    }
}
