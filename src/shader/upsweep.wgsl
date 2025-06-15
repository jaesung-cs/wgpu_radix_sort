const RADIX = 256;
const WORKGROUP_SIZE = 256;
const PARTITION_DIVISION = 8;
const PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;
const MAX_SUBGROUP_SIZE = 128;

@binding(0) @group(0) var<storage, read> elementCounts: array<u32>;
@binding(1) @group(0) var<storage, read_write> globalHistogram: array<atomic<u32>>;
@binding(2) @group(0) var<storage, read_write> partitionHistogram: array<u32>;
@binding(3) @group(0) var<storage, read> keys: array<u32>;

@binding(0) @group(1) var<uniform> sortPass: u32;

var<workgroup> localHistogram: array<atomic<u32>, RADIX>;

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(local_invocation_id) groupThreadID: vec3<u32>,
    @builtin(workgroup_id) groupId: vec3<u32>,
) {
    let elementCount = elementCounts[0];

    let index = groupThreadID.x;
    let partitionIndex = groupId.x;
    let partitionStart = partitionIndex * PARTITION_SIZE;

    // discard all workgroup invocations
    if partitionStart >= elementCount {
        return;
    }

    atomicStore(&localHistogram[index], 0);
    workgroupBarrier();
  
    // local histogram
    for (var i = 0u; i < PARTITION_DIVISION; i++) {
        let keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
        let key = select(0xffffffff, keys[keyIndex], keyIndex < elementCount);
        let radix = extractBits(key, 8 * sortPass, 8u);
        atomicAdd(&localHistogram[radix], 1u);
    }
    workgroupBarrier();

    // set to partition histogram
    let v = atomicLoad(&localHistogram[index]);
    partitionHistogram[RADIX * partitionIndex + index] = v;

    // add to global histogram
    atomicAdd(&globalHistogram[RADIX * sortPass + index], v);
}
