import { WrdxSorter } from './wgpu_sorter';

function arraysEqual(a: Uint32Array, b: Uint32Array) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function Uint32ArrayRandom(N: number) {
  const chunkSize = 16384;
  const result = new Uint32Array(N);

  for (let offset = 0; offset < N; offset += chunkSize) {
    const count = Math.min(chunkSize, N - offset);
    const slice = new Uint32Array(result.buffer, offset * 4, count);
    crypto.getRandomValues(slice);
  }

  return result;
}

async function testSortKeys(device: GPUDevice, sorter: WrdxSorter, keys: Uint32Array) {
  const N = keys.length;
  console.log(`test sort keys ${N}`);

  const keysBuffer = device.createBuffer({ size: keys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(keysBuffer, 0, keys.buffer, keys.byteOffset, keys.byteLength);

  sorter.sort(N, keysBuffer);

  const stage = device.createBuffer({ size: keysBuffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(keysBuffer, 0, stage, 0, keysBuffer.size);
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  await stage.mapAsync(GPUMapMode.READ);
  const keysArray = new Uint32Array(stage.getMappedRange());

  const result = arraysEqual(keysArray, keys.slice().sort());
  console.log("result: ", result);
  if (!result) {
    console.log("keys:", keysArray);
    console.log("ans :", keys.slice().sort());
  }

  stage.unmap();

  keysBuffer.destroy();
  stage.destroy();
}

async function testSortKeyValues(device: GPUDevice, sorter: WrdxSorter, keys: Uint32Array, values: Uint32Array) {
  const N = keys.length;
  console.log(`test sort key values ${N}`);

  const keysBuffer = device.createBuffer({ size: keys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const valuesBuffer = device.createBuffer({ size: keys.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(keysBuffer, 0, keys.buffer, keys.byteOffset, keys.byteLength);
  device.queue.writeBuffer(valuesBuffer, 0, values.buffer, values.byteOffset, values.byteLength);

  sorter.sortKeyValues(N, keysBuffer, valuesBuffer);

  const stage = device.createBuffer({ size: keysBuffer.size * 2, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(keysBuffer, 0, stage, 0, keysBuffer.size);
  commandEncoder.copyBufferToBuffer(valuesBuffer, 0, stage, keysBuffer.size, valuesBuffer.size);
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  await stage.mapAsync(GPUMapMode.READ);
  const keysArray = new Uint32Array(stage.getMappedRange(0, keysBuffer.size));
  const valuesArray = new Uint32Array(stage.getMappedRange(keysBuffer.size, valuesBuffer.size));

  const indices = Uint32Array.from(keys.keys());
  indices.sort((i, j) => keys[i] - keys[j]);
  const sortedKeys = new Uint32Array(indices.map(i => keys[i]));
  const sortedValues = new Uint32Array(indices.map(i => values[i]));

  const resultKeys = arraysEqual(keysArray, sortedKeys);
  console.log("result (keys): ", resultKeys);
  if (!resultKeys) {
    console.log("keys:", keysArray);
    console.log("ans :", sortedKeys);
  }
  const resultValues = arraysEqual(valuesArray, sortedValues);
  console.log("result (vals): ", resultValues);
  if (!resultValues) {
    console.log("vals:", valuesArray);
    console.log("ans :", sortedValues);
  }

  stage.unmap();

  keysBuffer.destroy();
  valuesBuffer.destroy();
  stage.destroy();
}

export default async function start() {
  const adapter = await navigator.gpu.requestAdapter();
  const features = adapter!.features;
  if (features.has("subgroups")) {
    console.log("Subgroup operations supported!");
  }

  const device = await adapter?.requestDevice({
    requiredFeatures: ["subgroups"],
  });
  if (!device) throw new Error("WebGPU not supported");

  console.log("Adaptor:", adapter);
  console.log("Device:", device);

  const MAX_N = 1048576;
  const wrdxSorter = new WrdxSorter(device, MAX_N);

  {
    const N = 1048576;
    const keys = Uint32ArrayRandom(N);
    await testSortKeys(device, wrdxSorter, keys);
  }
  {
    const N = 1048576;
    const keys = Uint32ArrayRandom(N);
    const values = Uint32ArrayRandom(N);
    await testSortKeyValues(device, wrdxSorter, keys, values);
  }
}
