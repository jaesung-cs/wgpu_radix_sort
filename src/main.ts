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

  const N = 1048576;
  const data = Uint32ArrayRandom(N);

  console.log("sorting: ", data);

  const keys = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  device.queue.writeBuffer(keys, 0, data.buffer, data.byteOffset, data.byteLength);

  wrdxSorter.sort(N, keys);

  const stage = device.createBuffer({ size: keys.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(keys, 0, stage, 0, keys.size);
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  await stage.mapAsync(GPUMapMode.READ);
  const keysArray = new Uint32Array(stage.getMappedRange());

  const result = arraysEqual(keysArray, data.slice().sort());
  console.log("result: ", result);
  if (!result) {
    console.log("keys:", keysArray);
    console.log("ans :", data.slice().sort());
  }

  stage.unmap();

  keys.destroy();
  stage.destroy();
}
