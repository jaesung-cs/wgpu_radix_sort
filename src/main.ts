import { WrdxSorter } from './wgpu_sorter';

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

  const wrdxSorter = new WrdxSorter(device);

  const N = 10;
  const data = new Uint32Array(N);
  crypto.getRandomValues(data);

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
  console.log("keys:", keysArray);
  console.log("ans :", data.sort());
  stage.unmap();

  keys.destroy();
  stage.destroy();
}
