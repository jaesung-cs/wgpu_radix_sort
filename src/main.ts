import upsweepShaderCode from './shader/upsweep.wgsl?raw';
import spineShaderCode from './shader/spine.wgsl?raw';

const RADIX = 256;

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

  const upsweepShaderModule = device.createShaderModule({ code: upsweepShaderCode });
  console.log("upsweep shader module:", upsweepShaderModule);
  console.log(await upsweepShaderModule.getCompilationInfo());

  const spineShaderModule = device.createShaderModule({ code: spineShaderCode });
  console.log("spine shader module:", spineShaderModule);
  console.log(await spineShaderModule.getCompilationInfo());

  const elementCounts = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const globalHistogram = device.createBuffer({ size: 4096 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const partitionHistogram = device.createBuffer({ size: 4096 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const keys = device.createBuffer({ size: 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const sortPass = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  let data = new Uint32Array([0, 1, 3, 1]);
  device.queue.writeBuffer(keys, 0, data.buffer, data.byteOffset, data.byteLength);
  data = new Uint32Array([4]);
  device.queue.writeBuffer(elementCounts, 0, data.buffer, data.byteOffset, data.byteLength);
  data = new Uint32Array(4096);
  device.queue.writeBuffer(globalHistogram, 0, data.buffer, data.byteOffset, data.byteLength);
  device.queue.writeBuffer(partitionHistogram, 0, data.buffer, data.byteOffset, data.byteLength);
  data = new Uint32Array([0]);
  device.queue.writeBuffer(sortPass, 0, data.buffer, data.byteOffset, data.byteLength);

  const storageBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    }, {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    }, {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    }, {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    }],
  });

  const uniformBindGroupLayout = device.createBindGroupLayout({
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    }],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [storageBindGroupLayout, uniformBindGroupLayout]
  });

  const upsweepPipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: upsweepShaderModule,
      entryPoint: "main"
    }
  });

  const spinePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: spineShaderModule,
      entryPoint: "main"
    }
  });

  const storageBindGroup = device.createBindGroup({
    layout: storageBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: elementCounts },
    }, {
      binding: 1,
      resource: { buffer: globalHistogram },
    }, {
      binding: 2,
      resource: { buffer: partitionHistogram },
    }, {
      binding: 3,
      resource: { buffer: keys },
    }]
  });

  const uniformBindGroup = device.createBindGroup({
    layout: uniformBindGroupLayout,
    entries: [{
      binding: 0,
      resource: { buffer: sortPass },
    }]
  });

  const encoder = device.createCommandEncoder();
  const upsweepPass = encoder.beginComputePass();
  upsweepPass.setPipeline(upsweepPipeline);
  upsweepPass.setBindGroup(0, storageBindGroup);
  upsweepPass.setBindGroup(1, uniformBindGroup);
  upsweepPass.dispatchWorkgroups(1);
  upsweepPass.end();

  const spinePass = encoder.beginComputePass();
  spinePass.setPipeline(spinePipeline);
  spinePass.setBindGroup(0, storageBindGroup);
  spinePass.setBindGroup(1, uniformBindGroup);
  spinePass.dispatchWorkgroups(RADIX);
  spinePass.end();

  device.queue.submit([encoder.finish()]);

  const stage0 = device.createBuffer({ size: 4096 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const stage1 = device.createBuffer({ size: 4096 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    globalHistogram, 0,
    stage0, 0,
    4096 * 4
  );
  commandEncoder.copyBufferToBuffer(
    partitionHistogram, 0,
    stage1, 0,
    4096 * 4
  );
  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);

  await stage0.mapAsync(GPUMapMode.READ);
  const globalHistogramArray = new Uint32Array(stage0.getMappedRange());
  await stage1.mapAsync(GPUMapMode.READ);
  const partitionHistogramArray = new Uint32Array(stage1.getMappedRange());
  console.log("globalHistogram   :", globalHistogramArray);
  console.log("partitionHistogram:", partitionHistogramArray);
  stage0.unmap();
  stage1.unmap();
}
