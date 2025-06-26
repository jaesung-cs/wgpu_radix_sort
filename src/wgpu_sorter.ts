import upsweepShaderCode from './shader/upsweep.wgsl?raw';
import spineShaderCode from './shader/spine.wgsl?raw';
import downsweepShaderCode from './shader/downsweep.wgsl?raw';

const RADIX = 256;
const WORKGROUP_SIZE = 256;
const PARTITION_DIVISION = 8;
const PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

function roundUp(a: number, b: number) {
  return Math.ceil(a / b);
}


interface ShaderPreprocessOptions {
  code: string;
  defines?: { [key: string]: boolean };
}

function preprocessShader(options: ShaderPreprocessOptions) {
  return options.code.replace(/#ifdef\s+(\w+)[\s\S]*?#endif/g, (block, name) => {
    return options.defines && options.defines[name] ? block
      .replace(`#ifdef ${name}`, "")
      .replace("#endif", "") : "";
  });
}

export class WrdxSorter {
  private storageBindGroupLayout: GPUBindGroupLayout;
  private inoutBindGroupLayout: GPUBindGroupLayout;
  private uniformBindGroupLayout: GPUBindGroupLayout;
  private upsweepPipeline: GPUComputePipeline;
  private spinePipeline: GPUComputePipeline;
  private downsweepPipeline: GPUComputePipeline;
  private downsweepKeyValuePipeline: GPUComputePipeline;
  private storageBindGroup: GPUBindGroup;
  private uniformBindGroup: GPUBindGroup;

  // buffers
  private elementCounts: GPUBuffer;
  private globalHistogram: GPUBuffer;
  private partitionHistogram: GPUBuffer;
  private inout: GPUBuffer;
  private sortPass: GPUBuffer;
  private dummy: GPUBuffer;

  constructor(private readonly device: GPUDevice, private readonly maxElementCount: number) {
    const upsweepShaderModule = device.createShaderModule({ code: upsweepShaderCode });
    const spineShaderModule = device.createShaderModule({ code: spineShaderCode });
    const downsweepShaderModule = device.createShaderModule({ code: preprocessShader({ code: downsweepShaderCode }) });

    const downsweepKeyValueShaderCode = preprocessShader({
      code: downsweepShaderCode,
      defines: { KEY_VALUE: true },
    });
    const downsweepKeyValueShaderModule = device.createShaderModule({ code: downsweepKeyValueShaderCode });

    this.storageBindGroupLayout = device.createBindGroupLayout({
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
      }],
    });

    this.inoutBindGroupLayout = device.createBindGroupLayout({
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
        buffer: { type: "read-only-storage" },
      }, {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      }],
    });

    this.uniformBindGroupLayout = device.createBindGroupLayout({
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      }],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.storageBindGroupLayout, this.inoutBindGroupLayout, this.uniformBindGroupLayout]
    });

    this.upsweepPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: upsweepShaderModule,
        entryPoint: "main"
      }
    });

    this.spinePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: spineShaderModule,
        entryPoint: "main"
      }
    });

    this.downsweepPipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: downsweepShaderModule,
        entryPoint: "main"
      }
    });

    this.downsweepKeyValuePipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: downsweepKeyValueShaderModule,
        entryPoint: "main"
      }
    });

    this.elementCounts = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.globalHistogram = device.createBuffer({ size: 4 * RADIX * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.partitionHistogram = device.createBuffer({ size: roundUp(maxElementCount, PARTITION_SIZE) * RADIX * 4, usage: GPUBufferUsage.STORAGE });
    this.inout = device.createBuffer({ size: 2 * maxElementCount * 4, usage: GPUBufferUsage.STORAGE });
    this.sortPass = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummy = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });

    this.storageBindGroup = device.createBindGroup({
      layout: this.storageBindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.elementCounts },
      }, {
        binding: 1,
        resource: { buffer: this.globalHistogram },
      }, {
        binding: 2,
        resource: { buffer: this.partitionHistogram },
      }],
    });

    this.uniformBindGroup = device.createBindGroup({
      layout: this.uniformBindGroupLayout,
      entries: [{
        binding: 0,
        resource: { buffer: this.sortPass },
      }]
    });
  }

  destroy() {
    this.elementCounts.destroy();
    this.globalHistogram.destroy();
    this.partitionHistogram.destroy();
    this.inout.destroy();
    this.sortPass.destroy();
    this.dummy.destroy();
  }

  sortKeys(elementCount: number, keys: GPUBuffer) {
    this.sortImpl(elementCount, keys, undefined);
  }

  sortKeysIndirect(elementCount: GPUBuffer, keys: GPUBuffer) {
    this.sortImpl(elementCount, keys, undefined);
  }

  sortKeyValues(elementCount: number, keys: GPUBuffer, values: GPUBuffer) {
    this.sortImpl(elementCount, keys, values);
  }

  sortKeyValuesIndirect(elementCount: GPUBuffer, keys: GPUBuffer, values: GPUBuffer) {
    this.sortImpl(elementCount, keys, values);
  }

  private sortImpl(elementCount: GPUBuffer | number, keys: GPUBuffer, values: GPUBuffer | undefined) {
    const device = this.device;

    let partitionCount = roundUp(this.maxElementCount, PARTITION_SIZE);
    if (elementCount instanceof GPUBuffer) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(elementCount, this.elementCounts, 4);
      device.queue.submit([encoder.finish()]);
    } else {
      partitionCount = roundUp(elementCount, PARTITION_SIZE)
      const data = new Uint32Array([elementCount]);
      device.queue.writeBuffer(this.elementCounts, 0, data.buffer, data.byteOffset, data.byteLength);
    }

    let downsweepPipeline: GPUComputePipeline;
    if (values === undefined) {
      values = this.dummy;
      downsweepPipeline = this.downsweepPipeline;
    } else {
      downsweepPipeline = this.downsweepKeyValuePipeline;
    }

    const inoutBindGroup0 = device.createBindGroup({
      layout: this.inoutBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: keys } },
        { binding: 1, resource: { buffer: this.inout, size: this.maxElementCount * 4 } },
        { binding: 2, resource: { buffer: values } },
        { binding: 3, resource: { buffer: this.inout, offset: this.maxElementCount * 4 } },
      ]
    });

    const inoutBindGroup1 = device.createBindGroup({
      layout: this.inoutBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.inout, size: this.maxElementCount * 4 } },
        { binding: 1, resource: { buffer: keys } },
        { binding: 2, resource: { buffer: this.inout, offset: this.maxElementCount * 4 } },
        { binding: 3, resource: { buffer: values } },
      ]
    });

    const inoutBindGroups = [inoutBindGroup0, inoutBindGroup1];

    for (let pass = 0; pass < 4; pass++) {
      const data = new Uint32Array([pass]);
      device.queue.writeBuffer(this.sortPass, 0, data.buffer, data.byteOffset, data.byteLength);

      const inoutBindGroup = inoutBindGroups[pass % 2];

      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(this.globalHistogram, RADIX * pass * 4, RADIX * 4);

      const upsweepPass = encoder.beginComputePass();
      upsweepPass.setPipeline(this.upsweepPipeline);
      upsweepPass.setBindGroup(0, this.storageBindGroup);
      upsweepPass.setBindGroup(1, inoutBindGroup);
      upsweepPass.setBindGroup(2, this.uniformBindGroup);
      upsweepPass.dispatchWorkgroups(partitionCount);
      upsweepPass.end();

      const spinePass = encoder.beginComputePass();
      spinePass.setPipeline(this.spinePipeline);
      spinePass.setBindGroup(0, this.storageBindGroup);
      spinePass.setBindGroup(1, inoutBindGroup);
      spinePass.setBindGroup(2, this.uniformBindGroup);
      spinePass.dispatchWorkgroups(RADIX);
      spinePass.end();

      const downsweepPass = encoder.beginComputePass();
      downsweepPass.setPipeline(downsweepPipeline);
      downsweepPass.setBindGroup(0, this.storageBindGroup);
      downsweepPass.setBindGroup(1, inoutBindGroup);
      downsweepPass.setBindGroup(2, this.uniformBindGroup);
      downsweepPass.dispatchWorkgroups(partitionCount);
      downsweepPass.end();

      device.queue.submit([encoder.finish()]);
    }
  }
};
