#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import sys
import argparse
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import random

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

#TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

class DeviceBuffer(object):
    def __init__(self, shape, dtype=trt.int32):
        self.buf = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

    def binding(self):
        return int(self.buf)

    def free(self):
        self.buf.free()

def load_inputs(seq_len, batch_size=1):
    batch_input_ids = []
    batch_input_masks = []
    batch_segment_ids = []
    for _ in range(batch_size):
        batch_input_ids.append([ 101, 3416,  891, 3144, 2945,  118,  122,  102])
        batch_input_masks.append([1,1,1,1,1,1,1,1])
        batch_segment_ids.append([0,0,0,0,0,0,0,0])

    np_input_ids = np.array(batch_input_ids, dtype=np.int32)
    np_mask_ids = np.array(batch_input_masks, dtype=np.int32)
    np_segment_ids = np.array(batch_segment_ids, dtype=np.int32)
    return np_input_ids, np_segment_ids, np_mask_ids

def main():
    parser = argparse.ArgumentParser(description='BERT Inference Benchmark')
    parser.add_argument("-e", "--engine", default="../engines/bge_b10_s8_fixseq_fp16_a10_cu117_trt86.engine", help='Path to BERT TensorRT engine')
    #parser.add_argument("-e", "--engine", default="../engines/bge_b10_s8_fixseq_a10_cu117_trt86.engine", help='Path to BERT TensorRT engine')
    #parser.add_argument("-e", "--engine", default="../engines/bge_a10_cu117_trt86.engine", help='Path to BERT TensorRT engine')
    #parser.add_argument("-e", "--engine", default="./model-large-fp32.engine", help='Path to BERT TensorRT engine')
    parser.add_argument('-b', '--batch-size', default=[2], action="append", help='Batch size(s) to benchmark. Can be specified multiple times for more than one batch size. This script assumes that the engine has been built with one optimization profile for each batch size, and that these profiles are in order of increasing batch size.', type=int)
    parser.add_argument('-s', '--sequence-length', default=8, help='Sequence length of the BERT model', type=int)
    parser.add_argument('-i', '--iterations', default=1, help='Number of iterations to run when benchmarking each batch size.', type=int)
    parser.add_argument('-w', '--warm-up-runs', default=10, help='Number of iterations to run prior to benchmarking.', type=int)
    parser.add_argument('-r', '--random-seed', required=False, default=12345, help='Random seed.', type=int)
    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        # Allocate buffers large enough to store the largest batch size
        max_input_shape = (max(args.batch_size), args.sequence_length)
        #max_output_shape = (max(args.batch_size), 128)
        #max_output_shape = (32, 768)
        #max_output_shape = (1, 768)
        #max_output_shape = (args.sequence_length, max(args.batch_size), 120)
        max_output_shape = (max(args.batch_size), 1024)
        buffers = [
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            #DeviceBuffer(max_output_shape)
            DeviceBuffer(max_output_shape, dtype=trt.float32)
        ]

        # Prepare random input
        pseudo_vocab_size = 30522
        pseudo_type_vocab_size = 2
        #np.random.seed(args.random_seed)
        #test_word_ids = np.random.randint(0, pseudo_vocab_size, (max(args.batch_size), args.sequence_length), dtype=np.int32)
        #test_segment_ids = np.random.randint(0, pseudo_type_vocab_size, (max(args.batch_size), args.sequence_length), dtype=np.int32)
        #test_input_mask = np.ones((max(args.batch_size), args.sequence_length), dtype=np.int32)

        #test_word_ids, test_segment_ids, test_input_mask = load_inputs_from_file("./rele_mtt_case.txt.pre1", max(args.batch_size))
        test_word_ids, test_segment_ids, test_input_mask = load_inputs(args.sequence_length, max(args.batch_size))
        print("test_word_ids shape: {}, value: {}".format(test_word_ids.shape, test_word_ids))
        #print("test_segment_ids shape: {}, value: {}".format(test_segment_ids.shape, test_segment_ids))
        #print("test_input_mask shape: {}, value: {}".format(test_input_mask.shape, test_input_mask))

        # Copy input h2d
        cuda.memcpy_htod(buffers[0].buf, test_word_ids.ravel())
        cuda.memcpy_htod(buffers[1].buf, test_segment_ids.ravel())
        cuda.memcpy_htod(buffers[2].buf, test_input_mask.ravel())
        print("input_ids:", test_word_ids)
        print("segment_ids:", test_segment_ids)
        print("input_mask", test_input_mask)

        h_output = cuda.pagelocked_empty(max_output_shape, dtype=np.float32)

        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

        print(num_binding_per_profile, engine.num_bindings)
        bench_times = {}

        stream = cuda.Stream()
        for idx, batch_size in enumerate(sorted(args.batch_size)):
            context.set_optimization_profile_async(idx, stream.handle)

            # Each profile has unique bindings
            binding_idx_offset = idx * num_binding_per_profile
            bindings = [0] * binding_idx_offset + [buf.binding() for buf in buffers]
            print("len bindings: {}".format(len(bindings)))

            shapes = {
                "input_ids": (batch_size, args.sequence_length),
                "segment_ids": (batch_size, args.sequence_length),
                "input_mask": (batch_size, args.sequence_length),
            }
            print(shapes)

            for binding, shape in shapes.items():
                context.set_binding_shape(engine[binding] + binding_idx_offset, shape)
            assert context.all_binding_shapes_specified

            print(binding, shape)
            # Inference
            total_time = 0
            start = cuda.Event()
            end = cuda.Event()

            # Warmup
            for _ in range(args.warm_up_runs):
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()
            

            num_inputs = engine.num_bindings

            # 遍历所有的输入和输出
            for i in range(num_inputs):
                # 获取字段名称
                name = engine.get_binding_name(i)
                # 获取字段类型
                dtype = engine.get_binding_dtype(i)
                # 获取字段维度
                shape = engine.get_binding_shape(i)
                # 打印字段名称、类型和维度
                print(f"Name: {name}")
                print(f"Type: {dtype}")
                print(f"Shape: {shape}")


            # Timing loop
            times = []
            for _ in range(args.iterations):
                start.record(stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, buffers[3].buf, stream)
                end.record(stream)
                stream.synchronize()
                times.append(end.time_since(start))
                print("output shape: {}\n {}".format(h_output.shape, h_output))
            # Compute average time, 95th percentile time and 99th percentile time.
            bench_times[batch_size] = times

        [b.free() for b in buffers]

        outfd = open("./perf.txt", 'a')

        for batch_size, times in bench_times.items():
            total_time = sum(times)
            avg_time = total_time / float(len(times))
            times.sort()
            percentile95 = times[int(len(times) * 0.95)]
            percentile99 = times[int(len(times) * 0.99)]
            msg = "Running {:} iterations with Batch Size: {:}\n\tTotal Time: {:} ms \tAverage Time: {:} ms\t"\
                  "95th Percentile Time: {:} ms\t99th Percentile Time: {:}".format(
                  args.iterations, batch_size, total_time, avg_time, percentile95, percentile99)
            outfd.write("\n" + msg + "\n\n")
            print(msg)


if __name__ == '__main__':
    main()

