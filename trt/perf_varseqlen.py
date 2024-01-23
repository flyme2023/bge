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

import argparse
import ctypes
from types import DynamicClassAttribute
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
from torch._C import dtype

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class DeviceBuffer(object):
    def __init__(self, shape, dtype=trt.int32):
        self.buf = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

    def binding(self):
        return int(self.buf)

    def free(self):
        self.buf.free()


def load_inputs_from_file(input_file_name):
    data = dict()
    with open(input_file_name, 'r') as input_file:
        lines = input_file.readlines()
        data['word_ids'] = list(map(int, lines[0].split()))
        data['segment_ids'] = list(map(int, lines[1].split()))
        data['cu_seq_lens'] = list(map(int, lines[2].split()))

    return np.array(data['word_ids'], dtype=np.int32), np.array(data['segment_ids'], dtype=np.int32), np.array(data['cu_seq_lens'], dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description='BERT Inference Benchmark')
    parser.add_argument("-e", "--engine", help='Path to BERT TensorRT engine')
    parser.add_argument('-b', '--batch-size', default=[], action="append", help='Batch size(s) to benchmark. Can be specified multiple times for more than one batch size. This script assumes that the engine has been built with one optimization profile for each batch size, and that these profiles are in order of increasing batch size.', type=int)
    parser.add_argument('-s', '--sequence-length', default=128, help='Sequence length of the BERT model', type=int)
    parser.add_argument('-i', '--iterations', default=200, help='Number of iterations to run when benchmarking each batch size.', type=int)
    parser.add_argument('-w', '--warm-up-runs', default=10, help='Number of iterations to run prior to benchmarking.', type=int)
    parser.add_argument('-r', '--random-seed', required=False, default=12345, help='Random seed.', type=int)
    parser.add_argument('-f', '--input-file', required=True, help='Input file name')
    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:
        # Allocate buffers large enough to store the largest batch size
        max_input_shape = (args.sequence_length * max(args.batch_size), )
        max_output_shape = (max(args.batch_size), 5)
        # buffers = [
        #     DeviceBuffer(max_input_shape),
        #     DeviceBuffer(max_input_shape),
        #     DeviceBuffer((max(args.batch_size) + 1, )),
        #     DeviceBuffer((args.sequence_length, )),
        #     DeviceBuffer(max_output_shape)
        # ]

        # # Prepare random input
        # pseudo_vocab_size = 30522
        # pseudo_type_vocab_size = 2
        # np.random.seed(args.random_seed)
        # test_word_ids = np.random.randint(0, pseudo_vocab_size, (args.sequence_length * max(args.batch_size)), dtype=np.int32)
        # test_segment_ids = np.random.randint(0, pseudo_type_vocab_size, (args.sequence_length * max(args.batch_size)), dtype=np.int32)
        # test_cu_seq_lens = np.arange(0, args.sequence_length * max(args.batch_size) + 1, args.sequence_length, dtype=np.int32)

        test_word_ids, test_segment_ids, test_cu_seq_lens = load_inputs_from_file(args.input_file)

        buffers = [
            DeviceBuffer(test_word_ids.shape),
            DeviceBuffer(test_word_ids.shape),
            DeviceBuffer((max(args.batch_size) + 1, )),
            DeviceBuffer((args.sequence_length, )),
            DeviceBuffer(max_output_shape)
        ]
        # Copy input h2d
        cuda.memcpy_htod(buffers[0].buf, cuda.register_host_memory(np.ascontiguousarray(test_word_ids.ravel())))
        cuda.memcpy_htod(buffers[1].buf, cuda.register_host_memory(np.ascontiguousarray(test_segment_ids.ravel())))
        cuda.memcpy_htod(buffers[2].buf, cuda.register_host_memory(np.ascontiguousarray(test_cu_seq_lens.ravel())))

        bench_times = {}
        h_output = cuda.pagelocked_empty((max(args.batch_size), 5), dtype=np.float32)
        # d_output = cuda.mem_alloc(h_output.nbytes)

        for idx, batch_size in enumerate(sorted(args.batch_size)):
            context.active_optimization_profile = 0

            # Each profile has unique bindings
            bindings = [buf.binding() for buf in buffers]

            shapes = {
                "input_ids": test_word_ids.shape,
                "segment_ids": test_segment_ids.shape,
                "cu_seqlens": test_cu_seq_lens.shape,
                "max_seqlen": (args.sequence_length, ),
            }

            for binding, shape in shapes.items():
                context.set_binding_shape(engine[binding], shape)
            assert context.all_binding_shapes_specified

            # Inference
            total_time = 0
            start = cuda.Event()
            end = cuda.Event()
            stream = cuda.Stream()

            # Warmup
            for i in range(args.warm_up_runs):
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                cuda.memcpy_dtoh_async(h_output, buffers[4].buf, stream)
                stream.synchronize()
                if (i == 0):
                    with np.printoptions(formatter={'float': '{: 0.5f}'.format}, suppress=True, linewidth=np.inf):
                        print(h_output.ravel())

            # Timing loop
            times = []
            for _ in range(args.iterations):
                start.record(stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                end.record(stream)
                stream.synchronize()
                times.append(end.time_since(start))

            # Compute average time, 95th percentile time and 99th percentile time.
            bench_times[batch_size] = times

        [b.free() for b in buffers]

        for batch_size, times in bench_times.items():
            total_time = sum(times)
            avg_time = total_time / float(len(times))
            times.sort()
            percentile95 = times[int(len(times) * 0.95)]
            percentile99 = times[int(len(times) * 0.99)]
            print("Running {:} iterations with Batch Size: {:}\n\tTotal Time: {:} ms \tAverage Time: {:} ms\t95th Percentile Time: {:} ms\t99th Percentile Time: {:}".format(args.iterations, batch_size, total_time, avg_time, percentile95, percentile99))


if __name__ == '__main__':
    main()
