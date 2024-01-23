import numpy as np
from transformers import AutoTokenizer
import torch
import onnxruntime
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"]='0'
sentences = ["样例数据-1", "样例数据-2"]

device="cuda:3"

tokenizer = AutoTokenizer.from_pretrained("./bge-large-zh")
session = onnxruntime.InferenceSession("onnx/model.onnx", providers=['CUDAExecutionProvider'])
#session = onnxruntime.InferenceSession("onnx/model-large-fp32.onnx", providers=['CUDAExecutionProvider'])

cost_ms=[]
for i in range(1000):
    start = time.time() * 1000
    inputs = tokenizer(sentences, return_tensors="np")
    print(inputs)
    outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
    print(outputs)
    outputs = np.array(outputs)
    embeddings = torch.from_numpy(outputs)
    
    #sentence_embeddings = embeddings[0, :, 0, :]
    
    sentence_embeddings = embeddings.float()
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("sentence {}:".format(i), sentence_embeddings.shape, sentence_embeddings)
    end = time.time() * 1000
    cost_ms.append(end-start)
print("avg:", np.mean(cost_ms), ",p99:",np.percentile(cost_ms, 99))
