from transformers import AutoTokenizer, AutoModel
import torch
import time
import os
import numpy as np
# Sentences we want sentence embeddings for

sentences = ["样例数据-1", "样例数据-2"]

os.environ["CUDA_VISIBLE_DEVICES"]='0'

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('./bge-large-zh')
model = AutoModel.from_pretrained('./bge-large-zh').cuda()

encoded_input = tokenizer(sentences, padding='max_length', max_length=8, truncation=True, return_tensors='pt').to("cuda:0")

cost_ms=[]
for i in range(1):
    # Tokenize sentences
    start = time.time()*1000
    encoded_input = tokenizer(sentences, padding='max_length', max_length=8, truncation=True, return_tensors='pt').to("cuda:0")
    
    # 对于短查询到长文档的检索任务, 为查询加上指令
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')
    torch.set_printoptions(threshold=float('inf'))
   
    # Compute embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
        print(sentence_embeddings)
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu()
    end = time.time()*1000
    cost_ms.append(end-start)
    print("sentence {}:".format(i), sentence_embeddings.shape, sentence_embeddings)
print("avg:", np.mean(cost_ms), ",p99:",np.percentile(cost_ms, 99))
