gpu="a10"
precision="fp32"

model_dir=../models//bge-large-zh/
qat_model_dir_skiplnfp16=$model_dir/pytorch_model.bin
batchsize=10
seqlen=500
output_engine_name=../engines/bge_b${batchsize}_s${seqlen}_fixseq_fp32_a10_cu117_trt86.engine
mkdir -p ../engines; PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3.8 builder.py \
  -pt ${qat_model_dir_skiplnfp16} \
  -o ${output_engine_name} \
  -b $batchsize \
  -s ${seqlen} \
  -c ${model_dir} \
  -v ${model_dir}/vocab.txt
