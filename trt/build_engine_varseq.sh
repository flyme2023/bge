gpu="a10"
precision="int8"

v="v7"
model_dir=../models//bge-large-zh/
qat_model_dir_skiplnfp16=$model_dir/pytorch_model.bin
batchsize=5
seqlen=32
output_engine_name=../engines/bge_b${batchsize}_s${seqlen}_varseq_fp16_a10_cu117_trt86.engine
#output_engine_name=../engines/bge_a10_cu117_trt86.engine
mkdir -p ../engines; PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python3.8 builder_varseqlen.py \
  -pt ${qat_model_dir_skiplnfp16} \
  -o ${output_engine_name} \
  -b $batchsize \
  -s ${seqlen} \
  -c ${model_dir} \
  -v ${model_dir}/vocab.txt \
  --fp16
