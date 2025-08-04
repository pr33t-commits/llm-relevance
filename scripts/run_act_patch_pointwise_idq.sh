#!/bin/bash

set -e

data_path=data/exp_filtered.jsonl
model=meta-llama/Llama-3.1-8B-Instruct
model_path=../../model/Llama-3.1-8B-Instruct
nsamples=100

prompt="{inst}

Document: {document}

Query: {query}"


# for pos in "all" "last" "query_first" "query_last" "query_all" "doc_first" "doc_last" "doc_all" "inst_first" "inst_last" "inst_all"; do

# python main.py \
#   --model_name $model \
#   --model_path $model_path \
#   --method activation_patch \
#   --activation_name mlp_out \
#   --index_axis_names layer pos \
#   --patching_pos $pos \
#   --data $data_path \
#   --data_format pointwise \
#   --prompt_template "${prompt}" \
#   --nsamples $nsamples \
#   --output_path results/llama3.1--pointwise--mlp_out--layer-pos--$pos--prompt-idq

# done


# for pos in "all" "last" "query_first" "query_last" "query_all" "doc_first" "doc_last" "doc_all" "inst_first" "inst_last" "inst_all"; do

# python main.py \
#   --model_name $model \
#   --model_path $model_path \
#   --method activation_patch \
#   --activation_name attn_out \
#   --index_axis_names layer pos \
#   --patching_pos $pos \
#   --data $data_path \
#   --data_format pointwise \
#   --prompt_template "${prompt}" \
#   --nsamples $nsamples \
#   --output_path results/llama3.1--pointwise--attn_out--layer-pos--$pos--prompt-idq

# done



# python main.py \
#   --model_name $model \
#   --model_path $model_path \
#   --method activation_patch \
#   --activation_name z \
#   --index_axis_names layer head \
#   --data $data_path \
#   --data_format pointwise \
#   --prompt_template "${prompt}" \
#   --nsamples $nsamples \
#   --output_path results/llama3.1--pointwise--z--layer-head--prompt-idq


# for pos in "all" "last" "query_first" "query_last" "query_all" "doc_first" "doc_last" "doc_all" "inst_first" "inst_last" "inst_all"; do

# python main.py \
#   --model_name $model \
#   --model_path $model_path \
#   --method activation_patch \
#   --activation_name z \
#   --index_axis_names layer pos head \
#   --patching_pos $pos \
#   --data $data_path \
#   --data_format pointwise \
#   --prompt_template "${prompt}" \
#   --nsamples $nsamples \
#   --output_path results/llama3.1--pointwise--z--layer-pos-head--$pos--prompt-idq

# done


##################


for pos in "last doc_all" "last query_all" "query_all doc_all" "last inst_all" "inst_all query_all" "inst_all doc_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name attn_scores \
  --index_axis_names layer head dest_pos src_pos \
  --patching_pos $pos \
  --data $data_path \
  --data_format pointwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path "results/llama3.1--pointwise--attn_scores--layer-head-dest-src--$(echo $pos | tr ' ' '-')--prompt-idq"
  
done


# for pos in "all" "last" "query_first" "query_last" "query_all" "doc_first" "doc_last" "doc_all" "inst_first" "inst_last" "inst_all"; do

# python main.py \
#   --model_name $model \
#   --model_path $model_path \
#   --method activation_patch \
#   --activation_name q \
#   --index_axis_names layer pos head \
#   --patching_pos $pos \
#   --data $data_path \
#   --data_format pointwise \
#   --prompt_template "${prompt}" \
#   --nsamples $nsamples \
#   --output_path results/llama3.1--pointwise--q--layer-pos-head--$pos--prompt-idq

# done
