#!/bin/bash

set -e

data_path=data/exp_filtered.jsonl
model=meta-llama/Meta-Llama-3-8B-Instruct ## meta-llama/Llama-3.1-8B-Instruct NOTE -change in name format
model_path=../../model/Meta-Llama-3-8B-Instruct
nsamples=100


##### pointwise

prompt="Document: {document}

Query: {query}

{inst}"


for act in "mlp_out" "attn_out"; do
for pos in "all" "last" "query_all" "doc_all" "inst_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name $act \
  --index_axis_names layer pos \
  --patching_pos $pos \
  --data $data_path \
  --data_format pointwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path results/llama3--pointwise--$act--layer-pos--$pos--prompt-dqi

done
done


for pos in "all" "last" "query_all" "doc_all" "inst_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name z \
  --index_axis_names layer pos head \
  --patching_pos $pos \
  --data $data_path \
  --data_format pointwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path results/llama3--pointwise--z--layer-pos-head--$pos--prompt-dqi

done


for pos in "query_all doc_all" "last inst_all" "inst_all query_all"; do

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
  --output_path "results/llama3--pointwise--attn_scores--layer-head-dest-src--$(echo $pos | tr ' ' '-')--prompt-dqi"
  
done



##### pairwise

prompt="Document A: {document1}

Document B: {document2}

Query: {query}

{inst}"


for act in "mlp_out" "attn_out"; do
for pos in "all" "last" "query_all" "doc_all" "inst_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name $act \
  --index_axis_names layer pos \
  --patching_pos $pos \
  --data $data_path \
  --data_format pairwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path results/llama3--pairwise--$act--layer-pos--$pos--prompt-dqi

done
done


for pos in "all" "last" "query_all" "doc_all" "inst_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name z \
  --index_axis_names layer pos head \
  --patching_pos $pos \
  --data $data_path \
  --data_format pairwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path results/llama3--pairwise--z--layer-pos-head--$pos--prompt-dqi

done


for pos in "query_all doc_all" "last inst_all" "inst_all query_all"; do

python main.py \
  --model_name $model \
  --model_path $model_path \
  --method activation_patch \
  --activation_name attn_scores \
  --index_axis_names layer head dest_pos src_pos \
  --patching_pos $pos \
  --data $data_path \
  --data_format pairwise \
  --prompt_template "${prompt}" \
  --nsamples $nsamples \
  --output_path "results/llama3--pairwise--attn_scores--layer-head-dest-src--$(echo $pos | tr ' ' '-')--prompt-dqi"
  
done

