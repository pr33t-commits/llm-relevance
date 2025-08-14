import argparse
import datetime
import json
import sys
import os
import random
import torch
import numpy as np
import pandas as pd

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig

from data_utils import get_loaders, truncate_to_equal_length
from methods import activation_patch, path_patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Llama model") # 'mistralai/Mistral-7B-Instruct-v0.1'
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

    parser.add_argument("--method", type=str, default=None, choices=[
        "activation_patch",
        "path_patch",
        "mean_ablation",
        "zero_ablation",
    ], help="Method used for the experiment.")
    parser.add_argument("--activation_name", type=str, default="z", help="Name of the activation to be used for the experiment.")
    parser.add_argument("--index_axis_names", nargs="+", default=["layer", "head"], help="Names of the axes to be used for the experiment.")
    parser.add_argument("--patching_pos", nargs="*", default=None, help="Position for patching.")
    parser.add_argument("--receiver_activation_name", type=str, default="q", help="Receiver activation name.")
    parser.add_argument("--receiver_layer_or_heads", type=eval, default=None, help="Receiver heads to be used in path patching.")
    parser.add_argument("--receiver_pos", type=str, default=None, help="Receiver position to be used in path patching.")

    parser.add_argument("--data", type=str, default="data/exp.jsonl", help="Path to the data.")
    parser.add_argument("--nsamples", type=int, default=1, help="Number of calibration samples.")
    parser.add_argument("--prompt_template", type=str, default=None, help="Prompt template for data.")
    parser.add_argument("--data_format", type=str, default=None, help="Data format")

    parser.add_argument("--output_path", type=str, default=None, help="Path to save the results.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output path if it exists.")

    args = parser.parse_args()
    print(args)

    # Setting seeds for reproducibility
    set_seed(args.seed)

    if args.output_path is None:
        args.output_path = os.path.join("results", f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    if os.path.exists(args.output_path) and not args.overwrite:
        print(f"Output path {args.output_path} already exists")
        return

    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"loading llm: {args.model_name}")
    print(f"Model path: {args.model_path}")
    
    
    
    # if args.model_path:
    #     hf_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    #     hf_model = AutoModelForCausalLM.from_pretrained(
    #         args.model_path, 
    #         torch_dtype=torch.bfloat16, 
    #         low_cpu_mem_usage=True, 
    #         device_map=None,#"auto",
    #         # quantization_config=quant_config,
    #     )
    #     model = HookedTransformer.from_pretrained_no_processing(
    #         args.model_name,
    #         # dtype="bfloat16",
    #         dtype="bfloat16",
    #         device="cuda",
    #         hf_model=hf_model,
    #         tokenizer=hf_tokenizer,
    #     )
    # else:
    #     model = HookedTransformer.from_pretrained_no_processing(
    #         args.model_name,
    #         dtype="bfloat16",
    #         # dtype="bfloat8",
    #         device="cuda",
    #     )
    quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4")
    # Step 1: Load efficiently with HuggingFace
    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     torch_dtype=torch.bfloat16,  # Use torch_dtype instead of dtype
    #     device_map="cuda",
    #     low_cpu_mem_usage=True,
    #     # Optional: Add quantization here if needed
    #     # quantization_config=quant_config,
    # )
    
    # hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Converting to HookedTransformer...")
    
    # Step 2: Convert to HookedTransformer (inherits device placement)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name = args.model_name,
        # hf_model=hf_model,  # Pass the already-loaded model
        # tokenizer=hf_tokenizer,
        
        # Don't specify device_map again - it inherits from hf_model
    )
    
    print(f"Model successfully loaded on: {next(model.parameters()).device}")
    # Load data
    use_chat = 'instruct' in args.model_name.lower()

    clean_dataloader = get_loaders(
        args.data,
        tokenizer=model.tokenizer,
        format=args.data_format,
        prompt_template=args.prompt_template,
        use_pos=True,
        use_chat=use_chat,
        use_hard_neg=True
    )

    corrupted_dataloader = get_loaders(
        args.data,
        tokenizer=model.tokenizer,
        format=args.data_format,
        prompt_template=args.prompt_template,
        use_pos=False,
        use_chat=use_chat,
        use_hard_neg=True
    )

    if args.data_format == "pointwise":
        clean_dataloader, corrupted_dataloader = truncate_to_equal_length(clean_dataloader, corrupted_dataloader)

    correct = model.to_single_token("yes")
    wrong = model.to_single_token("no")

    # Experiment
    print(f"Running experiment with method: {args.method}")
    if args.method == "activation_patch":
        index_axis_values = [model.cfg.n_layers]
        if "pos" in args.index_axis_names:
            assert len(args.patching_pos) == 1
            index_axis_values.append(args.patching_pos[0])
        if "head" in args.index_axis_names:
            index_axis_values.append(model.cfg.n_heads)
        if "pos" not in args.index_axis_names and args.patching_pos is not None:
            index_axis_values.extend(args.patching_pos)
        df = activation_patch(
            model,
            clean_dataloader, 
            corrupted_dataloader,
            correct_token_id=correct,
            wrong_token_id=wrong,
            activation_name=args.activation_name,
            index_axis_names=args.index_axis_names,
            index_axis_values=index_axis_values,
            n_samples=args.nsamples,
        )
        df.to_csv(os.path.join(args.output_path, "results_df.csv"), index=False)
    elif args.method == "path_patch":
        index_axis_values = [model.cfg.n_layers]
        if "pos" in args.index_axis_names:
            assert len(args.patching_pos) == 1
            index_axis_values.append(args.patching_pos[0])
        if "head" in args.index_axis_names:
            index_axis_values.append(model.cfg.n_heads)
        if "pos" not in args.index_axis_names and args.patching_pos is not None:
            index_axis_values.extend(args.patching_pos)
        df = path_patch(
            model,
            clean_dataloader, 
            corrupted_dataloader,
            correct_token_id=correct,
            wrong_token_id=wrong,
            activation_name=args.activation_name,
            index_axis_names=args.index_axis_names,
            index_axis_values=index_axis_values,
            receiver_activation_name=args.receiver_activation_name,
            receiver_layer_or_heads=args.receiver_layer_or_heads,
            receiver_pos=args.receiver_pos,
            n_samples=args.nsamples,
        )
        df.to_csv(os.path.join(args.output_path, "results_df.csv"), index=False)
        

if __name__ == "__main__":

    main()















