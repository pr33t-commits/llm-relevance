import gc
import sys
import os
import random
import time
from pathlib import Path
import torch
from torch import Tensor
import numpy as np
import pandas as pd
import einops
from tqdm.auto import tqdm
import re
import itertools
from jaxtyping import Float, Int, Bool
from typing import Literal, Callable, Optional, Tuple, Union, List, Dict, Any, Sequence
from functools import partial
from rich.table import Table, Column
from rich import print as rprint
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, ActivationCache, patching
from transformer_lens.components import Embed, Unembed, LayerNorm, MLP


random.seed(42)

# torch.set_grad_enabled(False)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")



def get_logit(
    logits: Float[Tensor, "batch pos vocab"],
    token_idx: int,
):
    return logits[:, -1, token_idx].mean().item()


def get_prob(
    logits: Float[Tensor, "batch pos vocab"],
    token_idx: int,
):
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    return probs[:, -1, token_idx].mean().item()


def logit_diff(
    logits: Float[Tensor, "batch pos vocab"],
    correct: int,
    wrong: int,
) -> float:
    logitdiff = get_logit(logits, correct) - get_logit(logits, wrong)
    return logitdiff


def prob_diff(
    logits: Float[Tensor, "batch pos vocab"],
    correct: int,
    wrong: int,
) -> float:
    probdiff = get_prob(logits, correct) - get_prob(logits, wrong)
    return probdiff


AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]
PosNames = Literal["all", "last", "query_first", "query_last", "query_all", "doc_first", "doc_last", "doc_all", "inst_first", "inst_last", "inst_all", "qd_all"]


def get_pos(
    seqlen: int,
    pos_type: PosNames,
    query_range: Optional[Tuple[int, int]] = None,
    doc_range: Optional[Tuple[int, int]] = None,
    inst_range: Optional[Tuple[int, int]] = None,
) -> Union[int, slice, list[int]]:
    if pos_type == "all":
        return slice(0, seqlen)
    elif pos_type == "last":
        return seqlen - 1
    elif pos_type == "query_first":
        return query_range[0][0]
    elif pos_type == "query_last":
        return query_range[0][-1] - 1
    elif pos_type == "query_all":
        return slice(*query_range[0])
    elif pos_type == "doc_first":
        if len(doc_range) == 1:
            return doc_range[0][0]
        else:
            return [doc_range[0][0], doc_range[1][0]]
    elif pos_type == "doc_last":
        if len(doc_range) == 1:
            return doc_range[0][-1] - 1
        else:
            return [doc_range[0][-1] - 1, doc_range[1][-1] - 1]
    elif pos_type == "doc_all":
        if len(doc_range) == 1:
            return slice(*doc_range[0])
        else:
            return list(range(*doc_range[0])) + list(range(*doc_range[1]))
    elif pos_type == "qd_all":
        if len(doc_range) == 1:
            return list(range(*query_range[0])) + list(range(*doc_range[0]))
        else:
            return list(range(*query_range[0])) + list(range(*doc_range[0])) + list(range(*doc_range[1]))
    elif pos_type == "inst_first":
        return inst_range[0][0]
    elif pos_type == "inst_last":
        return inst_range[0][-1] - 1
    elif pos_type == "inst_all":
        return slice(*inst_range[0])
    elif pos_type == "other":
        # positions except the above all
        if len(doc_range) == 1:
            return list(set(range(seqlen - 1)) - set(range(*query_range[0])) - set(range(*doc_range[0])) - set(range(*inst_range[0])))
        else:
            return list(set(range(seqlen - 1)) - set(range(*query_range[0])) - set(range(*doc_range[0])) - set(range(*doc_range[1])) - set(range(*inst_range[0])))
    else:
        raise ValueError(f"Invalid pos_type: {pos_type}")


PATCH_SETTER = {
    ("layer", "pos"): patching.layer_pos_patch_setter,
    ("layer", "pos", "head"): patching.layer_pos_head_vector_patch_setter,
    ("layer", "head"): patching.layer_head_vector_patch_setter,
    ("layer", "head", "dest_pos"): patching.layer_head_pos_pattern_patch_setter,
    ("layer", "head", "dest_pos", "src_pos"): patching.layer_head_dest_src_pos_pattern_patch_setter,
}


def create_index_df(
    dataloader: List[Tuple[Tensor, ...]],
    n_samples: int = 10,
    **kwargs,
) -> pd.DataFrame:

    rows = []

    if "sender_layer_or_heads" in kwargs:
        sender_layer_or_heads = kwargs.pop("sender_layer_or_heads")
        if sender_layer_or_heads is None:
            sender_layers = None
            sender_heads = None
        elif isinstance(sender_layer_or_heads, int):
            sender_layers = {sender_layer_or_heads}
        else:
            sender_layers = set(next(zip(*sender_layer_or_heads)))
            sender_heads = sender_layer_or_heads
    else:
        sender_layers = None
        sender_heads = None

    for i in range(n_samples):
        clean_tokens, query_range, doc_range, inst_range, *_ = dataloader[i]

        for layer in range(kwargs["layer"]):
            if "head" in kwargs:
                for head in range(kwargs["head"]):
                    if sender_heads is not None and (layer, head) not in sender_heads:
                        continue
                    row = {"sample": i, "layer": layer, "head": head}
                    for pos in ["pos", "src_pos", "dest_pos"]:
                        if pos in kwargs:
                            row[pos] = get_pos(
                                clean_tokens.shape[1], kwargs[pos],
                                query_range=query_range, doc_range=doc_range, inst_range=inst_range,
                            )
                    if "receiver_pos" in kwargs:
                        row["receiver_pos"] = get_pos(
                            clean_tokens.shape[1], kwargs["receiver_pos"],
                            query_range=query_range, doc_range=doc_range, inst_range=inst_range,
                        )
                        row["receiver_layer_or_heads"] = kwargs["receiver_layer_or_heads"]
                        row["receiver_activation_name"] = kwargs["receiver_activation_name"]
                    rows.append(row)
            else:
                if sender_layers is not None and layer not in sender_layers:
                    continue
                row = {"sample": i, "layer": layer}
                for pos in ["pos", "src_pos", "dest_pos"]:
                    if pos in kwargs:
                        row[pos] = get_pos(
                            clean_tokens.shape[1], kwargs[pos],
                            query_range=query_range, doc_range=doc_range, inst_range=inst_range,
                        )
                if "receiver_pos" in kwargs:
                    row["receiver_pos"] = get_pos(
                        clean_tokens.shape[1], kwargs["receiver_pos"],
                        query_range=query_range, doc_range=doc_range, inst_range=inst_range,
                    )
                    row["receiver_layer_or_heads"] = kwargs["receiver_layer_or_heads"]
                    row["receiver_activation_name"] = kwargs["receiver_activation_name"]
                rows.append(row)

    df = pd.DataFrame(rows, columns=["sample"] + list(kwargs.keys()))
    return df


@torch.no_grad()
def activation_patch(
    model: HookedTransformer,
    clean_dataloader: List[Tuple[Tensor, ...]],
    corrupted_dataloader: List[Tuple[Tensor, ...]],
    correct_token_id: int,
    wrong_token_id: int,
    activation_name: str,
    index_axis_names: Sequence[AxisNames],
    index_axis_values: Sequence[Union[int, str]],
    n_samples: int = 10,
) -> pd.DataFrame:
    
    model.reset_hooks()

    n_samples = min(n_samples, len(clean_dataloader))

    index_df = create_index_df(clean_dataloader, n_samples, **dict(zip(index_axis_names, index_axis_values)))

    patch_setter = PATCH_SETTER[tuple(index_axis_names)]
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)

    correct_logit, wrong_logit = partial(get_logit, token_idx=correct_token_id), partial(get_logit, token_idx=wrong_token_id)
    correct_prob, wrong_prob = partial(get_prob, token_idx=correct_token_id), partial(get_prob, token_idx=wrong_token_id)

    metric_dict = {
        f"{input_type}_{metric}": np.zeros(len(index_df)) 
        for input_type in ["clean", "corrupted", "patched"] 
        for metric in ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob"]
    }

    for i in tqdm(range(n_samples)):

        clean_tokens = clean_dataloader[i][0]
        corrupted_tokens = corrupted_dataloader[i][0]

        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=[utils.get_act_name(activation_name, j) for j in range(model.cfg.n_layers)],
        )
        corrupted_logits = model(corrupted_tokens)

        sample_index_df = index_df[index_df["sample"] == i]
        for index_row in (list(sample_index_df.iterrows())):
            gi = index_row[0]
            _, *index = index_row[1].to_list()

            current_activation_name = utils.get_act_name(activation_name, layer=index[0])
            current_hook = partial(
                patching_hook,
                index=index,
                clean_activation=clean_cache[current_activation_name],
            )

            patched_logits = model.run_with_hooks(
                corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
            )

            for prefix, logits in zip(["clean", "corrupted", "patched"], [clean_logits, corrupted_logits, patched_logits]):
                for metric, func in zip(
                    ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob"],
                    [correct_logit, wrong_logit, correct_prob, wrong_prob],
                ):
                    metric_dict[f"{prefix}_{metric}"][gi] = func(logits)
    
    for key, value in metric_dict.items():
        index_df[key] = value

    return index_df


@torch.no_grad()
def path_patch(
    model: HookedTransformer,
    clean_dataloader: List[Tuple[Tensor, ...]],
    corrupted_dataloader: List[Tuple[Tensor, ...]],
    correct_token_id: int,
    wrong_token_id: int,
    activation_name: str,
    index_axis_names: Sequence[AxisNames],
    index_axis_values: Sequence[Union[int, str]],
    receiver_layer_or_heads: Union[int, List[Tuple[int, int]]],
    receiver_activation_name: str,
    receiver_pos: PosNames,
    sender_layer_or_heads: Optional[Union[int, List[int], List[Tuple[int, int]]]] = None,
    n_samples: int = 10,
) -> pd.DataFrame:

    n_samples = min(n_samples, len(clean_dataloader))

    correct_logit, wrong_logit = partial(get_logit, token_idx=correct_token_id), partial(get_logit, token_idx=wrong_token_id)
    correct_prob, wrong_prob = partial(get_prob, token_idx=correct_token_id), partial(get_prob, token_idx=wrong_token_id)

    model.reset_hooks()

    if isinstance(receiver_layer_or_heads, int):
        receiver_layers = {receiver_layer_or_heads}
    else:
        assert receiver_activation_name in ("k", "q", "v", "z")
        receiver_layers = set(next(zip(*receiver_layer_or_heads)))
        receiver_heads = receiver_layer_or_heads
        if model.cfg.n_key_value_heads != model.cfg.n_heads and receiver_activation_name in ("k", "v"):
            receiver_heads = [(layer, head // (model.cfg.n_heads // model.cfg.n_key_value_heads)) for layer, head in receiver_heads]

    receiver_hook_names = [utils.get_act_name(receiver_activation_name, layer) for layer in receiver_layers]
    receiver_hook_names_filter = lambda name: name in receiver_hook_names

    index_df = create_index_df(
        clean_dataloader,
        n_samples,
        layer=max(receiver_layers) + 1 if isinstance(receiver_layer_or_heads, int) else max(receiver_layers),
        **dict(zip(index_axis_names[1:], index_axis_values[1:])),
        receiver_layer_or_heads=receiver_layer_or_heads,
        receiver_activation_name=receiver_activation_name,
        receiver_pos=receiver_pos,
        sender_layer_or_heads=sender_layer_or_heads,
    )

    metric_dict = {
        f"{input_type}_{metric}": np.zeros(len(index_df)) 
        for input_type in ["clean", "corrupted", "patched"] 
        for metric in ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob"]
    }

    patch_setter = PATCH_SETTER[tuple(index_axis_names)]
    def patch_or_freeze_hook(activation, hook, index, orig_cache, new_cache):
        activation[...] = orig_cache[hook.name][...]
        if index[0] == hook.layer():
            patch_setter(activation, index, new_cache[hook.name])
        return activation

    def patch_head_input_hook(activation, hook, patched_cache, head_list, pos):
        heads_to_patch = [head for layer, head in head_list if layer == hook.layer()]
        activation[:, pos, heads_to_patch] = patched_cache[hook.name][:, pos, heads_to_patch]
        return activation

    def patch_vector_hook(activation, hook, patched_cache, pos):
        activation[:, pos] = patched_cache[hook.name][:, pos]
        return activation

    for i in tqdm(range(n_samples)):
        model.reset_hooks()

        clean_tokens = clean_dataloader[i][0]
        corrupted_tokens = corrupted_dataloader[i][0]
        assert clean_tokens.shape == corrupted_tokens.shape

        clean_logits, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=[utils.get_act_name(activation_name, j) for j in range(model.cfg.n_layers)],
        )
        corrupted_logits, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=[utils.get_act_name(activation_name, j) for j in range(model.cfg.n_layers)],
        )

        sample_index_df = index_df[index_df["sample"] == i]
        for index_row in list(sample_index_df.iterrows()):
            gi = index_row[0]
            _, *index, _, _, receiver_pos = index_row[1].to_list()

            layer = index_row[1]["layer"]
            head = index_row[1].get("head", None)

            hook_fn = partial(
                patch_or_freeze_hook,
                index=index,
                orig_cache=clean_cache,
                new_cache=corrupted_cache,
            )
            model.add_hook(utils.get_act_name(activation_name, layer), hook_fn, level=1)

            _, patched_cache = model.run_with_cache(
                clean_tokens,
                names_filter=receiver_hook_names_filter,
                return_type=None
            )
            # model.reset_hooks(including_permanent=True)
            assert set(patched_cache.keys()) == set(receiver_hook_names)

            if receiver_activation_name == "resid_post" and receiver_layer_or_heads == model.cfg.n_layers - 1:
                assert len(receiver_hook_names) == 1
                patched_logits = model.unembed(model.ln_final(patched_cache[receiver_hook_names[0]]))
            else:
                if isinstance(receiver_layer_or_heads, int):
                    hook_fn = partial(
                        patch_vector_hook,
                        patched_cache=patched_cache,
                        pos=receiver_pos
                    )
                else:
                    hook_fn = partial(
                        patch_head_input_hook,
                        patched_cache=patched_cache,
                        head_list=receiver_heads,
                        pos=receiver_pos
                    )
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(receiver_hook_names_filter, hook_fn)],
                    return_type="logits"
                )

            for prefix, logits in zip(["clean", "corrupted", "patched"], [clean_logits, corrupted_logits, patched_logits]):
                for metric, func in zip(
                    ["correct_logit", "wrong_logit", "correct_prob", "wrong_prob"],
                    [correct_logit, wrong_logit, correct_prob, wrong_prob],
                ):
                    metric_dict[f"{prefix}_{metric}"][gi] = func(logits)
    
    for key, value in metric_dict.items():
        index_df[key] = value

    return index_df


@torch.no_grad()
def get_mean_activations(
    model: HookedTransformer,
    mean_dataloader: List[Tuple[Tensor, ...]],
):
    model.reset_hooks()

    # Get the mean activations
    mean_activations = {layer: {} for layer in range(model.cfg.n_layers)}
    def get_activation_hook(activation, hook):
        if "z" in hook.name:
            mean_activations[hook.layer()]["z"] = mean_activations[hook.layer()].get("z", 0) + activation.sum(0)
        elif "mlp_out" in hook.name:
            mean_activations[hook.layer()]["mlp_out"] = mean_activations[hook.layer()].get("mlp_out", 0) + activation.sum(0)
        return activation

    for i in range(len(mean_dataloader)):
        tokens = mean_dataloader[i][0]
        _ = model.run_with_hooks(
            tokens,
            fwd_hooks=
                [(utils.get_act_name("z", layer), get_activation_hook) for layer in range(model.cfg.n_layers)] + \
                [(utils.get_act_name("mlp_out", layer), get_activation_hook) for layer in range(model.cfg.n_layers)],
        )

    for layer in mean_activations:
        mean_activations[layer]["z"] /= len(mean_dataloader)
        mean_activations[layer]["mlp_out"] /= len(mean_dataloader)

    return mean_activations


@torch.no_grad()
def mean_ablation(
    model: HookedTransformer,
    evaluation_dataloader: List[Tuple[Tensor, ...]],
    mean_activations: dict[int, dict[str, Tensor]],
    ablate_heads: dict[tuple[int, int], str],
    # ablate_layers: dict[int, str],
    mask_ablate_modules: bool = False,
    n_samples: int = 10,
):
    model.reset_hooks()

    n_samples = min(n_samples, len(evaluation_dataloader))

    seqlen = evaluation_dataloader[0][0].shape[1]
    _, query_range, doc_range, inst_range, *_ = evaluation_dataloader[0]
    query_range = list(range(*query_range[0]))
    doc_range = list(range(*doc_range[0])) if len(doc_range) == 1 else list(range(*doc_range[0])) + list(range(*doc_range[1]))
    inst_range = list(range(*inst_range[0]))

    pos_dict = {
        "query_all": query_range,
        "doc_all": doc_range,
        "qd_all": query_range + doc_range,
        "inst_all": inst_range,
        "last": seqlen - 1,
        "all": slice(0, seqlen),
    }

    ablate_pos_dict = {
        "query_all": list(set(range(seqlen)) - set(query_range)),
        "doc_all": list(set(range(seqlen)) - set(doc_range)),
        "qd_all": list(set(range(seqlen)) - set(query_range + doc_range)),
        "inst_all": list(set(range(seqlen)) - set(inst_range)),
        "last": slice(0, seqlen - 1),
        "all": slice(0, 0),
    }

    # Compute ablated outputs
    def mean_ablate_head_hook(activation, hook):
        for head_idx in range(model.cfg.n_heads):
            head = (hook.layer(), head_idx)
            if not mask_ablate_modules:
                if head not in ablate_heads:
                    activation[:, :, head_idx] = mean_activations[hook.layer()]["z"][:, head_idx]
                else:
                    pos = ablate_pos_dict[ablate_heads[head]]
                    activation[:, pos, head_idx] = mean_activations[hook.layer()]["z"][pos, head_idx]
            elif mask_ablate_modules and head in ablate_heads:
                pos = pos_dict[ablate_heads[head]]
                activation[:, pos, head_idx] = mean_activations[hook.layer()]["z"][pos, head_idx]
        return activation

    def mean_ablate_layer_hook(activation, hook):
        activation[...] = mean_activations[hook.layer()]["mlp_out"][...]
        return activation

    ablated_outputs = []
    for i in tqdm(range(n_samples), desc="Computing ablated outputs"):
        tokens = evaluation_dataloader[i][0]
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("z", layer), mean_ablate_head_hook) for layer in range(model.cfg.n_layers)],
            return_type="logits"
        )[0, -1, :]
        ablated_outputs.append(logits.cpu())

    return ablated_outputs


@torch.no_grad()
def zero_ablation(
    model: HookedTransformer,
    evaluation_dataloader: List[Tuple[Tensor, ...]],
    ablate_heads: dict[tuple[int, int], str],
    # ablate_layers: dict[int, str],
    mask_ablate_modules: bool = False,
    n_samples: int = 10,
):
    model.reset_hooks()

    n_samples = min(n_samples, len(evaluation_dataloader))

    seqlen = evaluation_dataloader[0][0].shape[1]
    _, query_range, doc_range, inst_range, *_ = evaluation_dataloader[0]
    query_range = list(range(*query_range[0]))
    doc_range = list(range(*doc_range[0])) if len(doc_range) == 1 else list(range(*doc_range[0])) + list(range(*doc_range[1]))
    inst_range = list(range(*inst_range[0]))

    pos_dict = {
        "query_all": query_range,
        "doc_all": doc_range,
        "qd_all": query_range + doc_range,
        "inst_all": inst_range,
        "last": seqlen - 1,
        "all": slice(0, seqlen),
    }

    ablate_pos_dict = {
        "query_all": list(set(range(seqlen)) - set(query_range)),
        "doc_all": list(set(range(seqlen)) - set(doc_range)),
        "qd_all": list(set(range(seqlen)) - set(query_range + doc_range)),
        "inst_all": list(set(range(seqlen)) - set(inst_range)),
        "last": slice(0, seqlen - 1),
        "all": slice(0, 0),
    }

    # Compute ablated outputs
    def zero_ablate_head_hook(activation, hook):
        for head_idx in range(model.cfg.n_heads):
            head = (hook.layer(), head_idx)
            if not mask_ablate_modules:
                if head not in ablate_heads:
                    activation[:, :, head_idx] = 0
                else:
                    pos = ablate_pos_dict[ablate_heads[head]]
                    activation[:, pos, head_idx] = 0
            elif mask_ablate_modules and head in ablate_heads:
                pos = pos_dict[ablate_heads[head]]
                activation[:, pos, head_idx] = 0
        return activation

    ablated_outputs = []
    for i in tqdm(range(n_samples), desc="Computing ablated outputs"):
        tokens = evaluation_dataloader[i][0]
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(utils.get_act_name("z", layer), zero_ablate_head_hook) for layer in range(model.cfg.n_layers)],
            return_type="logits"
        )[0, -1, :]
        ablated_outputs.append(logits.cpu())

    return ablated_outputs


@torch.no_grad()
def patched_ablation(
    model: HookedTransformer,
    evaluation_dataloader: List[Tuple[Tensor, ...]],
    patched_dataloader: List[Tuple[Tensor, ...]],
    ablate_heads: dict[tuple[int, int], str],
    ablate_mlps: dict[int, str],
    ablate_attns: dict[int, str],
    mask_ablate_modules: bool = False,
    n_samples: int = 10,
):
    model.reset_hooks()

    n_samples = min(n_samples, len(evaluation_dataloader))

    def patched_ablate_head_hook(activation, hook, new_cache, pos_dict, ablate_pos_dict):
        for head_idx in range(model.cfg.n_heads):
            head = (hook.layer(), head_idx)
            if not mask_ablate_modules:
                if head not in ablate_heads:
                    activation[:, :, head_idx] = new_cache[hook.name][:, :, head_idx]
                else:
                    pos = ablate_pos_dict[ablate_heads[head]]
                    activation[:, pos, head_idx] = new_cache[hook.name][:, pos, head_idx]
            elif mask_ablate_modules and head in ablate_heads:
                pos = pos_dict[ablate_heads[head]]
                activation[:, pos, head_idx] = new_cache[hook.name][:, pos, head_idx]
        return activation

    def patched_ablate_mlp_hook(activation, hook, new_cache, pos_dict, ablate_pos_dict):
        if not mask_ablate_modules:
            if hook.layer() not in ablate_mlps:
                activation[...] = new_cache[hook.name][...]
            else:
                pos = ablate_pos_dict[ablate_mlps[hook.layer()]]
                activation[:, pos] = new_cache[hook.name][:, pos]
        elif mask_ablate_modules and hook.layer() in ablate_mlps:
            pos = pos_dict[ablate_mlps[hook.layer()]]
            activation[:, pos] = new_cache[hook.name][:, pos]
        return activation

    def patched_ablate_attn_hook(activation, hook, new_cache, pos_dict, ablate_pos_dict):
        if not mask_ablate_modules:
            if hook.layer() not in ablate_attns:
                activation[...] = new_cache[hook.name][...]
            else:
                pos = ablate_pos_dict[ablate_attns[hook.layer()]]
                activation[:, pos] = new_cache[hook.name][:, pos]
        elif mask_ablate_modules and hook.layer() in ablate_attns:
            pos = pos_dict[ablate_attns[hook.layer()]]
            activation[:, pos] = new_cache[hook.name][:, pos]
        return activation

    ablated_outputs = []
    for i in tqdm(range(n_samples), desc="Computing ablated outputs"):
        orig_tokens, query_range, doc_range, inst_range, *_ = evaluation_dataloader[i]
        new_tokens, *_ = patched_dataloader[i]

        seqlen = orig_tokens.shape[1]

        query_range = list(range(*query_range[0]))
        doc_range = list(range(*doc_range[0])) if len(doc_range) == 1 else list(range(*doc_range[0])) + list(range(*doc_range[1]))
        inst_range = list(range(*inst_range[0]))

        pos_dict = {
            "query_all": query_range,
            "doc_all": doc_range,
            "qd_all": query_range + doc_range,
            "inst_all": inst_range,
            "last": seqlen - 1,
            "all": slice(0, seqlen),
        }

        ablate_pos_dict = {
            "query_all": list(set(range(seqlen)) - set(query_range)),
            "doc_all": list(set(range(seqlen)) - set(doc_range)),
            "qd_all": list(set(range(seqlen)) - set(query_range + doc_range)),
            "inst_all": list(set(range(seqlen)) - set(inst_range)),
            "last": slice(0, seqlen - 1),
            "all": slice(0, 0),
        }

        _, new_cache = model.run_with_cache(
            new_tokens,
            names_filter=[utils.get_act_name("z", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("mlp_out", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("attn_out", j) for j in range(model.cfg.n_layers)],
        )

        head_hook_fn = partial(
            patched_ablate_head_hook,
            new_cache=new_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )
        mlp_hook_fn = partial(
            patched_ablate_mlp_hook,
            new_cache=new_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )
        attn_hook_fn = partial(
            patched_ablate_attn_hook,
            new_cache=new_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )

        logits = model.run_with_hooks(
            orig_tokens,
            fwd_hooks=
                    # [(utils.get_act_name("z", layer), head_hook_fn) for layer in range(model.cfg.n_layers)] + \
                    [(utils.get_act_name("mlp_out", layer), mlp_hook_fn) for layer in range(model.cfg.n_layers)] + \
                    [(utils.get_act_name("attn_out", layer), attn_hook_fn) for layer in range(model.cfg.n_layers)],
            return_type="logits"
        )[0, -1, :]
        ablated_outputs.append(logits.cpu())

    return ablated_outputs


@torch.no_grad()
def ablation_for_each_query(
    model: HookedTransformer,
    clean_dataloader: List[Tuple[Tensor, ...]],
    corrupted_dataloader: List[Tuple[Tensor, ...]],
    ablate_heads: dict[tuple[int, int], str],
    ablate_mlps: dict[int, str],
    ablate_attns: dict[int, str],
    mask_ablate_modules: bool = False,
    ablate: Literal["mean", "zero"] = "mean",
    n_samples: int = 10,
):
    model.reset_hooks()

    n_samples = min(n_samples, len(clean_dataloader))

    def mean_ablate_head_hook(activation, hook, clean_cache, corrupted_cache, pos_dict, ablate_pos_dict):
        for head_idx in range(model.cfg.n_heads):
            head = (hook.layer(), head_idx)
            if not mask_ablate_modules:
                if head not in ablate_heads:
                    activation[:, :, head_idx] = (clean_cache[hook.name][:, :, head_idx] + corrupted_cache[hook.name][:, :, head_idx]) / 2
                else:
                    if ablate_heads[head] != "all":
                        pos = ablate_pos_dict[ablate_heads[head]]
                        activation[:, pos, head_idx] = (clean_cache[hook.name][:, pos, head_idx] + corrupted_cache[hook.name][:, pos, head_idx]) / 2
            elif mask_ablate_modules and head in ablate_heads:
                pos = pos_dict[ablate_heads[head]]
                activation[:, pos, head_idx] = (clean_cache[hook.name][:, pos, head_idx] + corrupted_cache[hook.name][:, pos, head_idx]) / 2
        return activation

    def mean_ablate_mlp_hook(activation, hook, clean_cache, corrupted_cache, pos_dict, ablate_pos_dict):
        if not mask_ablate_modules:
            if hook.layer() not in ablate_mlps:
                activation[...] = (clean_cache[hook.name] + corrupted_cache[hook.name]) / 2
            else:
                if ablate_mlps[hook.layer()] != "all":
                    pos = ablate_pos_dict[ablate_mlps[hook.layer()]]
                    activation[:, pos] = (clean_cache[hook.name][:, pos] + corrupted_cache[hook.name][:, pos]) / 2
        elif mask_ablate_modules and hook.layer() in ablate_mlps:
            pos = pos_dict[ablate_mlps[hook.layer()]]
            activation[:, pos] = (clean_cache[hook.name][:, pos] + corrupted_cache[hook.name][:, pos]) / 2
        return activation

    def mean_ablate_attn_hook(activation, hook, clean_cache, corrupted_cache, pos_dict, ablate_pos_dict):
        if not mask_ablate_modules:
            if hook.layer() not in ablate_attns:
                activation[...] = (clean_cache[hook.name] + corrupted_cache[hook.name]) / 2
            else:
                if ablate_attns[hook.layer()] != "all":
                    pos = ablate_pos_dict[ablate_attns[hook.layer()]]
                    activation[:, pos] = (clean_cache[hook.name][:, pos] + corrupted_cache[hook.name][:, pos]) / 2
        elif mask_ablate_modules and hook.layer() in ablate_attns:
            pos = pos_dict[ablate_attns[hook.layer()]]
            activation[:, pos] = (clean_cache[hook.name][:, pos] + corrupted_cache[hook.name][:, pos]) / 2
        return activation

    ablated_outputs = []
    for i in tqdm(range(n_samples), desc="Computing ablated outputs"):
        clean_tokens, query_range, doc_range, inst_range, *_ = clean_dataloader[i]
        corrupted_tokens, *_ = corrupted_dataloader[i]

        seqlen = clean_tokens.shape[1]

        query_range = list(range(*query_range[0]))
        doc_range = list(range(*doc_range[0])) if len(doc_range) == 1 else list(range(*doc_range[0])) + list(range(*doc_range[1]))
        inst_range = list(range(*inst_range[0]))

        pos_dict = {
            "query_all": query_range,
            "doc_all": doc_range,
            "qd_all": query_range + doc_range,
            "inst_all": inst_range,
            "last": seqlen - 1,
            "all": slice(0, seqlen),
        }

        ablate_pos_dict = {
            "query_all": list(set(range(seqlen)) - set(query_range)),
            "doc_all": list(set(range(seqlen)) - set(doc_range)),
            "qd_all": list(set(range(seqlen)) - set(query_range + doc_range)),
            "inst_all": list(set(range(seqlen)) - set(inst_range)),
            "last": slice(0, seqlen - 1),
            "all": slice(0, 0),
        }

        _, clean_cache = model.run_with_cache(
            clean_tokens,
            names_filter=[utils.get_act_name("z", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("mlp_out", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("attn_out", j) for j in range(model.cfg.n_layers)],
        )
        _, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=[utils.get_act_name("z", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("mlp_out", j) for j in range(model.cfg.n_layers)] + \
                [utils.get_act_name("attn_out", j) for j in range(model.cfg.n_layers)],
        )

        head_hook_fn = partial(
            mean_ablate_head_hook,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )
        mlp_hook_fn = partial(
            mean_ablate_mlp_hook,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )
        attn_hook_fn = partial(
            mean_ablate_attn_hook,
            clean_cache=clean_cache,
            corrupted_cache=corrupted_cache,
            pos_dict=pos_dict,
            ablate_pos_dict=ablate_pos_dict,
        )

        logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=
                    [(utils.get_act_name("z", layer), head_hook_fn) for layer in range(model.cfg.n_layers)] + \
                    [(utils.get_act_name("mlp_out", layer), mlp_hook_fn) for layer in range(model.cfg.n_layers)],
                    # [(utils.get_act_name("attn_out", layer), attn_hook_fn) for layer in range(model.cfg.n_layers)],
            return_type="logits"
        )[0, -1, :]
        ablated_outputs.append(logits.cpu())
    
        logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=
                    [(utils.get_act_name("z", layer), head_hook_fn) for layer in range(model.cfg.n_layers)] + \
                    [(utils.get_act_name("mlp_out", layer), mlp_hook_fn) for layer in range(model.cfg.n_layers)],
                    # [(utils.get_act_name("attn_out", layer), attn_hook_fn) for layer in range(model.cfg.n_layers)],
            return_type="logits"
        )[0, -1, :]
        ablated_outputs.append(logits.cpu())

    return ablated_outputs
