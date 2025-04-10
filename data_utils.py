import json
import random
import numpy as np
import torch
import transformers
from typing import Optional


def get_pointwise_ranges(tokenizer, prompt, query, document, inst, use_chat=False):
    if use_chat:
        _get_length = lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x.strip()}], continue_final_message=True))
    else:
        _get_length = lambda x: len(tokenizer(x.strip())[0])

    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]

    prefix, *_ = prompt.split(document)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document)
    doc_range = [(doc_start, doc_end)]

    prefix, *_ = prompt.split(inst)
    inst_start = _get_length(prefix)
    if len(prefix) == 0:
        inst_start -= 1
    inst_end = _get_length(prefix + inst)
    inst_range = [(inst_start, inst_end)]

    return query_range, doc_range, inst_range


def get_pointwise_ranges_v2(tokenizer, prompt, query, document, inst, query_length, doc_length, use_chat=True):
    # only work for chat templates now
    _get_ids_chat = lambda x: tokenizer.apply_chat_template([{"role": "user", "content": x}], continue_final_message=True, return_tensors="pt")[0]
    _get_ids = lambda x: tokenizer.encode(x, return_tensors="pt", add_special_tokens=False)[0].long()

    prefix, suffix = prompt.split(document)
    mid, suffix = suffix.split(query)
    mid2, suffix = suffix.split(inst)
    suffix += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # hotfix for chat templates

    prefix_ids = _get_ids_chat(prefix.strip())
    doc_ids = _get_ids(document.strip())
    if len(doc_ids) > doc_length:
        doc_ids = doc_ids[:doc_length]
    elif len(doc_ids) < doc_length:
        doc_ids = doc_ids.repeat(10)[:doc_length]
    mid_ids = _get_ids(mid.strip())
    query_ids = _get_ids(query.strip())
    if len(query_ids) > query_length:
        query_ids = query_ids[:query_length]
    elif len(query_ids) < query_length:
        query_ids = query_ids.repeat(10)[:query_length]
    mid2_ids = _get_ids(mid2.strip())
    inst_ids = _get_ids(inst.strip())
    suffix_ids = _get_ids(suffix)
    doc_range = [(len(prefix_ids), len(prefix_ids) + len(doc_ids))]
    query_range = [(len(prefix_ids) + len(doc_ids) + len(mid_ids), len(prefix_ids) + len(doc_ids) + len(mid_ids) + len(query_ids))]
    inst_range = [(
        len(prefix_ids) + len(doc_ids) + len(mid_ids) + len(query_ids) + len(mid2_ids), 
        len(prefix_ids) + len(doc_ids) + len(mid_ids) + len(query_ids) + len(mid2_ids) + len(inst_ids))]
    input_ids = torch.cat([prefix_ids, doc_ids, mid_ids, query_ids, mid2_ids, inst_ids, suffix_ids]).unsqueeze(0)

    return input_ids, query_range, doc_range, inst_range


def get_pairwise_ranges(tokenizer, prompt, query, document1, document2, inst, use_chat=False):
    if use_chat:
        _get_length = lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x}], continue_final_message=True))
    else:
        _get_length = lambda x: len(tokenizer(x)[0])


    prefix, *_ = prompt.split(query)
    query_start = _get_length(prefix)
    query_end = _get_length(prefix + query)
    query_range = [(query_start, query_end)]

    prefix, *_ = prompt.split(document1)
    doc_start = _get_length(prefix)
    doc_end = _get_length(prefix + document1)
    doc_range_1 = (doc_start, doc_end)

    prefix, *_ = prompt.split(document2)
    doc_start = _get_length(prefix.strip())
    doc_end = _get_length(prefix + document2)
    doc_range_2 = (doc_start, doc_end)

    doc_length = min(doc_range_1[1] - doc_range_1[0], doc_range_2[1] - doc_range_2[0])
    doc_range = [(doc_range_1[0], doc_range_1[0] + doc_length), (doc_range_2[0], doc_range_2[0] + doc_length)]

    prefix, *_ = prompt.split(inst)
    inst_start = _get_length(prefix)
    if len(prefix) == 0:
        inst_start -= 1
    inst_end = _get_length(prefix + inst)
    inst_range = [(inst_start, inst_end)]

    return query_range, doc_range, inst_range


def get_pairwise_ranges_v2(tokenizer, prompt, query, document1, document2, inst, query_length, doc_length, use_chat=True):
    # only work for chat templates now
    _get_ids_chat = lambda x: tokenizer.apply_chat_template([{"role": "user", "content": x}], continue_final_message=True, return_tensors="pt")[0]
    _get_ids = lambda x: tokenizer.encode(x, return_tensors="pt", add_special_tokens=False)[0].long()

    prefix, *suffix = prompt.split(document1)
    suffix = document1.join(suffix)  # to handle cases where document1 is a substring of document2
    mid, suffix = suffix.split(document2)
    mid1, suffix = suffix.split(query)
    mid2, suffix = suffix.split(inst)
    suffix += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"  # hotfix for chat templates

    prefix_ids = _get_ids_chat(prefix.strip())
    doc1_ids = _get_ids(document1.strip())
    if len(doc1_ids) > doc_length:
        doc1_ids = doc1_ids[:doc_length]
    elif len(doc1_ids) < doc_length:
        doc1_ids = doc1_ids.repeat(10)[:doc_length]
    mid_ids = _get_ids(mid.strip())
    doc2_ids = _get_ids(document2.strip())
    if len(doc2_ids) > doc_length:
        doc2_ids = doc2_ids[:doc_length]
    elif len(doc2_ids) < doc_length:
        doc2_ids = doc2_ids.repeat(10)[:doc_length]
    mid1_ids = _get_ids(mid1.strip())
    query_ids = _get_ids(query.strip())
    if len(query_ids) > query_length:
        query_ids = query_ids[:query_length]
    elif len(query_ids) < query_length:
        query_ids = query_ids.repeat(5)[:query_length]
    mid2_ids = _get_ids(mid2.strip())
    inst_ids = _get_ids(inst.strip())
    suffix_ids = _get_ids(suffix)
    doc1_range = [(len(prefix_ids), len(prefix_ids) + len(doc1_ids))]
    doc2_range = [(len(prefix_ids) + len(doc1_ids) + len(mid_ids), len(prefix_ids) + len(doc1_ids) + len(mid_ids) + len(doc2_ids))]
    doc_range = doc1_range + doc2_range
    query_range = [(len(prefix_ids) + len(doc1_ids) + len(mid_ids) + len(doc2_ids) + len(mid1_ids), 
                    len(prefix_ids) + len(doc1_ids) + len(mid_ids) + len(doc2_ids) + len(mid1_ids) + len(query_ids))]
    inst_range = [(
        len(prefix_ids) + len(doc1_ids) + len(mid_ids) + len(doc2_ids) + len(mid1_ids) + len(query_ids) + len(mid2_ids),
        len(prefix_ids) + len(doc1_ids) + len(mid_ids) + len(doc2_ids) + len(mid1_ids) + len(query_ids) + len(mid2_ids) + len(inst_ids))]
    input_ids = torch.cat([prefix_ids, doc1_ids, mid_ids, doc2_ids, mid1_ids, query_ids, mid2_ids, inst_ids, suffix_ids]).unsqueeze(0)

    return input_ids, query_range, doc_range, inst_range


def get_listwise_ranges(tokenizer, prompt, query, documents):
    _get_length = lambda x: len(tokenizer.apply_chat_template([{"role": "user", "content": x}], continue_final_message=True))
    pass


def get_loaders(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    format: str,
    prompt_template: str,
    use_chat: bool = True,
    use_pos: bool = None,
    use_hard_neg: bool = False,
    nsamples: int = 100,
    seqlen: int = 2048
):
    random.seed(42)  # for reproducibility

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f][:nsamples]

    if format == "pointwise":
        inst = "Does the document answer the query? Only output 'yes' or 'no'."
        # inst = "Is the document relevant to the query? the answer is: "
    elif format == "pairwise":
        # inst = "Which document is more relevant to the query? Only output 'A' or 'B'."
        inst = "Is document A more relevant to the query? Only output 'yes' or 'no'."

    dataloader = []
    for i, item in enumerate(data):
        if format == "pointwise" and use_pos is True:
            query = item["query"]
            document = item["positive_document"]
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 1
            query_range, doc_range, inst_range = get_pointwise_ranges(tokenizer, prompt, query, document, inst, use_chat)
        elif format == "pointwise" and use_pos is False:
            query = item["query"]
            if use_hard_neg:
                document = random.choice(item["hard_negative_document"])
            else:
                document = random.choice(item["random_negative_document"])
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 0
            query_range, doc_range, inst_range = get_pointwise_ranges(tokenizer, prompt, query, document, inst, use_chat)
        elif format == "pointwise" and use_pos is None:
            query = item["query"]
            document = random.choice(item["random_negative_document"] + item["hard_negative_document"] + [item["positive_document"]] * 4)
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 1 if document == item["positive_document"] else 0
            query_range, doc_range, inst_range = get_pointwise_ranges(tokenizer, prompt, query, document, inst, use_chat)
        elif format == "pairwise":
            query = item["query"]
            document1 = item["positive_document"]
            if use_hard_neg:
                document2 = random.choice(item["hard_negative_document"])
            else:
                document2 = random.choice(item["random_negative_document"])
            # document1 is more relevant if `use_pos=True`, else document2 is more relevant
            if use_pos is False:
                document1, document2 = document2, document1

            document1, document2 = document1.strip(), document2.strip()
            input_ids = tokenizer.batch_encode_plus([document1 + "\n\n", document2 + "\n\n"], add_special_tokens=False).input_ids
            if len(input_ids[0]) > len(input_ids[1]):
                input_ids = [input_ids[0][:len(input_ids[1]) - 1] + [input_ids[0][len(input_ids[0]) - 1]], input_ids[1]]
            else:
                input_ids = [input_ids[0], input_ids[1][:len(input_ids[0]) - 1] + [input_ids[1][len(input_ids[1]) - 1]]]
            document1, document2 = tokenizer.decode(input_ids[0]), tokenizer.decode(input_ids[1])
            document1, document2 = document1.strip(), document2.strip()

            label = None
            prompt = prompt_template.format(
                query=query,
                document1=document1,
                document2=document2,
                inst=inst
            )
            query_range, doc_range, inst_range = get_pairwise_ranges(tokenizer, prompt, query, document1, document2, inst, use_chat)
            assert (doc_range[0][1] - doc_range[0][0]) - (doc_range[1][1] - doc_range[1][0]) <= 1

        elif format == "listwise":
            pass

        if use_chat:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding="longest",
                max_length=seqlen,
                truncation=True,
                add_generation_prompt=True
            )
        else:
            input_ids = tokenizer(prompt, return_tensors="pt", padding="longest", max_length=seqlen, truncation=True).input_ids

        dataloader.append((input_ids, query_range, doc_range, inst_range, label))

    return dataloader


def truncate_to_equal_length(clean_dataloader, corrupted_dataloader):
    new_clean_dataloader = []
    new_corrupted_dataloader = []
    for (clean_tokens, clean_query_range, clean_doc_range, clean_inst_range, _), \
        (corrupted_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, _) in zip(clean_dataloader, corrupted_dataloader):

        clean_query_start, clean_query_end = clean_query_range[0]
        corrupted_query_start, corrupted_query_end = corrupted_query_range[0]
        clean_doc_start, clean_doc_end = clean_doc_range[0]
        corrupted_doc_start, corrupted_doc_end = corrupted_doc_range[0]

        assert torch.equal(clean_tokens[0, clean_query_start:clean_query_end], corrupted_tokens[0, corrupted_query_start:corrupted_query_end])

        if clean_tokens.shape[1] > corrupted_tokens.shape[1]:
            new_clean_tokens = corrupted_tokens.clone()
            new_clean_tokens[:, corrupted_doc_start:corrupted_doc_end - 1] = clean_tokens[:, clean_doc_start:clean_doc_start + (corrupted_doc_end - corrupted_doc_start) - 1]
            new_clean_dataloader.append((new_clean_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
            new_corrupted_dataloader.append((corrupted_tokens, corrupted_query_range, corrupted_doc_range, corrupted_inst_range, None))
        else:
            new_corrupted_tokens = clean_tokens.clone()
            new_corrupted_tokens[:, clean_doc_start:clean_doc_end - 1] = corrupted_tokens[:, corrupted_doc_start:corrupted_doc_start + (clean_doc_end - clean_doc_start) - 1]
            new_clean_dataloader.append((clean_tokens, clean_query_range, clean_doc_range, clean_inst_range, None))
            new_corrupted_dataloader.append((new_corrupted_tokens, clean_query_range, clean_doc_range, clean_inst_range, None))

    return new_clean_dataloader, new_corrupted_dataloader



def get_loaders_with_equal_length(
    data_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    format: str,
    prompt_template: str,
    query_seqlen: int = 5,
    doc_seqlen: int = 100,
    use_chat: bool = True,
    use_pos: bool = None,
    use_hard_neg: bool = False,
    nsamples: int = 100,
    seqlen: int = 2048,
):
    random.seed(42)  # for reproducibility

    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f][:nsamples]

    if format == "pointwise":
        inst = "Does the document answer the query? Only output 'yes' or 'no'."
        # inst = "Is the document relevant to the query? Only output 'yes' or 'no'."
    elif format == "pairwise":
        # inst = "Which document is more relevant to the query? Only output 'A' or 'B'."
        inst = "Is document A more relevant to the query? Only output 'yes' or 'no'."

    dataloader = []
    for i, item in enumerate(data):
        if format == "pointwise" and use_pos is True:
            query = item["query"]
            document = item["positive_document"]
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 1
            input_ids, query_range, doc_range, inst_range = get_pointwise_ranges_v2(tokenizer, prompt, query, document, inst, query_seqlen, doc_seqlen, use_chat)
        elif format == "pointwise" and use_pos is False:
            query = item["query"]
            if use_hard_neg:
                document = random.choice(item["hard_negative_document"])
            else:
                document = random.choice(item["random_negative_document"])
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 0
            input_ids, query_range, doc_range, inst_range = get_pointwise_ranges_v2(tokenizer, prompt, query, document, inst, query_seqlen, doc_seqlen, use_chat)
        elif format == "pointwise" and use_pos is None:
            query = item["query"]
            document = random.choice(item["random_negative_document"] + item["hard_negative_document"] + [item["positive_document"]] * 4)
            prompt = prompt_template.format(query=query, document=document, inst=inst)
            label = 1 if document == item["positive_document"] else 0
            input_ids, query_range, doc_range, inst_range = get_pointwise_ranges_v2(tokenizer, prompt, query, document, inst, query_seqlen, doc_seqlen, use_chat)
        elif format == "pairwise":
            query = item["query"]
            document1 = item["positive_document"]
            if use_hard_neg:
                document2 = random.choice(item["hard_negative_document"])
            else:
                document2 = random.choice(item["random_negative_document"])
            label = 1
            # document1 is more relevant if `use_pos=True`, else document2 is more relevant
            if use_pos is False:
                document1, document2 = document2, document1
                label = 0
            if use_pos is None:
                label = random.choice([0, 1])
                if label == 0:
                    document1, document2 = document2, document1

            prompt = prompt_template.format(
                query=query,
                document1=document1,
                document2=document2,
                inst=inst
            )
            input_ids, query_range, doc_range, inst_range = get_pairwise_ranges_v2(tokenizer, prompt, query, document1, document2, inst, query_seqlen, doc_seqlen, use_chat)

        elif format == "listwise":
            pass

        dataloader.append((input_ids, query_range, doc_range, inst_range, label))

    return dataloader

