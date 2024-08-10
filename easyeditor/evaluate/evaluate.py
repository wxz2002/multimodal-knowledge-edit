"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    elif hparams.alg_name=="GRACE":
        # ppl = PPL(model, tok, prompt, target_new, device)
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
        f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
        ret = {
            f"{key}_acc": acc,
            # f"{key}_PPL": ppl,
            f"{key}_F1":f1     
        }        
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
        ret = {
            f"{key}_acc": acc
        }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target_new, new_fact)
    ret = {
        f"rewrite_acc": edit_acc
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'],list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],record['locality'][locality_key]['prompt']):             
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                            f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans,label in zip(pre_neighbor,post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)
            
                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix=""
            else:
                icl_input = icl_examples
                x_prefix=f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'],list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],record['portability'][portability_key]['prompt']): 
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok,icl_input, x_a,
                                            f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, [''], record['portability'][portability_key]['ground_truth'],
                                                record['portability'][portability_key]['prompt'])
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                                f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]   
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()        
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

def compute_icl_multimodal_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    # vis_tok,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    # First, unpack rewrite evaluation record.
    original_question = record["original_question"]
    question = record["question"]
    answer = record["answer"]
    pred = record["pred"]
    rephrase_question = record["rephrase_question"] if 'rephrase_question' in record.keys() else None
    one_hop_question = record["one_hop_question"] if 'one_hop_question' in record.keys() else None
    one_hop_answer = record["one_hop_answer"] if 'one_hop_answer' in record.keys() else None
    same_entity_question = record["same_entity_question"] if 'same_entity_question' in record.keys() else None
    same_entity_answer = record["same_entity_answer"] if 'same_entity_answer' in record.keys() else None
    diff_entity_question = record["diff_entity_question"] if 'diff_entity_question' in record.keys() else None
    diff_entity_answer = record["diff_entity_answer"] if 'diff_entity_answer' in record.keys() else None
    locality_question = record["locality_question"] if 'locality_question' in record.keys() else None
    locality_answer = record["locality_answer"] if 'locality_answer' in record.keys() else None
    image_locality_answer = record["image_locality_answer"] if 'image_locality_answer' in record.keys() else None
    image_locality_question = record["image_locality_question"] if 'image_locality_question' in record.keys() else None
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    image_rephrase = record["image_rephrase"] if record["image_rephrase"].is_cuda else record["image_rephrase"].to(hparams.device)
    same_entity_image = record["same_entity_image"] if record["same_entity_image"].is_cuda else record["same_entity_image"].to(hparams.device)
    diff_entity_image = record["diff_entity_image"] if record["diff_entity_image"].is_cuda else record["diff_entity_image"].to(hparams.device)
    locality_image = record["locality_image"] if record["locality_image"].is_cuda else record["locality_image"].to(hparams.device)
    
    new_fact = f'New Fact: {original_question} {answer}\nPrompt: {question}'

    if pre_edit:
        edit_acc, pred_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       answer, question, image)
    else:
        edit_acc, pred_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              answer, new_fact, image)
    ret = {
        f"inner_acc": edit_acc,
        f"pred_ids": pred_ids
    }

    if pre_edit:
        language_model_acc, pred_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                        pred, original_question, None)
        multimodal_acc, pred_ids = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                        pred, question, image)
        ret['language_model_inner_acc'] = language_model_acc
        ret['multimodal_inner_acc'] = multimodal_acc
    
    if rephrase_question is not None:
        if pre_edit:
            rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                answer, rephrase_question, image)
        else :
            rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               answer, f'New Fact: {original_question} {answer}\nPrompt: {rephrase_question}', image)
        ret['rephrase_acc'] = rephrase_acc
    
    if one_hop_question is not None:
        if pre_edit:
            one_hop_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                 one_hop_answer, one_hop_question, image)
        else:
            one_hop_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               one_hop_answer, f'New Fact: {original_question} {answer}\nPrompt: {one_hop_question}', image)
        ret['one_hop_acc'] = one_hop_acc

    if image_rephrase is not None:
        if pre_edit:
            image_edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                 answer, question, image_rephrase)
        else:
            image_edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                 answer, new_fact, image_rephrase)
        ret['image_acc'] = image_edit_acc

        if pre_edit:
            image_rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                answer, rephrase_question, image_rephrase)
        else:
            image_rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                answer, f'New Fact: {original_question} {answer}\nPrompt: {rephrase_question}', image_rephrase)
        ret['image_rephrase_acc'] = image_rephrase_acc

        if pre_edit:
            image_one_hop_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                one_hop_answer, one_hop_question, image_rephrase)
        else:
            image_one_hop_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                one_hop_answer, f'New Fact: {original_question} {answer}\nPrompt: {one_hop_question}', image_rephrase)
        ret['image_one_hop_acc'] = image_one_hop_acc
        
    if same_entity_question is not None:
        if pre_edit:
            same_entity_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    same_entity_answer, same_entity_question, same_entity_image)
        else :
            same_entity_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                 same_entity_answer, f'New Fact: {original_question} {answer}\nPrompt: {same_entity_question}', same_entity_image)
        ret['same_entity_acc'] = same_entity_acc

        if pre_edit:
            same_entity_original_answer_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    answer, same_entity_question, same_entity_image)
        else :
            same_entity_original_answer_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    answer, f'New Fact: {original_question} {answer}\nPrompt: {same_entity_question}', same_entity_image)
        ret['same_entity_original_answer_acc'] = same_entity_original_answer_acc
    
    if diff_entity_question is not None:
        if pre_edit:
            diff_entity_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    diff_entity_answer, diff_entity_question, diff_entity_image)
        else:
            diff_entity_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    diff_entity_answer, f'New Fact: {original_question} {answer}\nPrompt: {diff_entity_question}', diff_entity_image)
        ret['diff_entity_acc'] = diff_entity_acc
        if pre_edit:
            diff_entity_original_answer_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    answer, diff_entity_question, diff_entity_image)
        else :
            diff_entity_original_answer_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    answer, f'New Fact: {original_question} {answer}\nPrompt: {diff_entity_question}', diff_entity_image)
        ret['diff_entity_original_answer_acc'] = diff_entity_original_answer_acc


    if "locality_question" in record.keys():
        if pre_edit:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    locality_answer, locality_question, None, is_loc=True) 
        else:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    locality_answer, f'New Fact: {original_question} {answer}\nPrompt: {locality_question}', None, is_loc=True) 
        ret['locality_output'] = locality_output
    
    if "image_locality_question" in record.keys():
        if pre_edit:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    image_locality_answer, image_locality_question, locality_image, is_loc=True) 
        else:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    image_locality_answer, f'New Fact: {original_question} {answer}\nPrompt: {image_locality_question}', locality_image, is_loc=True) 
        ret['image_locality_output'] = locality_image_output
            
    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        is_loc=False,
        neighborhood=False,
        display=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    
    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    if display:
        print(samples)

    return compute_multimodal_edit_quality(model, samples, hparams.exact_match) if not is_loc else compute_multimodal_edit_quality_demo(model, samples)

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4' or hparams.model_name == 'llava':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt,  add_special_tokens=False)) for prompt in prompts]  
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def prepare_multimodal_edit_demo(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    prompt_template = "Question: {} Short Answer: " 
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_template.format(prompt_) + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4' or hparams.model_name == 'llava':
        prompts_len = [len(tok.encode(prompt_template.format(prompt), add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt_template.format(prompt),  add_special_tokens=False)) for prompt in prompts]  
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def compute_multimodal_edit_quality(model, batch, exach_match=False, return_targ=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()    
            targ = outputs.labels.detach().cpu()
    full_pred_ids = logits.argmax(-1)
    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    if exach_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding
    
    if return_targ:
        return acc, pred_ids[:,-targ.shape[1]:].numpy(), targ
    else :
        return acc, pred_ids[:,-targ.shape[1]:].numpy()
  
def compute_multimodal_edit_quality_demo(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    logits_ = logits.clone()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy(), logits_

def compute_multimodal_edit_results(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    question = record["question"]
    answer = record["answer"]
    rephrase_question = record["rephrase_question"] if 'rephrase_question' in record.keys() else None
    one_hop_question = record["one_hop_question"] if 'one_hop_question' in record.keys() else None
    one_hop_answer = record["one_hop_answer"] if 'one_hop_answer' in record.keys() else None
    same_entity_question = record["same_entity_question"] if 'same_entity_question' in record.keys() else None
    same_entity_answer = record["same_entity_answer"] if 'same_entity_answer' in record.keys() else None
    diff_entity_question = record["diff_entity_question"] if 'diff_entity_question' in record.keys() else None
    diff_entity_answer = record["diff_entity_answer"] if 'diff_entity_answer' in record.keys() else None
    locality_question = record["locality_question"] if 'locality_question' in record.keys() else None
    locality_answer = record["locality_answer"] if 'locality_answer' in record.keys() else None
    image_locality_answer = record["image_locality_answer"] if 'image_locality_answer' in record.keys() else None
    image_locality_question = record["image_locality_question"] if 'image_locality_question' in record.keys() else None
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    image_rephrase = record["image_rephrase"] if record["image_rephrase"].is_cuda else record["image_rephrase"].to(hparams.device)
    same_entity_image = record["same_entity_image"] if record["same_entity_image"].is_cuda else record["same_entity_image"].to(hparams.device)
    diff_entity_image = record["diff_entity_image"] if record["diff_entity_image"].is_cuda else record["diff_entity_image"].to(hparams.device)
    locality_image = record["locality_image"] if record["locality_image"].is_cuda else record["locality_image"].to(hparams.device)
    
    ret['subject'] = record['subject']
    ret['original_question'] = question

    text_edit_inner = prepare_multimodal_edit_demo(hparams, tok, answer, record['original_question'], None)
    ret['text_inner_acc'], text_pred_ids, text_targ = compute_multimodal_edit_quality(model, text_edit_inner, hparams.exact_match, return_targ=True)
    ret['text_pred'] = tok.decode(text_pred_ids[0], skip_special_tokens=True)
    ret['text_answer'] = tok.decode(text_targ[0], skip_special_tokens=True)

    edit_inner = prepare_multimodal_edit_demo(hparams, tok, answer, question, image)
    ret['inner_acc'], pred_ids, targ = compute_multimodal_edit_quality(model, edit_inner, hparams.exact_match, return_targ=True)
    # ret['pred_ids'] = pred_ids
    # ret['target'] = targ
    ret['pred'] = tok.decode(pred_ids[0], skip_special_tokens=True)
    ret['answer'] = tok.decode(targ[0], skip_special_tokens=True)

    if rephrase_question is not None:
        rephrase_input = prepare_multimodal_edit_demo(hparams, tok, answer, rephrase_question, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, rephrase_input, hparams.exact_match)
    
    if one_hop_question is not None:
        one_hop_input = prepare_multimodal_edit_demo(hparams, tok, one_hop_answer, one_hop_question, image)
        ret['one_hop_acc'], _ = compute_multimodal_edit_quality(model, one_hop_input, hparams.exact_match)
    
    if image_rephrase is not None:
        image_input = prepare_multimodal_edit_demo(hparams, tok, answer, question, image_rephrase)
        ret['image_acc'], _ = compute_multimodal_edit_quality(model, image_input, hparams.exact_match)
        
        image_rephrase_input = prepare_multimodal_edit_demo(hparams, tok, answer, rephrase_question, image_rephrase)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, image_rephrase_input, hparams.exact_match)
        
        image_one_hop_input = prepare_multimodal_edit_demo(hparams, tok, one_hop_answer, one_hop_question, image_rephrase)
        ret['image_one_hop_acc'], _ = compute_multimodal_edit_quality(model, image_one_hop_input, hparams.exact_match)
    

    if same_entity_question is not None:
        same_entity_input = prepare_multimodal_edit_demo(hparams, tok, same_entity_answer, same_entity_question, same_entity_image)
        ret['same_entity_acc'], _ = compute_multimodal_edit_quality(model, same_entity_input, hparams.exact_match)
        
        same_entity_original_answer_input = prepare_multimodal_edit_demo(hparams, tok, answer, same_entity_question, same_entity_image)
        ret['same_entity_original_answer_acc'], _ = compute_multimodal_edit_quality(model, same_entity_original_answer_input, hparams.exact_match)
    
    if diff_entity_question is not None:
        diff_entity_input = prepare_multimodal_edit_demo(hparams, tok, diff_entity_answer, diff_entity_question, diff_entity_image)
        ret['diff_entity_acc'], _ = compute_multimodal_edit_quality(model, diff_entity_input, hparams.exact_match)
        
        diff_entity_original_answer_input = prepare_multimodal_edit_demo(hparams, tok, answer, diff_entity_question, diff_entity_image)
        ret['diff_entity_original_answer_acc'], _ = compute_multimodal_edit_quality(model, diff_entity_original_answer_input, hparams.exact_match)
    
    if "locality_question" in record.keys():
        locality_input = prepare_multimodal_edit_demo(hparams, tok, locality_answer, locality_question, None)
        _, _, ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality_input)
    
    if "image_locality_question" in record.keys():
        image_locality_input = prepare_multimodal_edit_demo(hparams, tok, image_locality_answer, image_locality_question, locality_image)
        _, _, ret['image_locality_output'] = compute_multimodal_edit_quality_demo(model, image_locality_input)

    # Form a list of lists of prefixes to test.
    del edit_inner, rephrase_input, one_hop_input, image_input, image_rephrase_input, image_one_hop_input, same_entity_input, same_entity_original_answer_input, diff_entity_input, diff_entity_original_answer_input, locality_input, image_locality_input
    return ret
  
def compute_multimodal_edit_results_demo(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(hparams.device)
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits


    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]

def compute_sent_metric(
    model,
    edited_model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    metric_kwargs: typing.Dict,
    device,
    test_generation=True
    ):
    
    if "llama" not in model_name:
        raise NotImplementedError("currently only support for llama")
        
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return labels
        
    same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])], device=device)
    edit_toks = {
        f"{k1}_{k2}": v2.to(device)
        for k1, v1 in {
            "inner": metric_kwargs["inner_all_qa"],
            "outer": metric_kwargs["outer_all_qa"],
        }.items()
        for k2, v2 in tok(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True,
        ).items()
    }
    for key in ["inner", "outer"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tok.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = torch.tensor(mask).to(device)

    with torch.no_grad():
        inner_base_logits = model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        inner_edit_logits = edited_model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        
        outer_base_logits = model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
        outer_edit_logits = edited_model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
    
    result = {
        "es": es(inner_base_logits, inner_edit_logits, edit_toks["inner_q_mask"], get_edit_labels(edit_toks["inner_input_ids"]), same_mask).item(),
        "dd": kl_loc_loss(outer_base_logits, outer_edit_logits, edit_toks["outer_q_mask"]).item(),
    }
    if  test_generation:
        result['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=metric_kwargs["inner_q"] if isinstance(metric_kwargs["inner_q"],list) else [metric_kwargs["inner_q"],], max_out_len=100)
    return result


def compute_per_ike_metric(
    example,
    model,
    tok,
    device,
    test_generation=False,
):
    with torch.no_grad():

        outer_base_logits = model(
            input_ids=example["outer_pre"]["input_ids"],
            attention_mask=example["outer_pre"]["attention_mask"],   
            labels=example["outer_pre"]["labels"],
        )["logits"]

        outer_edit_logits = model(
            input_ids=example["outer_edit"]["input_ids"],
            attention_mask=example["outer_edit"]["attention_mask"],   
            labels=example["outer_edit"]["labels"],
        )["logits"]
        
        loc_base_logits = model(
            input_ids=example["loc_pre"]["input_ids"],
            attention_mask=example["loc_pre"]["attention_mask"],   
            labels=example["loc_pre"]["labels"],
        )["logits"]

        loc_edit_logits = model(
            input_ids=example["loc_edit"]["input_ids"],
            attention_mask=example["loc_edit"]["attention_mask"],   
            labels=example["loc_edit"]["labels"],
        )["logits"]
        
        result = {
            "es": es_per_icl(example, outer_base_logits, outer_edit_logits)["acc_per"].item(),
            "dd": kl_loc_loss(loc_base_logits, loc_edit_logits, example["loc_pre"]["q_mask"]).item()
        }

        if test_generation:
            result.update(per_generation(
                model=model,
                tok=tok,
                max_out_len=60,
                target_per=example["target_per_text"],
                device=device,
                pre_q=example["pre_q"],
                edit_q=example["edit_q"],
                IKE=True,
            ))
        
    return result


def compute_per_metric(
    example,
    model,
    edited_model,
    tok,
    device,
    test_generation=False,
):
    with torch.no_grad():
        
        edit_q_mask = example["edit_outer"].pop("q_mask")
        kl_mask = example["loc"].pop("q_mask")
        
        outer_base_logits = model(**example["edit_outer"])["logits"]
        outer_edit_logits = edited_model.model(**example["edit_outer"])["logits"]
        
        loc_base_logits = model(**example["loc"])["logits"]
        loc_edit_logits = edited_model.model(**example["loc"])["logits"]
            
        result = {
            "es": es(
                pre_logits=outer_base_logits,
                edit_logits=outer_edit_logits,
                q_mask=edit_q_mask,
                labels=example["edit_outer"]["labels"],
                same_mask=example["same_mask"]
            ).item(),
            "dd": kl_loc_loss(
                pre=loc_base_logits, 
                post=loc_edit_logits, 
                mask=kl_mask
            ).item()
        }

        if test_generation:
            result.update(per_generation(
                model=model,
                edited_model=edited_model,
                tok=tok,
                max_out_len=60,
                target_per=example["target_per_text"][0],
                device=device,
                inner_q=example["inner_q"][0]
            ))
        
    return result
    

def compute_concept_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_concept_consistency = False,
    P = None
) -> typing.Dict:
    
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    if P is None:
        PMT= ''
    else:
        PMT= str(P)

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              PMT + rewrite_prompts, target_new, device=device, eval_metric=eval_metric)
    if test_concept_consistency:
        least_length_gen = 40
        ret['gen_concept_text']= test_concept_gen(model,tok,least_length_gen,
                                                PMT + rewrite_prompts,target_new,device=device)

    ret['locality'] = {}
    ret['instance'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                PMT + rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         PMT + record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    
    if 'instance' in record.keys() and any(record['instance']):
        for instance_key in record['instance'].keys():
            ret['instance'].update(
                {'instance_change': test_instance_change(model,tok,hparams.max_length,
                                     record['instance'][instance_key]['prompt'], 'yes', device=device, P=P)[0]}
            )

    return ret


def compute_safety_edit_quality(
    model,
    # model_name,
    # hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    # test_generation = False
    max_output_tokens: int = 600,
) -> typing.Dict:
    batch = [record["prompt"]] + record['general_prompt']
    DS, DG_onlyQ, DG_otherA, DG_otherQ, DG_otherAQ = test_safety_gen(model, tok, batch, device, max_output_tokens)
    ret = {
        "DS": DS,
        "DG_onlyQ": DG_onlyQ,
        "DG_otherA": DG_otherA,
        "DG_otherQ": DG_otherQ,
        "DG_otherAQ": DG_otherAQ
    }
    return ret
