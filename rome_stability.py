from easyeditor import apply_rome_to_multimodal_model, ROMEMultimodalHyperParams
from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons
from PIL import Image
import torch
import json
import argparse
import os
import sys
from tqdm import tqdm
from multiprocessing import Pool, set_start_method, current_process
import random
import copy
import fcntl
from transformers import LlamaConfig, LlamaTokenizer, LlavaForConditionalGeneration, LlamaForCausalLM, LlavaProcessor
import time

def get_kn_neurons(data_chunk, model_path, image_path, hparams, device_id, mode):
    output_path = f'./stability_results/{mode}_results.jsonl'

    if os.path.exists(output_path):
        subjects = []
        with open(output_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                subjects.append(data['subject'])
    else:
        subjects = []
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    hparams.device = device_id
    processor = LlavaProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    language_model = model.language_model
    results = []

    for data in tqdm(data_chunk):
        if data['subject'] in subjects:
            continue

        # knowledge edit
        model = model.to('cpu')
        edited_model = copy.deepcopy(model).to(device)
        rome_request = prepare_data_for_rome(data)
        edited_language_model, weights_copy = apply_rome_to_multimodal_model(
            language_model,
            tokenizer,
            rome_request,
            hparams,
            copy=True,
            return_orig_weights=True,
            keep_original_weight=True
        )
        edited_model.language_model = edited_language_model.to(device)

        # knowledge edit neurons
        kn = KnowledgeNeurons(edited_model, tokenizer, model_type='llava', device=device, processor=processor)
        for i, hop in enumerate(data['multimodal_hops']):
            if hop['image'] is not None:
                image = Image.open(os.path.join(image_path, hop['image']))
                single_hop_prompt = '<image> Question: {} Short Answer: '.format(hop['question'])
            else:
                single_hop_prompt = 'Question: {} Short Answer: '.format(hop['question'])
                hop['answer'] = data['knowledge_edit']['answer_new']
                image = None

            if i == 0:
                after_edit_a_to_b_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                                  batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
            else:
                after_edit_b_to_c_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                                  batch_size=1, steps=20, adaptive_threshold=0.15, image=image)

        multi_hop_prompt = '<image> Question: {} Short Answer: '.format(data['knowledge_edit']['image_question'])
        multi_image = Image.open(os.path.join(image_path, data['image']))
        after_edit_a_to_c_neurons = kn.get_coarse_neurons(prompt=multi_hop_prompt, ground_truth=data['knowledge_edit']['answer_new'],
                                                          batch_size=1, steps=20, adaptive_threshold=0.15, image=multi_image)


        # stability neurons
        stability_a_to_b_prompt = '<image> Question: {} Short Answer: '.format(data['multimodal_hops'][0]['question'])
        stability_a_to_b_image = Image.open(os.path.join(image_path, data['same_entity_image']))
        stability_a_to_b_neurons = kn.get_coarse_neurons(prompt=stability_a_to_b_prompt, ground_truth=data['same_type_entity_question']['same_type_entity'],
                                                            batch_size=1, steps=20, adaptive_threshold=0.15, image=stability_a_to_b_image)
        
        stability_b_to_c_prompt = 'Question: {} Short Answer: '.format(data['same_type_entity_question']['question'])
        stability_b_to_c_image = None
        stability_b_to_c_neurons = kn.get_coarse_neurons(prompt=stability_b_to_c_prompt, ground_truth=data['knowledge_edit']['answer_new'],
                                                            batch_size=1, steps=20, adaptive_threshold=0.15, image=stability_b_to_c_image)

        stability_a_to_c_prompt = '<image> Question: {} Short Answer: '.format(data['same_type_entity_question']['image_question'])
        stability_a_to_c_image = Image.open(os.path.join(image_path, data['same_entity_image']))
        stability_a_to_c_neurons = kn.get_coarse_neurons(prompt=stability_a_to_c_prompt, ground_truth=data['knowledge_edit']['answer_new'],
                                                    batch_size=1, steps=20, adaptive_threshold=0.15, image=stability_a_to_c_image)
             
    
        del kn
        del edited_model
        del edited_language_model

        result = {
            "subject": data['subject'],
            "knowledge_edit": data['knowledge_edit'],
            "multimodal_hops": data['multimodal_hops'],
            "after_edit_a_to_b_neurons": after_edit_a_to_b_neurons,
            "after_edit_b_to_c_neurons": after_edit_b_to_c_neurons,
            "after_edit_a_to_c_neurons": after_edit_a_to_c_neurons,
            "stability_a_to_b_neurons": stability_a_to_b_neurons,
            "stability_b_to_c_neurons": stability_b_to_c_neurons,
            "stability_a_to_c_neurons": stability_a_to_c_neurons,
        }
        results.append(result)
        result_json = json.dumps(result)
        # 打开文件并获取锁
        with open(output_path, 'a') as f:
            # 获取文件锁（阻塞模式）
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(result_json + '\n')
            finally:
                # 释放文件锁
                fcntl.flock(f, fcntl.LOCK_UN)
    return results

def prepare_data_for_rome(request):
    rome_request = {
        'prompt': request['knowledge_edit']['question'],
        'target_new': request['knowledge_edit']['answer_new'],
        'subject': request['subject'],
    }
    return [rome_request]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="../our_dataset/final_image_rephrase_test_multimodal_hops.json")
    parser.add_argument('--model_path', type=str, default="../hugging_cache/llava-v1.5-7b-conv")
    parser.add_argument('--image_path', type=str, default="../new_download_images")
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('--chunk_id', type=int, default=0)
    args = parser.parse_args()
    hparams = ROMEMultimodalHyperParams.from_hparams('hparams/ROME/llava.yaml')
    stability_original_answer_datas = json.load(open("./rome_results/stability_original_answer_datas.json", 'r'))
    stability_no_answer_datas = json.load(open("./rome_results/stability_no_answer_datas.json", 'r'))
    data_size = min(len(stability_original_answer_datas), len(stability_no_answer_datas))
    stability_original_answer_datas = stability_original_answer_datas[:data_size]
    stability_no_answer_datas = stability_no_answer_datas[:data_size]

    chunk_size = len(stability_original_answer_datas) // args.num_chunks
    chunk_id = args.chunk_id
    print(f"chunk_size: {chunk_size}")
    print(f"device_id: {chunk_id}")
    # stability_original_answer_datas
    original_answer_data_chunks = [stability_original_answer_datas[i:i + chunk_size] for i in range(0, len(stability_original_answer_datas), chunk_size)]
    original_answer_results = get_kn_neurons(original_answer_data_chunks[chunk_id], args.model_path, args.image_path, hparams, 0, "stability_original_answer")
    
    # stability_no_answer_datas
    no_answer_data_chunks = [stability_no_answer_datas[i:i + chunk_size] for i in range(0, len(stability_no_answer_datas), chunk_size)]
    no_answer_results = get_kn_neurons(no_answer_data_chunks[chunk_id], args.model_path, args.image_path, hparams, 0, "stability_no_answer")
