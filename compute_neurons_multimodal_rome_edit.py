from easyeditor import apply_rome_to_multimodal_model, ROMEMultimodalHyperParams
from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons
from PIL import Image
import torch
import json
import argparse
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import copy
from transformers import LlamaConfig, LlamaTokenizer, LlavaForConditionalGeneration, LlamaForCausalLM, LlavaProcessor


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="../our_dataset/final_image_rephrase_test_multimodal_hops.json")
parser.add_argument('--model_path', type=str, default="../hugging_cache/llava-v1.5-7b-conv")
parser.add_argument('--image_path', type=str, default="../new_download_images")
args = parser.parse_args()

def get_kn_neurons(datas, model_path, processor, tokenizer, image_path, device_id, hparams, mode):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    language_model = model.language_model
    results = []
    for data in tqdm(datas):
        # before edit
        model = model.to(device)
        kn = KnowledgeNeurons(model, tokenizer, model_type='llava', device=device, processor=processor)
        for i, hop in enumerate(data['multimodal_hops']):
            if hop['image'] is not None:
                image = Image.open(os.path.join(image_path, hop['image']))
                single_hop_prompt = '<image> Qustion:{} Answer:'.format(hop['question'])
            else :
                single_hop_prompt = 'Qustion:{} Answer:'.format(hop['question'])
                image = None
            if i==0:
                before_edit_a_to_b_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                       batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
            else :
                before_edit_b_to_c_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                       batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
        multi_hop_prompt = '<image> Qustion:{} Answer:'.format(data['knowledge_edit']['image_question'])
        multi_image = Image.open(os.path.join(image_path, data['image']))
        before_edit_a_to_c_neurons = kn.get_coarse_neurons(prompt=multi_hop_prompt, ground_truth=data['knowledge_edit']['answer_true'],
                                                   batch_size=1, steps=20, adaptive_threshold=0.15, image=multi_image)
        
        del kn
        
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

        # after edit
        kn = KnowledgeNeurons(edited_model, tokenizer, model_type='llava', device=device, processor=processor)
        for i, hop in enumerate(data['multimodal_hops']):
            if hop['image'] is not None:
                image = Image.open(os.path.join(image_path, hop['image']))
                single_hop_prompt = '<image> Qustion:{} Answer:'.format(hop['question'])
            else :
                single_hop_prompt = 'Qustion:{} Answer:'.format(hop['question'])
                hop['answer'] = data['knowledge_edit']['answer_new']
                image = None
            if i==0:
                after_edit_a_to_b_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                       batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
            else :
                after_edit_b_to_c_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                       batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
        multi_hop_prompt = '<image> Qustion:{} Answer:'.format(data['knowledge_edit']['image_question'])
        multi_image = Image.open(os.path.join(image_path, data['image']))
        after_edit_a_to_c_neurons = kn.get_coarse_neurons(prompt=multi_hop_prompt, ground_truth=data['knowledge_edit']['answer_true'],
                                                    batch_size=1, steps=20, adaptive_threshold=0.15, image=multi_image)
        
        del kn
        del edited_model
        del edited_language_model

        # compare neurons
        before_single_hop_neurons = before_edit_a_to_b_neurons + before_edit_b_to_c_neurons
        before_multi_hop_neurons = before_edit_a_to_c_neurons
        before_shared_neurons = []
        for before_single_hop_neuron in before_single_hop_neurons:
            if before_single_hop_neuron in before_multi_hop_neurons:
                before_shared_neurons.append(before_single_hop_neuron)

        a_to_b_shared_neurons = []
        for before_edit_a_to_b_neuron in before_edit_a_to_b_neurons:
            if before_edit_a_to_b_neuron in after_edit_a_to_b_neurons:
                a_to_b_shared_neurons.append(before_edit_a_to_b_neuron)
        b_to_c_shared_neurons = []
        for before_edit_b_to_c_neuron in before_edit_b_to_c_neurons:
            if before_edit_b_to_c_neuron in after_edit_b_to_c_neurons:
                b_to_c_shared_neurons.append(before_edit_b_to_c_neuron)
        a_to_c_shared_neurons = []
        for before_edit_a_to_c_neuron in before_edit_a_to_c_neurons:
            if before_edit_a_to_c_neuron in after_edit_a_to_c_neurons:
                a_to_c_shared_neurons.append(before_edit_a_to_c_neuron)

        result = {
            "subject": data['subject'],
            "knowledge_edit": data['knowledge_edit'],
            "multimodal_hops": data['multimodal_hops'],
            'before_single_hop_neurons': before_single_hop_neurons,
            'before_multi_hop_neurons': before_multi_hop_neurons,
            'before_shared_neurons': before_shared_neurons,
            "before_edit_a_to_b_neurons": before_edit_a_to_b_neurons,
            "after_edit_a_to_b_neurons": after_edit_a_to_b_neurons,
            "a_to_b_shared_neurons": a_to_b_shared_neurons,
            "before_edit_b_to_c_neurons": before_edit_b_to_c_neurons,
            "after_edit_b_to_c_neurons": after_edit_b_to_c_neurons,
            "b_to_c_shared_neurons": b_to_c_shared_neurons,
            "before_edit_a_to_c_neurons": before_edit_a_to_c_neurons,
            "after_edit_a_to_c_neurons": after_edit_a_to_c_neurons,
            "a_to_c_shared_neurons": a_to_c_shared_neurons
        }
        results.append(result)
        with open('./neurons/{}_edit_results_{}.json'.format(mode, device_id), 'w') as f:
            json.dump(results, f, indent=4)
    return results

def prepare_data_for_rome(request):
    rome_request = {
        'prompt': request['knowledge_edit']['question'],
        'target_new': request['knowledge_edit']['answer_new'],
        'subject': request['subject'],
    }
    return [rome_request]

if __name__ == '__main__':
    hparams = ROMEMultimodalHyperParams.from_hparams('hparams/ROME/llava.yaml')
    processor = LlavaProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    can_rome_edit_datas = json.load(open("./rome_results/can_rome_edit_datas.json", 'r'))
    no_rome_edit_datas = json.load(open("./rome_results/no_rome_edit_datas.json", 'r'))
    no_rome_edit_datas = no_rome_edit_datas[:len(can_rome_edit_datas)]

    # datas can be rome edit
    chunk_size = len(can_rome_edit_datas) // 8
    can_rome_edit_data_chunks = [can_rome_edit_datas[i:i + chunk_size] for i in range(0, len(can_rome_edit_datas), chunk_size)]

    # open 8 threads to compute neurons
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_kn_neurons, data_chunk, args.model_path, processor, tokenizer, args.image_path, device_id, hparams, "can_rome_edit")
                   for device_id, data_chunk in enumerate(can_rome_edit_data_chunks)]

    can_rome_edit_results = []
    for future in futures:
        can_rome_edit_results.extend(future.result())

    with open('./neurons/final_can_rome_edit_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # datas can not be rome edit
    chunk_size = len(no_rome_edit_datas) // 8
    no_rome_edit_data_chunks = [no_rome_edit_datas[i:i + chunk_size] for i in range(0, len(no_rome_edit_datas), chunk_size)]
    
    # open 8 threads to compute neurons
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_kn_neurons, data_chunk, args.model_path, processor, tokenizer, args.image_path, device_id, hparams, "no_rome_edit")
                   for device_id, data_chunk in enumerate(no_rome_edit_data_chunks)]
    
    no_rome_edit_results = []
    for future in futures:
        no_rome_edit_results.extend(future.result())
    
    with open('./neurons/final_no_rome_edit_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    