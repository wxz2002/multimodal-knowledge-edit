from easyeditor import apply_rome_to_multimodal_model
from easyeditor import FT
from easyeditor import FTMultimodalHparams, ROMEMultimodalHyperParams
from easyeditor import MultimodalTrainer, VQADataset
from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons
from torch.utils.data import Dataset, DataLoader
import fcntl
import argparse
import copy
import torch
import os
from PIL import Image
from transformers import LlamaConfig, LlamaTokenizer, LlavaForConditionalGeneration, LlamaForCausalLM, LlavaProcessor
import json
from tqdm import tqdm
import random

def get_kn_neurons(model, data, image_path, tokenizer, processor, device, flag):
    model = model.to(device)
    kn = KnowledgeNeurons(model, tokenizer, model_type='llava', device=device, processor=processor)
    for i, hop in enumerate(data['multimodal_hops']):
        if hop['image'] is not None:
            image = Image.open(os.path.join(image_path, hop['image']))
            single_hop_prompt = '<image> Qustion:{} Answer:'.format(hop['question'])
        else:
            single_hop_prompt = 'Qustion:{} Answer:'.format(hop['question'])
            hop['answer'] = data['knowledge_edit']['answer_new']
            image = None

        if i == 0:
            a_to_b_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                               batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
        else:
            b_to_c_neurons = kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                               batch_size=1, steps=20, adaptive_threshold=0.15, image=image)
    multi_hop_prompt = '<image> Qustion:{} Answer:'.format(data['knowledge_edit']['image_question'])
    multi_image = Image.open(os.path.join(image_path, data['image']))
    a_to_c_neurons = kn.get_coarse_neurons(prompt=multi_hop_prompt, ground_truth=data['knowledge_edit']['answer_new'],
                                                         batch_size=1, steps=20, adaptive_threshold=0.15, image=multi_image)
    
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

    if flag:
        return a_to_b_neurons, b_to_c_neurons, a_to_c_neurons, a_to_b_neurons, b_to_c_neurons, a_to_c_neurons
    else:
        return a_to_b_neurons, b_to_c_neurons, a_to_c_neurons, stability_a_to_b_neurons, stability_b_to_c_neurons, stability_a_to_c_neurons

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="./rome_results/stability_original_answer_datas.json")
    parser.add_argument('--model_path', type=str, default="../hugging_cache/llava-v1.5-7b-conv")
    parser.add_argument('--image_path', type=str, default="../new_download_images")
    parser.add_argument('--data_size', type=int, default=80)
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('--chunk_id', type=int, default=0)
    args = parser.parse_args()
    ft_hparams = FTMultimodalHparams.from_hparams('./hparams/FT/llava.yaml')

    original_data = json.load(open(args.dataset_path, 'r'))
    original_data = original_data[:args.data_size]
    chunk_size = len(original_data) // args.num_chunks
    chunk_start = args.chunk_id * chunk_size
    chunk_end = min((args.chunk_id + 1) * chunk_size, len(original_data))
    data = original_data[chunk_start:chunk_end]


    if os.path.exists('./stability_results/ft_unimodal_results.jsonl'):
        ft_unimodal_subjects = []
        with open('./stability_results/ft_unimodal_results.jsonl', 'r') as f:
            lines = f.readlines()
            for line in lines:
                ft_unimodal_subjects.append(json.loads(line)['subject'])
    else :
        ft_unimodal_subjects = []
    if os.path.exists('./stability_results/ft_multimodal_results.jsonl'):
        ft_multimodal_subjects = []
        with open('./stability_results/ft_multimodal_results.jsonl', 'r') as f:
            lines = f.readlines()
            for line in lines:
                ft_multimodal_subjects.append(json.loads(line)['subject'])
    else :
        ft_multimodal_subjects = []
    
    eval_ds = VQADataset(args.dataset_path, config=ft_hparams, only_text=False, annotation=data)
    val_loader = DataLoader(eval_ds, batch_size=ft_hparams.val_batch_size,
                                       shuffle=False, collate_fn=eval_ds.collate_fn)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = LlavaProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16)
    language_model = model.language_model

    size = len(data)
    unimodal_flag = [False] * int(size * 0.8) + [True] * (size - int(size * 0.8))
    multimodal_flag = [False] * int(size * 0.5) + [True] * (size - int(size * 0.5))
    random.shuffle(unimodal_flag)
    random.shuffle(multimodal_flag)

    for i, batch in tqdm(enumerate(val_loader)):
        # ft unimodal edit and get knowledge neurons
        if data[i]['subject'] not in ft_unimodal_subjects:
            model = model.to('cpu')
            copy_model = copy.deepcopy(model)
            ft_trainer = MultimodalTrainer(config=ft_hparams, train_set=eval_ds, val_set=eval_ds)
            ft_edited_model, model_info = ft_trainer.model.edit(batch["knowledge_edit_without_image"], batch["cond"])
            copy_model.language_model.model.layers = ft_edited_model.model.model.layers.to(dtype=torch.float16)

            ft_a_to_b_neurons, ft_b_to_c_neurons, ft_a_to_c_neurons, ft_stability_a_to_b_neurons, ft_stability_b_to_c_neurons, ft_stability_a_to_c_neurons = get_kn_neurons(copy_model, data[i], args.image_path, tokenizer, processor, device, unimodal_flag[i])

            ft_unimodal_result = {
                'subject': data[i]['subject'],
                'knowledge_edit': data[i]['knowledge_edit'],
                'multimodal_hops': data[i]['multimodal_hops'],
                'a_to_b': ft_a_to_b_neurons,
                'b_to_c': ft_b_to_c_neurons,
                'a_to_c': ft_a_to_c_neurons,
                'stability_a_to_b': ft_stability_a_to_b_neurons,
                'stability_b_to_c': ft_stability_b_to_c_neurons,
                'stability_a_to_c': ft_stability_a_to_c_neurons
            }
            ft_unimodal_result_json = json.dumps(ft_unimodal_result)
            # 打开文件并获取锁
            with open('./stability_results/ft_unimodal_results.jsonl', 'a') as f:
                # 获取文件锁（阻塞模式）
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(ft_unimodal_result_json + '\n')
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
            del ft_trainer
            del ft_edited_model
            del copy_model

        # ft multimodal edit and get knowledge neurons
        if data[i]['subject'] not in ft_multimodal_subjects:
            model = model.to('cpu')
            copy_model = copy.deepcopy(model)
            ft_trainer = MultimodalTrainer(config=ft_hparams, train_set=eval_ds, val_set=eval_ds)
            ft_edited_model, model_info = ft_trainer.model.edit(batch["knowledge_edit"], batch["cond"])
            copy_model.language_model.model.layers = ft_edited_model.model.model.layers.to(dtype=torch.float16)

            ft_a_to_b_neurons, ft_b_to_c_neurons, ft_a_to_c_neurons, ft_stability_a_to_b_neurons, ft_stability_b_to_c_neurons, ft_stability_a_to_c_neurons = get_kn_neurons(copy_model, data[i], args.image_path, tokenizer, processor, device, multimodal_flag[i])
            ft_multimodal_result = {
                'subject': data[i]['subject'],
                'knowledge_edit': data[i]['knowledge_edit'],
                'multimodal_hops': data[i]['multimodal_hops'],
                'a_to_b': ft_a_to_b_neurons,
                'b_to_c': ft_b_to_c_neurons,
                'a_to_c': ft_a_to_c_neurons,
                'stability_a_to_b': ft_stability_a_to_b_neurons,
                'stability_b_to_c': ft_stability_b_to_c_neurons,
                'stability_a_to_c': ft_stability_a_to_c_neurons
            }
            ft_multimodal_result_json = json.dumps(ft_multimodal_result)
            # 打开文件并获取锁
            with open('./stability_results/ft_multimodal_results.jsonl', 'a') as f:
                # 获取文件锁（阻塞模式）
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(ft_multimodal_result_json + '\n')
                finally:
                    # 释放文件锁
                    fcntl.flock(f, fcntl.LOCK_UN)
                    
            del ft_trainer
            del ft_edited_model
            del copy_model

        
