from easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons
import transformers
from PIL import Image
import torch
import json
import argparse
import os
import sys
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default="../our_dataset/final_image_rephrase_test_multimodal_hops.json")
parser.add_argument('--model_path', type=str, default="../hugging_cache/llava-1.5-7b-hf")
parser.add_argument('--image_path', type=str, default="../new_download_images")
args = parser.parse_args()


if __name__ == '__main__':
    model = transformers.LlavaForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float16).to('cuda:0')
    processor = transformers.LlavaProcessor.from_pretrained(args.model_path)
    datas = json.load(open(args.dataset_path))
    results = []
    num_of_single_hop_neurons, num_of_multi_hop_neurons = 0, 0
    for data in tqdm(datas):
        kn = KnowledgeNeurons(model, processor.tokenizer, model_type='llava', device='cuda', processor=processor)
        single_hop_neurons = []
        for hop in data['multimodal_hops']:
            if hop['image'] is not None:
                image = Image.open(os.path.join(args.image_path, hop['image']))
                single_hop_prompt = '<image> Qustion:{} Answer:'.format(hop['question'])
            else :
                single_hop_prompt = 'Qustion:{} Answer:'.format(hop['question'])
                image = None
            single_hop_neurons += kn.get_coarse_neurons(prompt=single_hop_prompt, ground_truth=hop['answer'],
                                                       batch_size=1, steps=10, adaptive_threshold=0.3, image=image)
        multi_hop_prompt = '<image> Qustion:{} Answer:'.format(data['knowledge_edit']['image_question'])
        multi_image = Image.open(os.path.join(args.image_path, data['image']))
        multi_hop_neurons = kn.get_coarse_neurons(prompt=multi_hop_prompt, ground_truth=data['knowledge_edit']['answer_true'],
                                                   batch_size=1, steps=10, adaptive_threshold=0.3, image=multi_image)
        shared_neurons = []
        for single_hop_neuron in single_hop_neurons:
            if single_hop_neuron in multi_hop_neurons:
                shared_neurons.append(single_hop_neuron)
        
        result = {
            'subject': data['subject'],
            'multimodal_hops': data['multimodal_hops'],
            'single_hop_neurons': single_hop_neurons,
            'multi_hop_neurons': multi_hop_neurons,
            'shared_neurons': shared_neurons
        }
        results.append(result)
        
        with open('./neurons/results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        


# image=None
# inputs = processor(prompt, image, return_tensors="pt").to('cuda:0')
# print(inputs)
# vis_processor = transformers.CLIPImageProcessor.from_pretrained('../hugging_cache/clip-vit-large-patch14-336')
# tokenizer = transformers.AutoTokenizer.from_pretrained("../hugging_cache/llava-v1.5-7b")
# tokenizer.pad_token_id = tokenizer.eos_token_id

# kn = KnowledgeNeurons(model, processor.tokenizer, model_type='llava', device='cuda:0', processor=processor)

# prompt = "Who is the author of The French Lieutenant's Woman?"
# ground_truth = "John Fowles"
# single_hop_neurons = kn.get_coarse_neurons(prompt=prompt, ground_truth=ground_truth, batch_size=1, steps=20, adaptive_threshold=0.3)
# print(single_hop_neurons)