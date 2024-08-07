"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers
from tqdm import tqdm
from transformers import LlavaProcessor
from copy import deepcopy

class VQADataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, only_text=True, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        if config.model_class == "Blip2OPT":
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        elif config.model_class == "LLAVA":
            vis_processor = transformers.CLIPImageProcessor.from_pretrained(config.clip_name)
        else:
            raise NotImplementedError("unknown model class")

        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32
        self.only_text = only_text

        # if self.config.model_class == "LLAVA" and self.only_text==False:
        #     self.prompt = "Question: <image> {} Short answer: "
        # else:
        self.prompt = "Question: {} Short answer: "     
        
        data = []
        if size is not None:
            self.annotation = self.annotation[:size] 
        for i, record in tqdm(enumerate(self.annotation), total=len(self.annotation), desc="Processing annotations"):
            
            if record['knowledge_edit']['answer_new'] == "":
                continue
            
            if not self.only_text:  
                image_path = os.path.join(self.rephrase_root, record['image'])
                rephrase_image_path = os.path.join(self.rephrase_root, record['image_rephrase'])
                same_entity_image_path = os.path.join(self.rephrase_root, record['same_entity_image'])
                diff_entity_image_path = os.path.join(self.rephrase_root, record['diff_entity_image'])
                locality_image_path = os.path.join(self.vis_root, record['locality_image'])

                try :
                    image = Image.open(image_path).convert("RGB")
                    rephrase_image = Image.open(rephrase_image_path).convert("RGB")
                    same_entity_image = Image.open(same_entity_image_path).convert("RGB")
                    diff_entity_image = Image.open(diff_entity_image_path).convert("RGB")
                    locality_image = Image.open(locality_image_path).convert("RGB")
                
                    if self.config.model_class == "LLAVA":
                        image = self.vis_processor(image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
                        rephrase_image = self.vis_processor(rephrase_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
                        same_entity_image = self.vis_processor(same_entity_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
                        diff_entity_image = self.vis_processor(diff_entity_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
                        locality_image = self.vis_processor(locality_image, return_tensors='pt')['pixel_values'].to(dtype=torch.float16)
                    else:
                        image = self.vis_processor(image)
                        rephrase_image = self.vis_processor(rephrase_image)  
                        same_entity_image = self.vis_processor(same_entity_image)
                        diff_entity_image = self.vis_processor(diff_entity_image)
                        locality_image = self.vis_processor(locality_image)  
                except Exception as e:
                    print("Error in processing image: ", image_path)  
                    continue
            else:
                image = None
                rephrase_image = None
                same_entity_image = None
                diff_entity_image = None
                locality_image = None
            
            if not self.only_text:
                item = {
                    'subject': record['subject'],
                    'original_question': record['knowledge_edit']['question'],
                    'question': record['knowledge_edit']['image_question'],
                    'answer': record['knowledge_edit']['answer_new'],
                    'pred': record['knowledge_edit']['answer_true'],
                    'rephrase_question': record['rephrase_question']['rephrase_image_question'],
                    'one_hop_question': record['one_hop_question']['one_hop_image_question'],
                    'one_hop_answer': record['one_hop_question']['answer_new'],
                    'same_entity_question': record['same_type_entity_question']['image_question'],
                    'same_entity_answer': record['same_type_entity_question']['answer'],
                    'diff_entity_question': record['diff_type_entity_question']['image_question'],
                    'diff_entity_answer': record['diff_type_entity_question']['answer'],
                    'locality_question': record['locality_question']['question'],
                    'locality_answer': record['locality_question']['answer'],
                    'image_locality_question': record['image_locality_question']['question'],
                    'image_locality_answer': record['image_locality_question']['answer'],
                    'image': image,
                    'image_rephrase': rephrase_image,
                    'same_entity_image': same_entity_image,
                    'diff_entity_image': diff_entity_image,
                    'locality_image': locality_image,

                    'cond': "{} >> {} || {}".format(
                        record['knowledge_edit']['answer_true'],
                        record['knowledge_edit']['answer_new'],
                        record['knowledge_edit']['question']
                    )
                }
            else:
                item = {
                    'subject': record['subject'],
                    'original_question': record['knowledge_edit']['question'],
                    'question': record['knowledge_edit']['question'],
                    'answer': record['knowledge_edit']['answer_new'],
                    'pred': record['knowledge_edit']['answer_true'],
                    'rephrase_question': record['rephrase_question']['rephrase_question'],
                    'one_hop_question': record['one_hop_question']['one_hop_question'],
                    'one_hop_answer': record['one_hop_question']['answer_new'],
                    'locality_question': record['locality_question']['question'],
                    'locality_answer': record['locality_question']['answer'],

                    'cond': "{} >> {} || {}".format(
                        record['knowledge_edit']['answer_true'],
                        record['knowledge_edit']['answer_new'],
                        record['knowledge_edit']['question']
                    )
                }
            
            data.append(item)
            
        # if size is not None:
        #     data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        
        question = [b['question'] for b in batch]
        original_question = [b['original_question'] for b in batch]
        answer = [" " + b['answer'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase_question = [b['rephrase_question'] for b in batch]
        one_hop_question = [b['one_hop_question'] for b in batch]
        one_hop_answer = [" " + b['one_hop_answer'] for b in batch]
        locality_question = [b['locality_question'] for b in batch]
        locality_answer = [" " + b['locality_answer'] for b in batch]
        if not self.only_text:
            same_entity_question = [b['same_entity_question'] for b in batch]
            same_entity_answer = [" " + b['same_entity_answer'] for b in batch]
            diff_entity_question = [b['diff_entity_question'] for b in batch]
            diff_entity_answer = [" " + b['diff_entity_answer'] for b in batch]
            image_locality_question = [b['image_locality_question'] for b in batch]
            image_locality_answer = [" " + b['image_locality_answer'] for b in batch]
            image = [b['image'] for b in batch]
            image_rephrase = [b['image_rephrase'] for b in batch]
            same_entity_image = [b['same_entity_image'] for b in batch]
            diff_entity_image = [b['diff_entity_image'] for b in batch]
            locality_image = [b['locality_image'] for b in batch]

        # knowledge_edit_without_image
        knowledge_edit_without_image = {}
        knowledge_edit_without_image['image'] = None
        if self.config.alg_name == "MEND" :
            knowledge_edit_without_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(question, answer)]
        else :
            knowledge_edit_without_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(original_question, answer)]
        knowledge_edit_without_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in question]
        knowledge_edit_without_image['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # knowledge_edit
        knowledge_edit = {}
        if not self.only_text:
            knowledge_edit['image'] = torch.stack(image, dim=0)
        else :
            knowledge_edit['image'] = None
        knowledge_edit['text_input'] = [self.prompt.format(q) + a for q, a in zip(question, answer)]
        knowledge_edit['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in question]
        knowledge_edit['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # image_question
        if self.only_text:
            edit_image_question = None
        else :
            edit_image_question = {}
            edit_image_question['image'] = torch.stack(image_rephrase, dim=0)
            edit_image_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(question, answer)]
            edit_image_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in question]
            edit_image_question['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # rephrase_question
        edit_rephrase_question = {}
        if not self.only_text:
            edit_rephrase_question['image'] = torch.stack(image, dim=0)
        else :
            edit_rephrase_question['image'] = None
        edit_rephrase_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(rephrase_question, answer)]
        edit_rephrase_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in rephrase_question]
        edit_rephrase_question['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # image_rephrase_question
        if self.only_text:
            edit_image_rephrase_question = None
        else :
            edit_image_rephrase_question = {}
            edit_image_rephrase_question['image'] = torch.stack(image_rephrase, dim=0)
            edit_image_rephrase_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(rephrase_question, answer)]
            edit_image_rephrase_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in rephrase_question]
            edit_image_rephrase_question['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # one_hop_question
        edit_one_hop_question = {}
        if not self.only_text:
            edit_one_hop_question['image'] = torch.stack(image, dim=0)
        else :
            edit_one_hop_question['image'] = None
        edit_one_hop_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(one_hop_question, one_hop_answer)]
        edit_one_hop_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in one_hop_question]
        edit_one_hop_question['labels'] = self.tok(one_hop_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # image_one_hop_question
        if self.only_text:
            edit_image_one_hop_question = None
        else :
            edit_image_one_hop_question = {}
            edit_image_one_hop_question['image'] = torch.stack(image_rephrase, dim=0)
            edit_image_one_hop_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(one_hop_question, one_hop_answer)]
            edit_image_one_hop_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in one_hop_question]
            edit_image_one_hop_question['labels'] = self.tok(one_hop_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # same_type_entity_question
        if self.only_text:
            edit_same_type_entity_question = None
        else :
            edit_same_type_entity_question = {}
            edit_same_type_entity_question['image'] = torch.stack(same_entity_image, dim=0)
            edit_same_type_entity_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(same_entity_question, same_entity_answer)]
            edit_same_type_entity_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in same_entity_question]
            edit_same_type_entity_question['labels'] = self.tok(same_entity_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # diff_type_entity_question
        if self.only_text:
            edit_diff_type_entity_question = None
        else :
            edit_diff_type_entity_question = {}
            edit_diff_type_entity_question['image'] = torch.stack(diff_entity_image, dim=0)
            edit_diff_type_entity_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(diff_entity_question, diff_entity_answer)]
            edit_diff_type_entity_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in diff_entity_question]
            edit_diff_type_entity_question['labels'] = self.tok(diff_entity_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # locality_question
        edit_locality_question = {}
        edit_locality_question['image'] = None
        edit_locality_question['text_input'] = [q + a for q, a in zip(locality_question, locality_answer)]
        edit_locality_question['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in locality_question]
        edit_locality_question['labels'] = self.tok(locality_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # image_locality_question
        if self.only_text:
            edit_image_locality_question = None
        else :
            edit_image_locality_question = {}
            edit_image_locality_question['image'] = torch.stack(locality_image, dim=0)
            edit_image_locality_question['text_input'] = [self.prompt.format(q) + a for q, a in zip(image_locality_question, image_locality_answer)]
            edit_image_locality_question['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in image_locality_question]
            edit_image_locality_question['labels'] = self.tok(image_locality_answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        if self.only_text:
            edit_same_type_entity_question_original_answer = None
        else:
            edit_same_type_entity_question_original_answer = {}
            edit_same_type_entity_question_original_answer['image'] = torch.stack(same_entity_image, dim=0)
            edit_same_type_entity_question_original_answer['text_input'] = [self.prompt.format(q) + a for q, a in zip(same_entity_question,answer)]
            edit_same_type_entity_question_original_answer['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in same_entity_question]
            edit_same_type_entity_question_original_answer['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        if self.only_text:
            edit_diff_type_entity_question_original_answer = None
        else:
            edit_diff_type_entity_question_original_answer = {}
            edit_diff_type_entity_question_original_answer['image'] = torch.stack(diff_entity_image, dim=0)
            edit_diff_type_entity_question_original_answer['text_input'] = [self.prompt.format(q) + a for q, a in zip(diff_entity_question,answer)]
            edit_diff_type_entity_question_original_answer['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in diff_entity_question]
            edit_diff_type_entity_question_original_answer['labels'] = self.tok(answer, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # # edit_inner_without_image
        # edit_inner_without_image = {}
        # edit_inner_without_image['image'] = None
        # edit_inner_without_image['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        # edit_inner_without_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
        # edit_inner_without_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        
        # # edit_inner
        # edit_inner = {}
        # if not self.only_text   :
        #     edit_inner['image'] = torch.stack(image, dim=0)
        # else :
        #     edit_inner['image'] = None
        # edit_inner['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        # edit_inner['labels'] = trg
        # edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
        # edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # # edit_outer
        # edit_outer = {}
        # if not self.only_text:
        #     edit_outer['image'] = torch.stack(image, dim=0)
        # else :
        #     edit_outer['image'] = None
        # edit_outer['text_input'] = [self.prompt.format(r) + t for r, t in zip(rephrase, trg)]
        # edit_outer['labels'] = trg
        # edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase]
        # edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]

            
        # # edit_outer_image
        # edit_outer_image = {}
        # if not self.only_text:
        #     edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        # else :
        #     edit_outer_image['image'] = None
        # edit_outer_image['text_input'] = [self.prompt.format(s) + t for s, t in zip(src, trg)]
        # edit_outer_image['labels'] = trg
        # edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
        # edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    
        # # loc
        # loc = {}
        # loc['image'] = None
        # loc['text_input'] = [q + a for q, a in zip(loc_q, loc_a)]
        # loc['labels'] = loc_a
        # loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
        # loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
        # # m_loc
        # loc_image = {}
        # if not self.only_text:
        #     loc_image['image'] = torch.stack(m_loc_image, dim=0)
        # else :
        #     loc_image['image'] = None
        # loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        # loc_image['labels'] = m_loc_a
        # loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
        # loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "knowledge_edit_without_image": knowledge_edit_without_image,
            "knowledge_edit": knowledge_edit,
            "image_question": edit_image_question,
            "rephrase_question": edit_rephrase_question,
            "image_rephrase_question": edit_image_rephrase_question,
            "one_hop_question": edit_one_hop_question,
            "image_one_hop_question": edit_image_one_hop_question,
            "same_type_entity_question": edit_same_type_entity_question,
            "diff_type_entity_question": edit_diff_type_entity_question,
            "locality_question": edit_locality_question,
            "image_locality_question": edit_image_locality_question,
            "cond": cond,
            "same_type_entity_question_original_answer": edit_same_type_entity_question_original_answer,
            "diff_type_entity_question_original_answer": edit_diff_type_entity_question_original_answer
        }
        # if self.config.model_name == "llava-1.5":
        #     batch = {
        #     "edit_inner_without_image": self.process_for_llava(edit_inner_without_image),
        #     "edit_inner": self.process_for_llava(edit_inner),
        #     "edit_outer": self.process_for_llava(edit_outer),
        #     "edit_outer_image": self.process_for_llava(edit_outer_image),
        #     "loc": self.process_for_llava(loc),
        #     "loc_image": self.process_for_llava(loc_image),
        #     "cond": cond
        #     }
            
        return dict_to(batch, self.config.device)

    # def process_for_llava(self, batch):
    #     outputs = {}
    #     outputs["pixel_values"] = batch["image"]
    #     inputs = self.tok(batch["text_input"],return_tensors="pt",padding="longest",truncation=True,add_special_tokens=False)
    #     outputs["input_ids"] = inputs["input_ids"]
    #     outputs["attention_mask"] = inputs["attention_mask"]
    #     targets = inputs["input_ids"].masked_fill(
    #             inputs["input_ids"] == self.tok.pad_token_id, -100
    #         )
    #     if batch['prompts_len']:
    #     # targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
    #         for i, prompt_len in enumerate(batch['prompts_len']):
    #             targets[i, :prompt_len] = -100
    #     outputs["labels"] = targets
    #     # outputs["prompts_len"] = batch["prompts_len"]
    #     return outputs
