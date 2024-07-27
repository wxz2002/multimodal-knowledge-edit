from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultimodalTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)
        print("super init")
        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None
        print("lr_opt")
        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def train_edit_step(self, batch, training:bool):
        self.model.train(training)
        self.original_model.train(training)
        with torch.no_grad():
            locality_base_outputs = self.model(batch["locality_question"])
            if not isinstance(locality_base_outputs, torch.Tensor):
                locality_base_logits = locality_base_outputs.logits
            else:  
                locality_base_logits = locality_base_outputs
            
            if self.config.alg == "SERAC_MULTI" and self.config.model_name == "minigpt4" and locality_base_logits.shape[2] == 32001:
                locality_base_logits = locality_base_logits[:, :, :-1]

        start = time.time()
        edited_model, model_info = self.model.edit(batch["knowledge_edit_without_image"], batch["cond"])
        # edited_model = self.model
        edit_time = time.time() - start 

        with torch.set_grad_enabled(training):
            # Editing loss
            inner_edit_outputs = edited_model(batch["knowledge_edit"])
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                inner_batch_labels = inner_edit_outputs.labels
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["knowledge_edit"]["labels"]

            rephrase_edit_outputs = edited_model(batch["rephrase_question"])
            if not isinstance(rephrase_edit_outputs, torch.Tensor):
                rephrase_edit_logits = rephrase_edit_outputs.logits
                rephrase_batch_labels = rephrase_edit_outputs.labels
            else:
                rephrase_edit_logits = rephrase_edit_outputs
                rephrase_batch_labels = batch["rephrase_question"]["labels"]

            one_hop_edit_outputs = edited_model(batch["one_hop_question"])
            if not isinstance(one_hop_edit_outputs, torch.Tensor):
                one_hop_edit_logits = one_hop_edit_outputs.logits
                one_hop_batch_labels = one_hop_edit_outputs.labels
            else:
                one_hop_edit_logits = one_hop_edit_outputs
                one_hop_batch_labels = batch["one_hop_question"]["labels"]

            l_rephrase_edit = self.model.edit_loss_fn(self.config, rephrase_edit_logits, rephrase_batch_labels, multimodal=True)["nll"]
            l_one_hop_edit = self.model.edit_loss_fn(self.config, one_hop_edit_logits, one_hop_batch_labels, multimodal=True)["nll"]
            
            # Collect some useful metrics
            with torch.no_grad():
                rephrase_edit_dict = self.model.edit_loss_fn(self.config, rephrase_edit_logits, rephrase_batch_labels, multimodal=True)
                one_hop_edit_dict = self.model.edit_loss_fn(self.config, one_hop_edit_logits, one_hop_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
            
            locality_edit_outputs = edited_model(batch["locality_question"])
            if not isinstance(locality_edit_outputs, torch.Tensor):
                locality_edit_logits = locality_edit_outputs.logits
                kl_mask = locality_edit_outputs.attention_mask
            else:
                locality_edit_logits = locality_edit_outputs
                kl_mask = torch.ones(locality_edit_logits.shape[0], locality_edit_logits.shape[1]).to(locality_edit_logits.device)

            l_loc = kl_loc_loss(locality_base_logits.detach(), locality_edit_logits, mask=kl_mask)

        if l_rephrase_edit.isnan():
            print("l_rephrase_edit is nan")
            print("input: ", batch["rephrase_question"]['text_input'])
        elif l_one_hop_edit.isnan():
            print("l_one_hop_edit is nan")
            print("input: ", batch["one_hop_question"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["locality_question"]['text_input'])

        l_total_edit = self.config.cedit * l_rephrase_edit + self.config.cloc * l_loc + self.config.iedit * l_one_hop_edit

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        locality_edit_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(locality_edit_logits, dim=-1), k=1, dim=-1).indices
        locality_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(locality_base_logits, dim=-1), k=1, dim=-1).indices

        info_dict = {}
        info_dict['loss/edit'] = l_rephrase_edit.item()
        info_dict['loss/one_hop_edit'] = l_one_hop_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['rephrase/acc'] = rephrase_edit_dict["acc"].item()
        info_dict['one_hop_edit/acc'] = one_hop_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(locality_edit_logits_softmax_top_k.view(-1) == locality_base_logits_softmax_top_k.view(-1))/locality_edit_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_rephrase_edit, l_loc, l_base, info_dict
        
    
    def test_edit_step(self, batch, training:bool):
        self.model.train(training)
        self.original_model.train(training)
        with torch.no_grad():
            locality_base_outputs = self.model(batch["locality_question"])
            if not isinstance(locality_base_outputs, torch.Tensor):
                locality_base_logits = locality_base_outputs.logits
            else:  
                locality_base_logits = locality_base_outputs
            
            image_locality_base_outputs = self.model(batch["image_locality_question"])
            if not isinstance(image_locality_base_outputs, torch.Tensor):
                image_locality_base_logits = image_locality_base_outputs.logits
            else:
                image_locality_base_logits = image_locality_base_outputs

            if self.config.alg == "SERAC_MULTI" and self.config.model_name == "minigpt4" and locality_base_logits.shape[2] == 32001:
                locality_base_logits = locality_base_logits[:, :, :-1]
                image_locality_base_logits = image_locality_base_logits[:, :, :-1]
            

        start = time.time()
        edited_model, model_info = self.model.edit(batch["knowledge_edit_without_image"], batch["cond"])
        # edited_model = self.model
        edit_time = time.time() - start 

        with torch.set_grad_enabled(training):
            # Editing loss
            # inner
            inner_edit_outputs = edited_model(batch["knowledge_edit"])
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                inner_batch_labels = inner_edit_outputs.labels
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["knowledge_edit"]["labels"]
            # image
            image_edit_outputs = edited_model(batch["image_question"])
            if not isinstance(image_edit_outputs, torch.Tensor):
                image_edit_logits = image_edit_outputs.logits
                image_batch_labels = image_edit_outputs.labels
            else:
                image_edit_logits = image_edit_outputs
                image_batch_labels = batch["image_question"]["labels"]
            # rephrase
            rephrase_edit_outputs = edited_model(batch["rephrase_question"])
            if not isinstance(rephrase_edit_outputs, torch.Tensor):
                rephrase_edit_logits = rephrase_edit_outputs.logits
                rephrase_batch_labels = rephrase_edit_outputs.labels
            else:
                rephrase_edit_logits = rephrase_edit_outputs
                rephrase_batch_labels = batch["rephrase_question"]["labels"]
            # image rephrase
            image_rephrase_edit_outputs = edited_model(batch["image_rephrase_question"])
            if not isinstance(image_rephrase_edit_outputs, torch.Tensor):
                image_rephrase_edit_logits = image_rephrase_edit_outputs.logits
                image_rephrase_batch_labels = image_rephrase_edit_outputs.labels
            else:
                image_rephrase_edit_logits = image_rephrase_edit_outputs
                image_rephrase_batch_labels = batch["image_rephrase_question"]["labels"]
            # one hop
            one_hop_edit_outputs = edited_model(batch["one_hop_question"])
            if not isinstance(one_hop_edit_outputs, torch.Tensor):
                one_hop_edit_logits = one_hop_edit_outputs.logits
                one_hop_batch_labels = one_hop_edit_outputs.labels
            else:
                one_hop_edit_logits = one_hop_edit_outputs
                one_hop_batch_labels = batch["one_hop_question"]["labels"]
            # image one hop
            image_one_hop_edit_outputs = edited_model(batch["image_one_hop_question"])
            if not isinstance(image_one_hop_edit_outputs, torch.Tensor):
                image_one_hop_edit_logits = image_one_hop_edit_outputs.logits
                image_one_hop_batch_labels = image_one_hop_edit_outputs.labels
            else:
                image_one_hop_edit_logits = image_one_hop_edit_outputs
                image_one_hop_batch_labels = batch["image_one_hop_question"]["labels"]
            # same entity
            same_entity_edit_outputs = edited_model(batch["same_type_entity_question"])  
            if not isinstance(same_entity_edit_outputs, torch.Tensor):
                same_entity_edit_logits = same_entity_edit_outputs.logits
                same_entity_batch_labels = same_entity_edit_outputs.labels
            else:
                same_entity_edit_logits = same_entity_edit_outputs
                same_entity_batch_labels = batch["same_type_entity_question"]["labels"]
            # diff entity
            diff_entity_edit_outputs = edited_model(batch["diff_type_entity_question"])
            if not isinstance(diff_entity_edit_outputs, torch.Tensor):
                diff_entity_edit_logits = diff_entity_edit_outputs.logits
                diff_entity_batch_labels = diff_entity_edit_outputs.labels
            else:
                diff_entity_edit_logits = diff_entity_edit_outputs
                diff_entity_batch_labels = batch["diff_type_entity_question"]["labels"]
            # same entity with oringinal answer
            same_entity_edit_original_answer_outputs = edited_model(batch["same_type_entity_question_original_answer"])
            if not isinstance(same_entity_edit_original_answer_outputs, torch.Tensor):
                same_entity_edit_original_answer_logits = same_entity_edit_original_answer_outputs.logits
                same_entity_original_answer_batch_labels = same_entity_edit_original_answer_outputs.labels
            else:
                same_entity_edit_original_answer_logits = same_entity_edit_original_answer_outputs
                same_entity_original_answer_batch_labels = batch["same_type_entity_question_original_answer"]["labels"]
            # diff entity with oringinal answer
            diff_entity_edit_original_answer_outputs = edited_model(batch["diff_type_entity_question_original_answer"])
            if not isinstance(diff_entity_edit_original_answer_outputs, torch.Tensor):
                diff_entity_edit_original_answer_logits = diff_entity_edit_original_answer_outputs.logits
                diff_entity_original_answer_batch_labels = diff_entity_edit_original_answer_outputs.labels
            else:
                diff_entity_edit_original_answer_logits = diff_entity_edit_original_answer_outputs
                diff_entity_original_answer_batch_labels = batch["diff_type_entity_question_original_answer"]["labels"]


            l_image_edit = self.model.edit_loss_fn(self.config, image_edit_logits, image_batch_labels, multimodal=True)["nll"]
            l_rephrase_edit = self.model.edit_loss_fn(self.config, rephrase_edit_logits, rephrase_batch_labels, multimodal=True)["nll"]
            l_image_rephrase_edit = self.model.edit_loss_fn(self.config, image_rephrase_edit_logits, image_rephrase_batch_labels, multimodal=True)["nll"]
            l_one_hop_edit = self.model.edit_loss_fn(self.config, one_hop_edit_logits, one_hop_batch_labels, multimodal=True)["nll"]
            l_image_one_hop_edit = self.model.edit_loss_fn(self.config, image_one_hop_edit_logits, image_one_hop_batch_labels, multimodal=True)["nll"]
            l_same_entity_edit = self.model.edit_loss_fn(self.config, same_entity_edit_logits, same_entity_batch_labels, multimodal=True)["nll"]
            l_diff_entity_edit = self.model.edit_loss_fn(self.config, diff_entity_edit_logits, diff_entity_batch_labels, multimodal=True)["nll"]
            
            # Collect some useful metrics
            with torch.no_grad():
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, image_rephrase_edit_logits, image_rephrase_batch_labels, multimodal=True)
                image_edit_dict = self.model.edit_loss_fn(self.config, image_edit_logits, image_batch_labels, multimodal=True)
                rephrase_edit_dict = self.model.edit_loss_fn(self.config, rephrase_edit_logits, rephrase_batch_labels, multimodal=True)
                one_hop_edit_dict = self.model.edit_loss_fn(self.config, one_hop_edit_logits, one_hop_batch_labels, multimodal=True)
                image_one_hop_edit_dict = self.model.edit_loss_fn(self.config, image_one_hop_edit_logits, image_one_hop_batch_labels, multimodal=True)
                same_entity_edit_dict = self.model.edit_loss_fn(self.config, same_entity_edit_logits, same_entity_batch_labels, multimodal=True)
                diff_entity_edit_dict = self.model.edit_loss_fn(self.config, diff_entity_edit_logits, diff_entity_batch_labels, multimodal=True)
                same_entity_edit_original_answer_dict = self.model.edit_loss_fn(self.config, same_entity_edit_original_answer_logits, same_entity_original_answer_batch_labels, multimodal=True)
                diff_entity_edit_original_answer_dict = self.model.edit_loss_fn(self.config, diff_entity_edit_original_answer_logits, diff_entity_original_answer_batch_labels, multimodal=True)

            locality_edit_outputs = edited_model(batch["locality_question"])
            if not isinstance(locality_edit_outputs, torch.Tensor):
                locality_edit_logits = locality_edit_outputs.logits
                kl_mask = locality_edit_outputs.attention_mask
            else:
                locality_edit_logits = locality_edit_outputs
                kl_mask = torch.ones(locality_edit_logits.shape[0], locality_edit_logits.shape[1]).to(locality_edit_logits.device)
            
            image_locality_edit_outputs = edited_model(batch["image_locality_question"])
            if not isinstance(image_locality_edit_outputs, torch.Tensor):
                image_locality_edit_logits = image_locality_edit_outputs.logits
                kl_image_mask = image_locality_edit_outputs.attention_mask
            else:
                image_locality_edit_logits = image_locality_edit_outputs
                kl_image_mask = torch.ones(image_locality_edit_logits.shape[0], image_locality_edit_logits.shape[1]).to(image_locality_edit_logits.device)
            
            l_loc = kl_loc_loss(locality_base_logits.detach(), locality_edit_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(image_locality_base_logits.detach(), image_locality_edit_logits, mask=kl_image_mask)

        if l_rephrase_edit.isnan():
            print("l_rephrase_edit is nan")
            print("input: ", batch["rephrase_question"]['text_input'])
        elif l_image_rephrase_edit.isnan():
            print("l_image_rephrase_edit is nan")
            print("input: ", batch["image_rephrase_question"]['text_input'])
        elif l_one_hop_edit.isnan():
            print("l_one_hop_edit is nan")
            print("input: ", batch["one_hop_question"]['text_input'])
        elif l_image_one_hop_edit.isnan():
            print("l_image_one_hop_edit is nan")
            print("input: ", batch["image_one_hop_question"]['text_input'])
        elif l_same_entity_edit.isnan():
            print("l_same_entity_edit is nan")
            print("input: ", batch["same_type_entity_question"]['text_input'])
        elif l_diff_entity_edit.isnan():
            print("l_diff_entity_edit is nan")
            print("input: ", batch["diff_type_entity_question"]['text_input'])

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * (l_rephrase_edit + l_image_rephrase_edit) + self.config.cloc * l_loc + self.config.iedit * (l_one_hop_edit + l_image_one_hop_edit + l_same_entity_edit + l_diff_entity_edit)
        else:
            l_total_edit = self.config.cedit * (l_rephrase_edit + l_image_rephrase_edit) + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * (l_one_hop_edit + l_image_one_hop_edit + l_same_entity_edit + l_diff_entity_edit)
        
        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        locality_edit_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(locality_edit_logits, dim=-1), k=1, dim=-1).indices
        locality_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(locality_base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        image_locality_edit_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(image_locality_edit_logits, dim=-1), k=10, dim=-1).indices
        image_locality_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(image_locality_base_logits, dim=-1), k=10, dim=-1).indices


        info_dict = {}
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/rephrase_edit'] = l_rephrase_edit.item()
        info_dict['loss/image_rephrase_edit'] = l_image_rephrase_edit.item()
        info_dict['loss/one_hop_edit'] = l_one_hop_edit.item()
        info_dict['loss/image_one_hop_edit'] = l_image_one_hop_edit.item()
        info_dict['loss/same_entity_edit'] = l_same_entity_edit.item()
        info_dict['loss/diff_entity_edit'] = l_diff_entity_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['loss/image_loc'] = l_image_loc.item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image/acc'] = image_edit_dict["acc"].item()
        info_dict['rephrase/acc'] = rephrase_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict['one_hop_edit/acc'] = one_hop_edit_dict["acc"].item()
        info_dict['image_one_hop_edit/acc'] = image_one_hop_edit_dict["acc"].item()
        info_dict['same_entity_edit/acc'] = same_entity_edit_dict["acc"].item()
        info_dict['diff_entity_edit/acc'] = diff_entity_edit_dict["acc"].item()
        info_dict['same_entity_edit_original_answer/acc'] = same_entity_edit_original_answer_dict["acc"].item()
        info_dict['diff_entity_edit_original_answer/acc'] = diff_entity_edit_original_answer_dict["acc"].item()
        info_dict["loc/acc"] = sum(locality_edit_logits_softmax_top_k.view(-1) == locality_base_logits_softmax_top_k.view(-1))/locality_edit_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(image_locality_edit_logits_softmax_top_k.view(-1) == image_locality_base_logits_softmax_top_k.view(-1))/image_locality_edit_logits_softmax_top_k.view(-1).shape[0]
        info_dict["time/edit"] = edit_time
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_rephrase_edit, l_loc, l_base, info_dict
        
    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            # origi_outputs = self.model(batch["edit_inner"])
            # if not isinstance(origi_outputs, torch.Tensor):
            #     origi_logits = origi_outputs.logits
            #     origi_batch_labels = origi_outputs.labels
            # else:
            #     origi_logits = origi_outputs
            #     origi_batch_labels = batch["edit_inner"]["labels"]
            
            base_outputs = self.model(batch["loc"])
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:  
                base_logits = base_outputs
            
            base_image_outputs = self.model(batch["loc_image"])
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs

            if self.config.alg == "SERAC_MULTI" and self.config.model_name == "minigpt4" and base_logits.shape[2] == 32001:
                base_logits = base_logits[:, :, :-1]
                base_image_logits = base_image_logits[:, :, :-1]
        
        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner_without_image"], batch["cond"])
        # edited_model = self.model
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_outputs = edited_model(batch["edit_outer"])
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
                post_batch_labels = post_edit_outputs.labels
            else:
                post_edit_logits = post_edit_outputs
                post_batch_labels = batch["edit_outer"]["labels"]

            # rephrase image
            post_image_edit_outputs = edited_model(batch["edit_outer_image"])
            
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
                post_image_batch_labels = post_image_edit_outputs.labels
            else:
                post_image_edit_logits = post_image_edit_outputs
                post_image_batch_labels = batch["edit_outer_image"]["labels"]

            inner_edit_outputs = edited_model(batch["edit_inner"])
            
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
                inner_batch_labels = inner_edit_outputs.labels
            else:
                inner_edit_logits = inner_edit_outputs
                inner_batch_labels = batch["edit_inner"]["labels"]

            l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)["nll"]
            l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)["nll"]          
            
            # Collect some useful metrics
            with torch.no_grad():
                post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels, multimodal=True)
                inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels, multimodal=True)
                image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels, multimodal=True)
            
            post_base_outputs = edited_model(batch["loc"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_image"])
            if not isinstance(post_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        if l_edit.isnan():
            print("l_edit is nan")
            print("input: ", batch["edit_outer"]['text_input'])
        elif l_image_edit.isnan():
            print("l_image_edit is nan")
            print("input: ", batch["edit_outer_image"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["loc"]['text_input'])
        elif l_image_loc.isnan():
            print("l_image_loc is nan")
            print("input: ", batch["loc_image"]['text_input'])

        if self.config.alg == "SERAC_MULTI":
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc + self.config.iedit * l_image_edit
        else:
            l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc + l_image_loc) + self.config.iedit * l_image_edit
        

        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # Text locality
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        # Image locality
        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

        # print("编辑前的模型输出:", self.train_set.tok.decode(self.model.edit_loss_fn(self.config, origi_logits, origi_batch_labels, multimodal=True)["pred_ids"][0],skip_special_tokens=True).replace('<s>', ''))
        # print("编辑后的模型输出:", self.train_set.tok.decode(inner_edit_dict["pred_ids"][0],skip_special_tokens=True).replace('<s>', ''))
        # print("换表述后的模型输出:", self.train_set.tok.decode(post_edit_dict["pred_ids"][0],skip_special_tokens=True).replace('<s>', ''))
        # print("换图片的模型输出:", self.train_set.tok.decode(image_rephrase_edit_dict["pred_ids"][0],skip_special_tokens=True).replace('<s>', ''))

        info_dict = {}
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/image_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.train_edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        inner_acc = f"{stats['inner/acc_val']:<12.5f}"
        image_acc = f"{stats['image/acc_val']:<12.5f}"
        rephrase_acc = f"{stats['rephrase/acc_val']:<12.5f}"
        image_rephrase_acc = f"{stats['image_rephrase/acc_val']:<12.5f}"
        one_hop_acc = f"{stats['one_hop_edit/acc_val']:<12.5f}"
        image_one_hop_acc = f"{stats['image_one_hop_edit/acc_val']:<12.5f}"
        same_entity_acc = f"{stats['same_entity_edit/acc_val']:<12.5f}"
        diff_entity_acc = f"{stats['diff_entity_edit/acc_val']:<12.5f}"
        same_entity_original_answer_acc = f"{stats['same_entity_edit_original_answer/acc_val']:<12.5f}"
        diff_entity_original_answer_acc = f"{stats['diff_entity_edit_original_answer/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"
        loc_image_acc = f"{stats['image_loc/acc_val']:<12.5f}"

        LOG.info(
            f"Validation [{prog}] | "
            f"inner_acc: {inner_acc} | "
            f"image_acc: {image_acc} | "
            f"rephrase_acc: {rephrase_acc} | "
            f"image_rephrase_acc: {image_rephrase_acc} | "
            f"one_hop_acc: {one_hop_acc} | "
            f"image_one_hop_acc: {image_one_hop_acc} | "
            f"same_entity_acc: {same_entity_acc} | "
            f"diff_entity_acc: {diff_entity_acc} | "
            f"same_entity_original_answer_acc: {same_entity_original_answer_acc} | "
            f"diff_entity_original_answer_acc: {diff_entity_original_answer_acc} | "
            f"loc_acc: {loc_acc} | "
            f"loc_image_acc: {loc_image_acc} | "
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.test_edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats