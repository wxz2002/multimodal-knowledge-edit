import torch
import types
from statistics import mean

from easyeditor import BaseEditor, MultimodalTrainer, MultimodalEditor
from easyeditor import CaptionDataset, VQADataset
from easyeditor import MENDMultimodalTrainingHparams, SERACMultimodalTrainingHparams, IKEMultimodalHyperParams, MENDMultimodalHparams \
    , SERACMultimodalHparams, FTMultimodalHparams, ROMEMultimodalHyperParams
from easyeditor import encode_ike_facts_multimodal
from sentence_transformers import SentenceTransformer


def print_result(metrics):
    inner_acc = mean([m['post']['inner_acc'].item() for m in metrics])
    image_acc = mean([m['post']['image_acc'].item() for m in metrics])
    rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])
    image_rephrase_acc = mean([m['post']['image_rephrase_acc'].item() for m in metrics])
    one_hop_acc = mean([m['post']['one_hop_acc'].item() for m in metrics])
    image_one_hop_acc = mean([m['post']['image_one_hop_acc'].item() for m in metrics])
    same_entity_acc = mean([m['post']['same_entity_acc'].item() for m in metrics])
    same_entity_original_answer_acc = mean([m['post']['same_entity_original_answer_acc'].item() for m in metrics])
    diff_entity_acc = mean([m['post']['diff_entity_acc'].item() for m in metrics])
    diff_entity_original_answer_acc = mean([m['post']['diff_entity_original_answer_acc'].item() for m in metrics])
    locality_acc = mean([m['post']['locality_acc'].item() for m in metrics])
    image_locality_acc = mean([m['post']['image_locality_acc'].item() for m in metrics])
    print(f'inner_acc: {inner_acc}')
    print(f'image_acc: {image_acc}')
    print(f'rephrase_acc: {rephrase_acc}')
    print(f'image_rephrase_acc: {image_rephrase_acc}')
    print(f'one_hop_acc: {one_hop_acc}')
    print(f'image_one_hop_acc: {image_one_hop_acc}')
    print(f'same_entity_acc: {same_entity_acc}')
    print(f'same_entity_original_answer_acc: {same_entity_original_answer_acc}')
    print(f'diff_entity_acc: {diff_entity_acc}')
    print(f'diff_entity_original_answer_acc: {diff_entity_original_answer_acc}')
    print(f'locality_acc: {locality_acc}')
    print(f'image_locality_acc: {image_locality_acc}')  

def train_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/minigpt4.yaml')
    train_ds = VQADataset('../editing-data/vqa/final_zsre_train_text.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../editing-data/vqa/final_zsre_eval_rephrased.json',size=20, config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
  
        
def train_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/blip2.yaml')
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',size=20, config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()   
        

def train_MEND_LLAVA_VQA():
    hparams = MENDMultimodalTrainingHparams.from_hparams('hparams/TRAINING/MEND/llava.yaml')
    train_ds = VQADataset('../editing-data/vqa/final_zsre_train_text.json',config=hparams, only_text=True)
    eval_ds = VQADataset('../editing-data/vqa/final_zsre_eval_rephrased.json',config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()  

def test_MEND_LLAVA_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/llava.yaml')
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run() 
        
    
def test_MEND_MiniGPT4_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  

def test_MEND_Blip2OPT_VQA():
    hparams = MENDMultimodalHparams.from_hparams('hparams/MEND/blip2.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()  
  
def test_SERAC_MiniGPT4_VQA():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_SERAC_Blip2OPT_VQA():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/blip2.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams,only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()     
    
def train_SERAC_MiniGPT4_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/minigpt4.yaml')
    train_ds = VQADataset('../editing-data/vqa/final_zsre_train_text.json',config=hparams, only_text=True)
    eval_ds = VQADataset('../editing-data/vqa/final_zsre_eval_rephrased.json',size=20,config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    
    trainer.run()
    
def train_SERAC_Blip2OPT_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/blip2.yaml')
    train_ds = VQADataset('../editing-data/vqa/final_zsre_train_text.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../editing-data/vqa/final_zsre_eval_rephrased.json', size=20, config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def train_SERAC_LLAVA_VQA():
    hparams = SERACMultimodalTrainingHparams.from_hparams('hparams/TRAINING/SERAC/llava.yaml')
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', size=20,config=hparams, only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()

def test_SERAC_LLAVA_VQA():
    hparams = SERACMultimodalHparams.from_hparams('hparams/SERAC/llava.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',size=20,config=hparams,only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()    

def test_FT_Blip2OPT_VQA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/blip2.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams,only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()   
    
def test_FT_MiniGPT4_VQA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/minigpt4.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams,only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run()

def test_FT_LLAVA_VQA():
    hparams = FTMultimodalHparams.from_hparams('hparams/FT/llava.yaml')
    # train_ds = VQADataset('data/vqa_train.json', config=hparams)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json',config=hparams,only_text=False)
    trainer = MultimodalTrainer(
        config=hparams,
        train_set=eval_ds,
        val_set=eval_ds
    )
    
    trainer.run() 

def test_IKE_Blip2OPT_VQA():  
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    print_result(metrics)
    
def Generate_Embedding_for_IKE():
    ## Generate blip2 embedding files for IKE
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/blip2.yaml')
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    print("train data has been loaded")
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
    
    ## Generate minigpt4 embedding files for IKE
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    print("train data has been loaded")
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)

    ## Generate llava embedding files for IKE
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/llava.yaml')
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    print("train data has been loaded")
    encode_ike_facts_multimodal(sentence_model, train_ds, hparams)
     
    
def test_IKE_MiniGPT4_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)
    
def test_IKE_LLAVA_VQA():
    
    hparams = IKEMultimodalHyperParams.from_hparams('hparams/IKE/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    
    print_result(metrics)

def test_ROME_Blip2OPT_VQA():  
    hparams = ROMEMultimodalHyperParams.from_hparams('hparams/ROME/blip2.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    print_result(metrics)

def test_ROME_MiniGPT4_VQA():  
    hparams = ROMEMultimodalHyperParams.from_hparams('hparams/ROME/minigpt4.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    print_result(metrics)

def test_ROME_LLAVA_VQA():
    hparams = ROMEMultimodalHyperParams.from_hparams('hparams/ROME/llava.yaml')
    editor = MultimodalEditor.from_hparams(hparams)
    train_ds = VQADataset('../our_dataset/rephrase_train.json', config=hparams, only_text=True)
    eval_ds = VQADataset('../our_dataset/final_image_rephrase_test.json', config=hparams, only_text=False)
    metrics = editor.edit_dataset(
        ds=eval_ds,
        train_ds=train_ds,
        keep_original_weight=True        
    )
    print_result(metrics)
    
if __name__ == "__main__":
    
    # test_ROME_Blip2OPT_VQA()
    # test_ROME_MiniGPT4_VQA()
    # test_ROME_LLAVA_VQA()

    # test_FT_Blip2OPT_VQA()
    # test_FT_MiniGPT4_VQA()
    # test_FT_LLAVA_VQA()

    # Generate_Embedding_for_IKE()
    # test_IKE_Blip2OPT_VQA()
    # test_IKE_MiniGPT4_VQA()
    # test_IKE_LLAVA_VQA()

    # train_MEND_Blip2OPT_VQA()
    # test_MEND_Blip2OPT_VQA()
    # train_MEND_MiniGPT4_VQA()
    # test_MEND_MiniGPT4_VQA()
    # train_MEND_LLAVA_VQA()
    # test_MEND_LLAVA_VQA()

    # train_SERAC_Blip2OPT_VQA()
    # test_SERAC_Blip2OPT_VQA()
    # train_SERAC_MiniGPT4_VQA()
    # test_SERAC_MiniGPT4_VQA()
    train_SERAC_LLAVA_VQA()
    # test_SERAC_LLAVA_VQA()
