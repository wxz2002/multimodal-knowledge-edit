from easyeditor.trainer.blip2_models import Blip2OPT
from easyeditor.dataset.processor.blip_processors import BlipImageEvalProcessor
from transformers import GPT2Tokenizer
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# processor = LlavaProcessor.from_pretrained("../hugging_cache/llava-1.5-7b-hf")
# # print(processor)
# model = LlavaForConditionalGeneration.from_pretrained("../hugging_cache/llava-1.5-7b-hf").to("cuda")
# # print(model)
# image = Image.open("../test_image/trump1.png").convert("RGB")
# inputs = processor("The current president of the United States is", image, return_tensors="pt").to("cuda")
# inputs["pixel_values"] = None
# out = model.generate(**inputs, max_new_tokens=50)
# print(processor.decode(out[0], skip_special_tokens=True))

model = Blip2OPT(vit_model="eva_clip_g",
                img_size=364,
                use_grad_checkpoint=True,
                vit_precision="fp32",
                freeze_vit=True,
                opt_model="/root/autodl-tmp/cause_tracing/easyedit/hugging_cache/opt-2.7b",
                state_dict_file="/root/autodl-tmp/cause_tracing/easyedit/hugging_cache/eva_vit_g.pth",
                qformer_name_or_path="/root/autodl-tmp/cause_tracing/easyedit/hugging_cache/bert-base-uncased",
                qformer_checkpoint="/root/autodl-tmp/cause_tracing/easyedit/hugging_cache/blip2_pretrained_opt2.7b.pth").to("cuda")
print(model)
# processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
# input = {
#     "prompt": "Question: Is the person in the picture the current president of the United States? Answer:",
#     "image": processor(Image.open("../test_image/trump1.png").convert('RGB')).unsqueeze(0).to("cuda")
# }
# out = model.generate(input)
# print(out)


# raw_image = Image.open("../test_image/trump1.png").convert('RGB')
# inputs = {}
# inputs["text_input"] = ["Is the person in the picture the current president of the United States? Answer with yes or no."]
# inputs["image"] = processor(raw_image)

# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
# from transformers import pipeline

# question = ["Who discovered the law of universal gravitation? Answer:", "Who is the current president of the United States? Answer:",
# "Is the person in the picture the current president of the United States? Answer with yes or no. Answer:"]


# generator = pipeline('text-generation', model="../hugging_cache/opt-2.7b")
# print(generator.model)
# print(generator("Joe Biden is the vice", max_length=50))
# processor = Blip2Processor.from_pretrained("../hugging_cache/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("../hugging_cache/blip2-opt-2.7b").to("cuda")

# raw_image = Image.open("../test_image/Biden1.jpg").convert('RGB')
# question = "Joe Biden holds the position of"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

# out = model.generate(**inputs,max_new_tokens=50)
# print(processor.decode(out[0], skip_special_tokens=True).strip())