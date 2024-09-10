import json
from tqdm import tqdm

# can_rome_edit_datas = []
# no_rome_edit_datas = []

# with open('../our_dataset/final_image_rephrase_test_multimodal_hops.json', 'r') as f:
#     original_datas = json.load(f)

# with open('./rome_results/llava_no_exact_match.json', 'r') as f:
#     rome_datas = json.load(f)

# for original_data in tqdm(original_datas):
#     subject = original_data['subject']
#     image_question = original_data['knowledge_edit']['image_question']
#     for rome_data in rome_datas:
#         if rome_data['subject'] == subject and rome_data['original_question'] == image_question:
#             if rome_data['inner_acc'] >= 0.7:
#                 can_rome_edit_datas.append(original_data)
#             elif rome_data['inner_acc'] <= 0.1:
#                 no_rome_edit_datas.append(original_data)
#             break

# print(len(can_rome_edit_datas))
# print(len(no_rome_edit_datas))

# with open('./rome_results/can_rome_edit_datas.json', 'w') as f:
#     json.dump(can_rome_edit_datas, f, indent=4)

# with open('./rome_results/no_rome_edit_datas.json', 'w') as f:
#     json.dump(no_rome_edit_datas, f, indent=4)

stability_original_datas = []
stability_no_datas = []
with open('../our_dataset/final_image_rephrase_test.json', 'r') as f:
    test_datas = json.load(f)

with open('./rome_results/can_rome_edit_datas.json', 'r') as f:
    original_datas = json.load(f)
with open('./rome_results/no_rome_edit_datas.json', 'r') as f:
    original_datas += json.load(f)

new_original_datas = []
for original_data in original_datas:
    for test_data in test_datas:
        if original_data['subject'] == test_data['subject'] and original_data['knowledge_edit']== test_data['knowledge_edit']:
            new_original_datas.append(test_data)
            break
    
with open('./rome_results/llava_no_exact_match.json', 'r') as f:
    rome_datas = json.load(f)

for original_data in tqdm(new_original_datas):
    subject = original_data['subject']
    image_question = original_data['knowledge_edit']['image_question']
    for rome_data in rome_datas:
        if rome_data['subject'] == subject and rome_data['original_question'] == image_question:
            if rome_data['same_entity_original_answer_acc'] >= 0.7:
                stability_original_datas.append(original_data)
            elif rome_data['same_entity_original_answer_acc'] <= 0.1:
                stability_no_datas.append(original_data)
            break

print(len(stability_original_datas))
print(len(stability_no_datas))

# with open('./rome_results/stability_original_answer_datas.json', 'w') as f:
#     json.dump(stability_original_datas, f, indent=4)

# with open('./rome_results/stability_no_answer_datas.json', 'w') as f:
#     json.dump(stability_no_datas, f, indent=4)