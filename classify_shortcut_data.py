import json


with open('./rome_results/can_rome_edit_datas.json') as f:
    with_shortcut_datas = json.load(f)

with open('./rome_results/no_rome_edit_datas.json') as f:
    no_shortcut_datas = json.load(f)

with open('../our_dataset/final_image_rephrase_test_multimodal_hops.json') as f:
    all_datas = json.load(f)

tested_datas = []

with open('./neurons/can_rome_edit_results.jsonl') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        tested_datas.append(data)
with open('./neurons/no_rome_edit_results.jsonl') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        tested_datas.append(data)

for tested_data in tested_datas:
    for data in all_datas:
        if data['subject'] == tested_data['subject'] and data['knowledge_edit'] == tested_data['knowledge_edit'] and data['multimodal_hops'] == tested_data['multimodal_hops']:
            after_edit_b_to_c_neurons = tested_data['after_edit_b_to_c_neurons']
            after_edit_a_to_b_neurons = tested_data['after_edit_a_to_b_neurons']
            after_edit_a_to_c_neurons = tested_data['after_edit_a_to_c_neurons']
            after_edit_single_hop_neurons = after_edit_a_to_b_neurons + after_edit_b_to_c_neurons
            share_neurons = []
            for neuron in after_edit_single_hop_neurons:
                if neuron in after_edit_a_to_c_neurons:
                    share_neurons.append(neuron)
            acc = len(share_neurons) / len(after_edit_a_to_c_neurons)
            if acc > 0.2:
                no_shortcut_datas.append(data)
            elif acc < 0.1:
                with_shortcut_datas.append(data)

with open('../our_dataset/with_shortcut_datas.json', 'w') as f:
    json.dump(with_shortcut_datas, f, indent=4)

with open('../our_dataset/no_shortcut_datas.json', 'w') as f:
    json.dump(no_shortcut_datas, f, indent=4)
