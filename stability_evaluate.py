import json

def get_rome_accs(results):
    accs = []
    for result in results:
        after_edit_a_to_c_neurons = result['after_edit_a_to_c_neurons']
        stability_a_to_c_neurons = result['stability_a_to_c_neurons']
        share_neurons = []
        for neuron in after_edit_a_to_c_neurons:
            if neuron in stability_a_to_c_neurons:
                share_neurons.append(neuron)
        acc = len(share_neurons) / len(after_edit_a_to_c_neurons)
        accs.append(acc)
    return accs

def get_ft_accs(results):
    accs = []
    for result in results:
        after_edit_a_to_c_neurons = result['a_to_c']
        stability_a_to_c_neurons = result['stability_a_to_c']
        share_neurons = []
        for neuron in after_edit_a_to_c_neurons:
            if neuron in stability_a_to_c_neurons:
                share_neurons.append(neuron)
        acc = len(share_neurons) / len(after_edit_a_to_c_neurons)
        accs.append(acc)
    return accs

if __name__ == '__main__':
    with open('./stability_results/stability_original_answer_results.jsonl') as f:
        lines = f.readlines()
        rome_stability_original_answer_results = [json.loads(line) for line in lines]
    with open('./stability_results/stability_no_answer_results.jsonl') as f:
        lines = f.readlines()
        rome_stability_no_answer_results = [json.loads(line) for line in lines]
    with open('./stability_results/ft_unimodal_results.jsonl') as f:
        lines = f.readlines()
        ft_unimodal_results = [json.loads(line) for line in lines]
    with open('./stability_results/ft_multimodal_results.jsonl') as f:
        lines = f.readlines()
        ft_multimodal_results = [json.loads(line) for line in lines]
    
    rome_stability_original_answer_accs = get_rome_accs(rome_stability_original_answer_results)
    average_rome_stability_original_answer_acc = sum(rome_stability_original_answer_accs) / len(rome_stability_original_answer_accs)
    
    rome_stability_no_answer_accs = get_rome_accs(rome_stability_no_answer_results)
    average_rome_stability_no_answer_acc = sum(rome_stability_no_answer_accs) / len(rome_stability_no_answer_accs)
    
    ft_unimodal_accs = get_ft_accs(ft_unimodal_results)
    average_ft_unimodal_acc = sum(ft_unimodal_accs) / len(ft_unimodal_accs)

    ft_multimodal_accs = get_ft_accs(ft_multimodal_results)
    average_ft_multimodal_acc = sum(ft_multimodal_accs) / len(ft_multimodal_accs)
    
    print('rome_stability_original_answer_acc:', average_rome_stability_original_answer_acc)
    print('rome_stability_no_answer_acc:', average_rome_stability_no_answer_acc)
    print('ft_stability_unimodal_acc:', average_ft_unimodal_acc)
    print('ft_stability_multimodal_acc:', average_ft_multimodal_acc)

    if average_rome_stability_original_answer_acc > average_rome_stability_no_answer_acc:
        print('结论3: True')
    else:
        print('结论3: False')
    
    if average_ft_unimodal_acc < average_ft_multimodal_acc:
        print('结论4: True')
    else :
        print('结论4: False')
