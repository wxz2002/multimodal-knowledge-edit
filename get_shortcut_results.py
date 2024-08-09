import json

if __name__ == '__main__':
    with open('./neurons/can_rome_edit_results.jsonl') as f:
        lines = f.readlines()
        can_rome_edit_results = [json.loads(line) for line in lines]
    with open('./neurons/no_rome_edit_results.jsonl') as f:
        lines = f.readlines()
        no_rome_edit_results = [json.loads(line) for line in lines]
    
    accs = []
    for result in can_rome_edit_results:
        after_edit_b_to_c_neurons = result['after_edit_b_to_c_neurons']
        after_edit_a_to_b_neurons = result['after_edit_a_to_b_neurons']
        after_edit_a_to_c_neurons = result['after_edit_a_to_c_neurons']
        after_edit_single_hop_neurons = after_edit_a_to_b_neurons + after_edit_b_to_c_neurons
        share_neurons = []
        for neuron in after_edit_single_hop_neurons:
            if neuron in after_edit_a_to_c_neurons:
                share_neurons.append(neuron)
        acc = len(share_neurons) / len(after_edit_a_to_c_neurons)
        accs.append(acc)
    print('can_rome_edit acc:', sum(accs) / len(accs))

    accs = []
    for result in no_rome_edit_results:
        acc = len(result['before_shared_neurons']) / len(result['before_multi_hop_neurons'])
        accs.append(acc)
    print('no_rome_edit acc:', sum(accs) / len(accs))

    accs = []
    for result in no_rome_edit_results:
        acc = len(result['a_to_c_shared_neurons']) / len(result['before_edit_a_to_c_neurons'])
        accs.append(acc)
    print('no_rome_edit a_to_c acc:', sum(accs) / len(accs))