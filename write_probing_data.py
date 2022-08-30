
import json
from datasets import load_dataset
import torch
import tqdm
import numpy as np
from model import tokenize, from_numpy
def write_probing_data(model, finetuned, model_name, rep_retrieve, tokenizer = None):
    probing_data = {}
    with open('seen.json', 'r') as f:

        probing_data = json.loads(f.read())

    seen_set = set(probing_data['seen'])
    dataset = load_dataset('sst2', split='train')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True)
    rep_data_items = []
    with tqdm.tqdm(
            data_loader,
            desc="writing",
            unit="batch",
            total=len(data_loader)) as batch_iterator:
        for i, batch in enumerate(batch_iterator, start=1):

            model.eval()
            if finetuned:
                loss, logits, hidden_states = model.forward(batch)
                last_hidden_state = hidden_states[-1]
            else:
                tokens = tokenize(tokenizer, batch, 512)
                tokens = from_numpy(np.array(tokens)).long().to(model.device)
                last_hidden_state = model.forward(tokens).last_hidden_state
            for batch_idx in range(len(batch)):
                cur_rep  = rep_retrieve(last_hidden_state[batch_idx, :, :].squeeze())
                cur_rep.squeeze()
                seen = 0
                if batch['idx'][batch_idx].tolist() in seen_set:
                    seen=1
                rep_data_items.append({"idx": batch['idx'][batch_idx].tolist(),
                                       "rep": cur_rep.tolist(),
                                       "seen": seen})
    with open('rep_dataset_model_' + model_name+'.json','w') as file:
        json.dump(rep_data_items,file)


def retrieve_first_token_rep(last_hidden_state):
    return last_hidden_state[0,:].squeeze()

def retrieve_mean(last_hidden_state):
    return torch.mean(last_hidden_state, dim=0).squeeze()


