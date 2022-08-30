import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda()
else:
    from torch import from_numpy

def tokenize(tokenizer, instances, max_seq_len):
    tokens = []
    for ins in instances['sentence']:
        sent = tokenizer(ins)['input_ids'][:max_seq_len]
        tokens.append(sent)
    seq_max_len = max(len(seq) for seq in tokens)
    for seq in tokens:
        for i in range(seq_max_len - len(seq)):
            seq.append(0)

    return tokens

class BertSmallFinetunedModel(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
        self.max_seq_len=512
        self.model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-small", num_labels=2,
                                                                   output_attentions=True, output_hidden_states=True)
        self.to(self.device)


    def forward(self, batch):
        tokens = self.tokenize(batch)
        labels = batch['label'].to(self.device)
        out =self.model(input_ids=tokens, labels=labels, output_hidden_states=True, output_attentions=False)
        loss = out.loss
        logits = out.logits
        hidden_states = out.hidden_states
        return loss,logits,hidden_states



    def tokenize(self, instances):
        tokens = tokenize(self.tokenizer, instances, self.max_seq_len)
        tokens = from_numpy(np.array(tokens)).long().to(self.device)
        return tokens














































