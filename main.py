# from finetune import finetune_model
#
# finetune_model(7,32,1e-5,"bert_small_finetune")
from transformers import BertModel, BertTokenizer
import torch
from write_probing_data import write_probing_data, retrieve_first_token_rep, retrieve_mean
from model import BertSmallFinetunedModel
from probing_model import model_pred


def sanity_check():
    model = BertModel.from_pretrained('bert-base-uncased').to(torch.device('cuda'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    write_probing_data(model, False, 'bert_untuned', retrieve_mean, tokenizer)
    model_pred('rep_dataset_model_bert_untuned.json')


def run_probe(finetuned_epoch):
    model = BertSmallFinetunedModel(torch.device('cuda'))
    model.load_state_dict(torch.load('bert_small_finetune_epoch_' + str(finetuned_epoch) + '.pt'))

    write_probing_data(model, True, 'bert_small_finetune_epoch_' + str(finetuned_epoch), retrieve_mean)
    model_pred('rep_dataset_model_bert_small_finetune_epoch_' + str(finetuned_epoch) + '.json')

for i in range(1,8):
    run_probe(i)

sanity_check()