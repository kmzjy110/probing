from datasets import load_dataset
import torch
from model import BertSmallFinetunedModel
import tqdm
import json



def finetune_model(num_epochs, batch_size, learning_rate, model_file):
    dataset = load_dataset('sst2', split='train')
    dataset_length = len(dataset)
    seen_size = int(dataset_length*0.5)
    unseen_size = dataset_length - seen_size

    seen_dataset, unseen_dataset = torch.utils.data.random_split(dataset, [seen_size, unseen_size])

    write_seen_and_unseen_indices(seen_dataset,unseen_dataset)
    device = torch.device('cuda')
    model = BertSmallFinetunedModel(device)


    data_loader = torch.utils.data.DataLoader(
        seen_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, betas=(0.9, 0.999), eps=1e-9)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(data_loader),
        pct_start=0.1,  # Warm up for 10% of the total training time
    )
    for epoch in tqdm.trange(num_epochs, desc="training", unit="epoch"):
        with tqdm.tqdm(
                data_loader,
                desc="epoch {}".format(epoch + 1),
                unit="batch",
                total=len(data_loader)) as batch_iterator:
            model.train()
            total_loss = 0.0
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()

                loss, logits, hidden_states = model.forward(batch)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                scheduler.step()
                batch_iterator.set_postfix(mean_loss=total_loss / i)
                print(hidden_states)
                exit()

            torch.save(model.state_dict(), model_file + "_epoch_"+str(epoch+1)+".pt")


def write_seen_and_unseen_indices(seen_dataset, unseen_dataset):
    data_dict = {"seen": [], "unseen":[]}
    for item in seen_dataset:
        data_dict['seen'].append(item['idx'])
    for item in unseen_dataset:
        data_dict['unseen'].append(item['idx'])
    with open('seen.json','w') as file:
        json.dump(data_dict,file)




