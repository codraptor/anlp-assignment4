import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from table_bert import TableBertModel, Table, Column
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

class SalaryDataset(Dataset):
    def __init__(self, data, labels, model):
        self.data = data
        self.labels = labels
        self.model = model

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        description = ' '.join(row['FullDescription'].split(' ')[:400])
        label = self.labels.iloc[idx]['SalaryNormalized']

        context = self.model.tokenizer.tokenize(description)[:400]

        table = Table(
                    id='Job Salary Data',
                    header=[
                        Column('Title', 'text', sample_value='Senior Manager'),
                        # Column('Location Raw', 'text', sample_value='Bedforshire, UK'),
                        Column('Location Normalized', 'text', sample_value='Bedfordshire'),
                        Column('Contract Time', 'text', sample_value='permanent'),
                        Column('Company', 'text', sample_value='Indigo21'),
                        Column('Category', 'text', sample_value='Charity & Voluntary Jobs'),
                        Column('Source Name', 'text', sample_value='guardian.co.uk'),
                    ],
                    data=[row]).tokenize(tabert_model.tokenizer)

        return context, table, label

def collate_fn(list_items):
    contexts, tables, labels = [],[],[]
    for context,table, label in list_items:
        contexts.append(context)
        tables.append(table)
        labels.append(label)
    return contexts, tables, torch.tensor(labels, dtype=torch.float32)

class BertRegression(nn.Module):

    def __init__(self):
        super(BertRegression, self).__init__()
        self.tabert_model = TableBertModel.from_pretrained('../TaBERT/tabert_base_k1/model.bin')
        
        for param in self.tabert_model.parameters():
            param.requires_grad = False
       
        for param in self.tabert_model._bert_model.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        for param in self.tabert_model._bert_model.bert.pooler.parameters():
            param.requires_grad = True

        for param in self.tabert_model._bert_model.cls.parameters():
            param.requires_grad = True

        self.linear1 = nn.Linear(5376,768)
        self.linear_out = nn.Linear(768,1)        
        self.relu = nn.ReLU()

    def forward(self, contexts, tables):
        context_encoding, column_encoding, info_dict = self.tabert_model.encode(contexts=contexts, tables=tables)
        # print(f"inside bertregression: {context_encoding.shape}")
        cls_encoding = context_encoding[:,0:1,:]
        concat_embedding = torch.cat((cls_encoding, column_encoding),dim=1).view((-1,5376))
        out = self.linear_out(self.relu(self.linear1(concat_embedding)))
        return out


def train_model(model, tabert_model, train_loader, valid_loader, loss_fn, lr=5e-5,n_epochs=10,):
#     param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
#     optimizer = torch.optim.Adam(param_lrs, lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = n_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train()
        avg_loss = 0.
        
        for data in tqdm(train_loader, disable=False):

            contexts = [list(context) for context in data[0]]
            tables = data[1]

            y_batch = data[2].view((-1,1)).cuda()
            # print(f"before model")
            y_pred = model(contexts, tables)
            print(f"after model")

            loss = loss_fn(y_pred, y_batch)
            print(f"after loss")
            optimizer.zero_grad()
            print(f"after optim zero")

            loss.backward()
            print(f"after loss backwar")

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            print(f"avg loss: {avg_loss}")
        model.eval()
        valid_loss = 0.
        for i, data in enumerate(valid_loader):

            contexts = [list(context) for context in data[0]]
            tables = data[1]

            y_batch = data[2].view((-1,1)).cuda()
            
            y_pred = model(contexts, tables)

            loss = loss_fn(y_pred, y_batch)
            valid_loss += loss.item() / len(valid_loader)

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s \t validation loss={:.4f}'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time, valid_loss))
        
        if epoch %2 == 0:
            torch.save(model.state_dict(), f"model_weights_{epoch}.pt")


if __name__=="__main__":
    tabert_model = TableBertModel.from_pretrained(
    '../TaBERT/tabert_base_k1/model.bin').cuda()
    
    train_data, train_labels = pd.read_csv("processed_data/train_data.csv"), pd.read_csv("processed_data/train_labels.csv")
    val_data, val_labels = pd.read_csv("processed_data/valid_data.csv"), pd.read_csv("processed_data/valid_labels.csv")
    test_data, test_labels = pd.read_csv("processed_data/test_data.csv"), pd.read_csv("processed_data/test_labels.csv")

    train_dataset = SalaryDataset(train_data, train_labels, model=tabert_model)
    val_dataset = SalaryDataset(val_data, val_labels, model=tabert_model)
    test_dataset = SalaryDataset(test_data, test_labels, model=tabert_model)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = BertRegression()
    model.cuda()
    train_model(model, tabert_model, train_dataloader, val_dataloader, nn.L1Loss(), lr=5e-5,n_epochs=10)

    print('Training Done')