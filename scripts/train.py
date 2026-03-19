# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:48:57 2026

@author: HP
"""
import torch
import time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
import copy
from torch.utils.data import DataLoader
from datasets.dataset import MyDataset
from model.model import SimpleGeneModel

print("start")
train_dataset = MyDataset(
    promoter_file="data/processed/promoter_train.csv",
    scrna_file="data/processed/integrated_data.h5ad",
)
val_dataset = MyDataset(
    promoter_file="data/processed/promoter_val.csv",
    scrna_file="data/processed/integrated_data.h5ad",
)

train_loader = DataLoader(train_dataset, batch_size=128,num_workers=4,pin_memory=True)
val_loader = DataLoader(val_dataset,batch_size=128,num_workers=4,pin_memory=True)
#train_loader = val_loader

model = SimpleGeneModel()

def dryrun_cpu(model,train_loader,steps=50,learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    # test if dataset and dataloader work well 
    batch = next(iter(train_loader))
    promoters, exprs, ys = batch  # (batch, 400, 5), (batch, 16300), (batch,)
    print(promoters.shape, exprs.shape, ys.shape)
    ys = ys.float()
    #记录一个 LSTM 参数训练前的值
    params_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }
    
    losses = []
    for step in range(steps):
        batch = next(iter(train_loader))
        promoters, exprs, ys = batch
        ys = ys.float()
    
        optimizer.zero_grad()
        out = model(promoters, exprs).squeeze(1)
        loss = criterion(out, ys)
    
        loss.backward()
        optimizer.step()
    
        losses.append(loss.item())
    
    # parameters and gradients after training
    for name, p in model.named_parameters():
        diff = torch.norm(p.detach() - params_before[name]).item()
        print(f"{name:30s} param_diff = {diff:.6e}")
        print(name, p.grad is None, p.grad.norm().item())
    # loss 曲线
    plt.figure();plt.plot(losses);plt.xlabel("Step");plt.ylabel("Loss");plt.title("Dry-run loss (same batch)");plt.show()
    
    

def train_model(model,train_loader,val_loader,epochs=30,learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    
    for epoch in range(epochs):
        # set epoch for convinence of validation, 3000 steps for each epoch
        steps = 3000
        
        # train
        model.train() # set as train mode, open dropout and batchnorm
        train_loss = 0.0
        start_time = time.time()
        for step,batch in enumerate(train_loader):
            data_time = time.time() - start_time
            promoters, exprs, ys = batch
            promoters = promoters.to(device, non_blocking=True)
            exprs = exprs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True).float()
            
            compute_start = time.time()
            
            optimizer.zero_grad()
            out = model(promoters,exprs).squeeze(1)
            loss = criterion(out, ys)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()*ys.size(0) 
            torch.cuda.synchronize() 
            compute_time = time.time() - compute_start
            
            if step % 10 == 0:
                print(f"batch {step} | data time: {data_time:.4f}s | compute time: {compute_time:.4f}")
            
            if step >= steps-1:
                break
            start_time = time.time()
        avg_train_loss = train_loss / steps / ys.size(0)
        train_loss = train_loss.append(avg_train_loss)
        
        
        # validation
        model.eval() # set as validation mode, close dropout and batchnorm
        val_loss = 0.0
        
        with torch.no_grad():
            for step,batch in enumerate(val_loader):
                promoters,exprs,ys = batch
                promoters = promoters.to(device, non_blocking=True)
                exprs = exprs.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True).float()
                
                out = model(promoters,exprs).squeeze(1)
                loss = criterion(out,ys)
                
                val_loss += loss.item()*ys.size(0)
                if step >= 999:
                    break
            avg_val_loss = val_loss / steps / ys.size(0)
            val_loss = val_loss.append(avg_val_loss)
            
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} ')
        
        # save best model parameters
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(),'best_model.pth')
    
    print("Training done.")
                
        
train_model(model, train_loader, val_loader,1)



