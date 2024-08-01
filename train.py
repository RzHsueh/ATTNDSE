import torch
import torch.nn as nn
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from sklearn.metrics import r2_score
from tqdm import tqdm

from config import args
from dataset import DSEDataset
from model import Model
from utils import *



torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def MAPE(y_pred, y):
    y_pred = y_pred.cpu()
    y = y.cpu()
    mape_loss = metrics.mean_absolute_percentage_error(y, y_pred)
    return mape_loss

def train(dataset_name:str = "600.perlbench_s", stroe_prefix:str = "model/true_model/"):
    if stroe_prefix == "model/true_model/":
        prefix = "data/train_data/"
    elif stroe_prefix == "model/AttentionDSE/":
        prefix = "data/generated_data/"
    dataset_path = prefix + dataset_name + ".txt"
    ## ======================  data  ==================== ##
    trainset = DSEDataset(path=dataset_path, mode="train", target=args.target)
    trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, 
                             num_workers=4, pin_memory=True, drop_last=False, shuffle=True)
    
    valset = DSEDataset(path=dataset_path, mode="val", target=args.target)
    valloader = DataLoader(dataset=valset, batch_size=args.batch_size, 
                             num_workers=4, pin_memory=True, drop_last=False, shuffle=True)
 
    ## ======================  model  ==================== ##
    model = Model(depth=args.depth, embed_dim=args.embed_dim, num_heads=args.num_heads,dropout=args.dropout).to(args.device)
    
    ## ====================== misc ===================== ##
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    

    for epoch in (bar := tqdm(range(args.epochs))):
        
        train_loss = Averager()
        train_acc = Averager()
        # =========================train=======================
        for input, label in trainloader:
            optimizer.zero_grad()
            model.train()

            input = input.to(args.device)
            label = label.to(args.device)

            output, _ = model(input)
            output = output.reshape(label.shape)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss.add(loss.item())


            output_cpu = output.detach().cpu().numpy()
            label_cpu = label.detach().cpu().numpy()
            train_acc.add(r2_score(label_cpu, output_cpu))

            bar.set_description(
                "Epoch:{} Train: R^2 = {}, MSE = {}".format(epoch, train_acc.item(), train_loss.item()))

        lr_scheduler.step()


        # import pdb;pdb.set_trace()
        # =========================val=======================
        if epoch == args.epochs-1:
            if stroe_prefix != "model/true_model/":
                val_loss = Averager()
                val_acc = Averager()
                mape = Averager()
                with torch.no_grad():
                    for input, label in valloader:
                        model.eval()

                        input = input.to(args.device)
                        label = label.to(args.device)
                        
                        output, _ = model(input)
                        output = output.reshape(label.shape)

                        loss = criterion(output, label)
                        val_loss.add(loss.item())
                        mape.add(MAPE(output, label))

                        output_cpu = output.detach().cpu().numpy()
                        label_cpu = label.detach().cpu().numpy()
                        val_acc.add(r2_score(label_cpu, output_cpu))
                data_dict = {
                    f"R^2_{args.target}": val_acc.item(),
                    f"MSE_{args.target}": val_loss.item(),
                    f"MAPE_{args.target}": mape.item()
                }
                storeData(data_dict, f"AttentionDSE/{dataset_name}")
                
    os.makedirs(os.path.dirname(stroe_prefix), exist_ok=True)
    torch.save(model.state_dict(), stroe_prefix+dataset_name+"-"+args.target+".pth")

if __name__ == '__main__':
    train()



