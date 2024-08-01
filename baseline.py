import warnings
warnings.filterwarnings("ignore")
import os
import sys
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm


from config import args
from dataset import DSEDataset
from model import Model
from utils import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
import joblib


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)




def compute_ci(results, n_trials):
    results = np.array(results)
    mean = np.mean(results)
    std = np.std(results)
    ci = 1.96 * std / np.sqrt(n_trials)
    return mean, ci
    

def MAPE(y, y_pred):
    mape_loss = metrics.mean_absolute_percentage_error(y, y_pred)
    return mape_loss


def load_data(dataset_name):
    
    ## ======================  data  ==================== ##
    prefix = "data/generated_data/"
    dataset_path = osp.join(prefix, f"{dataset_name}.txt")
    
    trainset = DSEDataset(path=dataset_path, mode="train", target=args.target)
    valset = DSEDataset(path=dataset_path, mode="val", target=args.target)
    
    train_x, train_y = torch.stack([item[0] for item in trainset]), torch.stack([item[1] for item in trainset])
    val_x, val_y = torch.stack([item[0] for item in valset]), torch.stack([item[1] for item in valset])
    
    return train_x, train_y, val_x, val_y
    
def train_ml(dataset_name, model_name="MLP"):

    train_x, train_y, val_x, val_y = load_data(dataset_name)
    seed = random.randint(0, 10000)
    n_trials = 1
    r2s, mses, mapes = [], [], []

    for i in tqdm(range(n_trials)):
    ## ======================  model  =================== ##
        if model_name == "DecisionTreeRegressor":
            ml_model = DecisionTreeRegressor(criterion='squared_error')      
            ml_model = ml_model.fit(train_x, train_y)

        if model_name == "MoDSE":
            params = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8}
            base_model = GradientBoostingRegressor(**params, random_state=seed)
            ml_model = AdaBoostRegressor(
                estimator=base_model,
                learning_rate=0.001, # why
                n_estimators=40, # default 50
                random_state=seed
            )  

        if model_name == "BOOMExplorer":
            kernel = kernels.Matern()
            ml_model = GaussianProcessRegressor(kernel, random_state=seed)
            ml_model = ml_model.fit(train_x, train_y)
            
        if model_name == "MLP":
            ml_model = MLPRegressor(hidden_layer_sizes=(26, 10, 1), activation='relu', 
                                    solver='lbfgs', random_state=seed,
                                    # when use adam or sgd, the following options are required
                                    batch_size=32, learning_rate='invscaling', learning_rate_init=3e-4, 
                                    max_iter=100 )
        
        if model_name == "ActBoost":
            base_model = DecisionTreeRegressor(criterion='squared_error', max_depth=8)      
            ml_model = AdaBoostRegressor(
                estimator=base_model,
                learning_rate=0.001, # why
                n_estimators=20, # default 50
                random_state=seed
            )  
                
        ml_model = ml_model.fit(train_x, train_y)
        ## ======================  val  ===================== ##
        val_pred_y = ml_model.predict(val_x)

        if isinstance(val_y, torch.Tensor):
            val_y = val_y.cpu().numpy()

        if isinstance(val_pred_y, torch.Tensor):
            val_pred_y = val_pred_y.cpu().numpy()

        r2s.append(metrics.r2_score(val_y, val_pred_y))
        mses.append(np.power(val_y - val_pred_y, 2).mean())
        mapes.append(MAPE(val_y, val_pred_y))
        
    mean_r2, ci_r2 = compute_ci(r2s, n_trials)
    mean_mse, ci_mse = compute_ci(mses, n_trials)
    mean_mape, ci_mape = compute_ci(mapes, n_trials)
    data_dict = {
                f"R^2_{args.target}": mean_r2,
                f"MSE_{args.target}": mean_mse,
                f"MAPE_{args.target}": mean_mape
                }
    storeData(data_dict, f"{model_name}/{dataset_name}")

    os.makedirs(os.path.dirname(f"model/{model_name}/"), exist_ok=True)
    joblib.dump(ml_model, f"model/{model_name}/{dataset_name}-{args.target}.plk")

    
def load_ml_model(dataset_name, target, model_name):
    prefix = f"model/{model_name}/"
    model_path = f"{prefix}/{dataset_name}-{target}.plk"
    if not os.path.exists(model_path):
        train_ml(dataset_name, model_name)
    ml_model = joblib.load(model_path)
    return ml_model


