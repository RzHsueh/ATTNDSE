import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import time
from tqdm import tqdm
import joblib


from config import args
from utils import *
from dataset import TxTforCSV
from model import Model
from baseline import train_ml
from train import train


def loadTrueModel(dataset_name, target):
    model_path = "model/true_model/"+dataset_name+"-"+target+".pth"
    if not os.path.exists(model_path):
        print("Model file does not exist:", model_path)
        args.target = target
        train(dataset_name, "model/true_model/")
    model = Model(depth=args.depth, embed_dim=args.embed_dim, num_heads=args.num_heads,dropout=args.dropout).to(args.device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_true(dataset_name, input_tensor):
    predictor_ipc = loadTrueModel(dataset_name, "ipc")
    predictor_power = loadTrueModel(dataset_name, "power")
    predictor_area = loadTrueModel(dataset_name,"area")
    with torch.no_grad():
        pred_ipc, attn_weight_ipc = predictor_ipc(input_tensor)
        pred_power, attn_weight_power = predictor_power(input_tensor)
        pred_area, attn_weight_area = predictor_area(input_tensor)
        # drawAttnWeight(attn_weight_ipc+attn_weight_power+attn_weight_area)
    return pred_ipc, pred_power, pred_area, attn_weight_ipc, attn_weight_power, attn_weight_area

def loadModel(dataset_name, target, model_name):
    if model_name == "AttentionDSE":
        model_path = f"model/{model_name}/{dataset_name}-{target}.pth"
        if not os.path.exists(model_path):
            print("Model file does not exist:", model_path)
            args.target = target
            train(dataset_name, f"model/{model_name}/")
        model = Model(depth=args.depth, embed_dim=args.embed_dim, num_heads=args.num_heads,dropout=args.dropout).to(args.device)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
    elif model_name in ["MoDSE", "BOOMExplorer", "MLP", "ActBoost"]:
        model_path = f"model/{model_name}/{dataset_name}-{target}.plk"
        if not os.path.exists(model_path):
            print("Model file does not exist:", model_path)
            args.target = target
            train_ml(dataset_name, model_name)
        model = joblib.load(model_path)


    return model

def predict(dataset_name, input_tensor, model_name):
    attn_weight_ipc, attn_weight_power, attn_weight_area = [], [], []
    if model_name == "AttentionDSE":
        predictor_ipc = loadModel(dataset_name, "ipc", model_name)
        predictor_power = loadModel(dataset_name, "power", model_name)
        predictor_area = loadModel(dataset_name,"area", model_name)
        with torch.no_grad():
            pred_ipc, attn_weight_ipc = predictor_ipc(input_tensor)
            pred_power, attn_weight_power = predictor_power(input_tensor)
            pred_area, attn_weight_area = predictor_area(input_tensor)
            # drawAttnWeight(attn_weight_ipc+attn_weight_power+attn_weight_area)
    elif model_name in ["MoDSE", "BOOMExplorer", "MLP", "ActBoost"]:
        predictor_ipc = loadModel(dataset_name, "ipc", model_name)
        predictor_power = loadModel(dataset_name, "power", model_name)
        predictor_area = loadModel(dataset_name,"area", model_name)

        pred_ipc = predictor_ipc.predict(input_tensor) 
        pred_power = predictor_power.predict(input_tensor) 
        pred_area = predictor_area.predict(input_tensor) 

    return pred_ipc, pred_power, pred_area, attn_weight_ipc, attn_weight_power, attn_weight_area

# Get the Pareto front from current dataset
def readDataset(dataset_name):
    dataset_path = f"data/generated_data/{dataset_name}.txt"
    if not os.path.exists(dataset_path):
        print("Dataset file does not exist:", dataset_path)
        generateDataset(dataset_name, 1000000)

    origin_parameter, ipc, power, area = TxTforCSV(f"data/generated_data/{dataset_name}.txt")
    cpi = [1/ipc_instance for ipc_instance in ipc]
    parameter = []
    for i, item in enumerate(origin_parameter):
        parameter.append([int(num) for num in item.split(',')])
    if args.moo == "cpi-power":
        points = list(zip(cpi, power))
    elif args.moo == "cpi-area":
        points = list(zip(cpi, area)) 
    print(f"The number of all design points is {len(points)}.")
    return points

# generate data by the predict model
def generateDataset(dataset_name, num):
    filename = "data/generated_data/" + dataset_name + ".txt"
    parameter_generator = Parameter()
    print(f"Generating dataset...")
    with open(filename, "a", encoding="utf-8") as file:
        for _ in tqdm(range(num)):
            parameter_vector = parameter_generator.generateParameters()
            
            input_tensor = torch.tensor(parameter_vector).unsqueeze(0).to(args.device)
            true_ipc, true_power, true_area, _, _, _ = predict_true(dataset_name, input_tensor)
            true_ipc = true_ipc.item()
            true_power = true_power.item()
            true_area = true_area.item()

            parameter_vector = ','.join(map(str, parameter_vector))

            data = f"{parameter_vector} {pred_ipc} {pred_power} {pred_area}"
            
            file.write(data + '\n')



if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="600.perlbench_s")
    parser.add_argument('--moo', type=str, default="cpi-power")
    parser_instance = parser.parse_args()
    args.set_from_parser(parser_instance)

    print(f"Current dataset is {args.dataset}, optimization objetive is {args.moo}.")

    dataset_name = args.dataset
    dataset = readDataset(dataset_name) # dataset [(objective1, objective2)]
    true_Pareto_optimal_set = paretoFront(dataset)
    reference_point = getReferencePoint(dataset)
    print(f"reference_point is {reference_point}")

    # true model
    print(f"Load true Predictor...")
    true_predictor_ipc = loadTrueModel(dataset_name, "ipc")
    true_predictor_power = loadTrueModel(dataset_name, "power")
    true_predictor_area = loadTrueModel(dataset_name,"area")
    print(f"Load true Predictor success.")


    sample_num = 10
    iteration = 100
    parameter_generator = Parameter()



# AttentionDSE
    model_name = "AttentionDSE"
    PHV_file_path = f'experiment_results/efficiency_HEA/{model_name}/{dataset_name}/{model_name}_PHV_{args.moo}.txt'
    ADRS_file_path = f'experiment_results/efficiency_HEA/{model_name}/{dataset_name}/{model_name}_ADRS_{args.moo}.txt'
    os.makedirs(os.path.dirname(PHV_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(ADRS_file_path), exist_ok=True)
    if os.path.exists(PHV_file_path):
        open(PHV_file_path, 'w').close()
    if os.path.exists(ADRS_file_path):
        open(ADRS_file_path, 'w').close()

    print(f"{model_name} Search...")
    with open(PHV_file_path, 'a') as phv_file, open(ADRS_file_path, 'a') as adrs_file:
        all_searched_parameter_values = [] # record all searched parameters, using for backtracking
        all_searched_parameter_vector = [] # record all searched parameter vectors, using for backtracking
        all_searched_design_point_with_parameters = [] # record all design point metrics,  [(index, (metrics1, metrics2))]
        all_random_searched_design_point_with_parameters = [] # record all design point metrics which generated random,  [(index, (metrics1, metrics2))]
        search_queue = []
        current_Pareto_front = []
        for it in (bar := tqdm(range(iteration))):
            random_design_point_search_num = 0
            while random_design_point_search_num < sample_num:
                # generate the design point
                generated_random = False
                if(len(search_queue) == 0):
                    initial_vector = parameter_generator.generateParameters()
                    search_queue.append(initial_vector)
                    generated_random = True
                    random_design_point_search_num += 1

                parameter_vector = search_queue.pop(0)
                all_searched_parameter_vector.append(parameter_vector)
                # print(f"parameter_vector is {parameter_vector}")
                parameter_value = parameter_generator.getValue(parameter_vector)
                # print(f"parameter_value is {parameter_value}")
                all_searched_parameter_values.append(parameter_value)
                current_index = len(all_searched_parameter_values)-1

                # predict the metrics
                input_tensor = torch.tensor(parameter_vector).unsqueeze(0).to(args.device)
                pred_ipc, pred_power, pred_area, attn_weight_ipc, attn_weight_power, attn_weight_area = predict(dataset_name, input_tensor, model_name)
                pred_ipc = pred_ipc.item()
                pred_power = pred_power.item()
                pred_area = pred_area.item()

                # generate the Pareto optimal set
                if args.moo == "cpi-power":
                    design_point = list(zip([1/(pred_ipc)], [pred_power]))
                elif args.moo == "cpi-area":
                    design_point = list(zip([1/(pred_ipc)], [pred_area]))

                design_point_with_parameters = list(zip([current_index], design_point))
                all_searched_design_point_with_parameters.append(design_point_with_parameters)
                if generated_random:
                    all_random_searched_design_point_with_parameters.append(design_point_with_parameters)
                current_Pareto_front = extendParetoFront(design_point_with_parameters, current_Pareto_front)

                is_dominated = any(dominates(q, design_point[0]) for q in decouplePointfullList(current_Pareto_front))
                if not is_dominated and generated_random:
                    # update search_queue
                    # remove the last row and column
                    attn_weight_ipc = attn_weight_ipc[:-1, :-1].sum(axis=0).cpu()
                    attn_weight_power = attn_weight_power[:-1, :-1].sum(axis=0).cpu()
                    attn_weight_area = attn_weight_area[:-1, :-1].sum(axis=0).cpu()

                    attn_weight_ipc_min_index = np.argmin(attn_weight_ipc).item()
                    attn_weight_power_max_index = np.argmax(attn_weight_power).item()
                    attn_weight_area_max_index = np.argmax(attn_weight_area).item()

                    if args.moo == "cpi-power":
                        search_queue.extend(parameter_generator.updateOneParemeter(parameter_vector, attn_weight_ipc_min_index))
                        search_queue.extend(parameter_generator.updateOneParemeter(parameter_vector, attn_weight_power_max_index))
                    elif args.moo == "cpi-area":
                        search_queue.extend(parameter_generator.updateOneParemeter(parameter_vector, attn_weight_ipc_min_index))
                        search_queue.extend(parameter_generator.updateOneParemeter(parameter_vector, attn_weight_area_max_index))
            
            all_searched_design_point = []
            for point_full in all_searched_design_point_with_parameters:
                _, point = decoupleZip(point_full)
                all_searched_design_point.append(point)
            # print(f"The number of searched design point is {len(all_searched_design_point)}.")

            final_Pareto_optimal_vector_set = []
            for point_full in current_Pareto_front:
                index, _ = decoupleZip(point_full)
                final_Pareto_optimal_vector_set.append(all_searched_parameter_vector[index])

            # Calculate the true performance metrice of the design points DSE framework searched
            final_Pareto_optimal_set = []
            for vector in final_Pareto_optimal_vector_set:
                input_tensor = torch.tensor(vector).unsqueeze(0).to(args.device)
                true_ipc, _ = true_predictor_ipc(input_tensor)
                true_power, _ = true_predictor_power(input_tensor)
                true_area, _ = true_predictor_area(input_tensor)

                true_ipc = true_ipc.item()
                true_power = true_power.item()
                true_area = true_area.item()

                if args.moo == "cpi-power":
                    design_point = [1/(true_ipc), true_power]
                elif args.moo == "cpi-area":
                    design_point = [1/(true_ipc), true_area]
                
                final_Pareto_optimal_set.append(design_point)
            
            phv = calPHV(final_Pareto_optimal_set, reference_point)
            adrs = calculate_adrs(final_Pareto_optimal_set, true_Pareto_optimal_set)
            
            phv_file.write(f'{phv}\n')
            adrs_file.write(f'{adrs}\n')

    print(f"Finished {model_name}")





