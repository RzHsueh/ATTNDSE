import random
import time
import os
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
import pygmo as pg

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    
    def reset(self):
        self.n = 0
        self.v = 0



class Parameter():
    def __init__(self):
        self.cpu_clock_list = ["1GHz", "1.5GHz", "2GHz", "2.5GHz", "3GHz"]
        self.pipelineWidth_list = getRange(1, 12, 1)
        self.fetchBufferSize_list = [16, 32, 64]
        self.fetchQueueSize_list = getRange(8, 48, 4)
        self.branchPred_type_list = ["BiModeBP", "TournamentBP"]
        self.branchPred_choicePredictorSize_list = getRange(2048, 8192, 2048)
        self.branchPred_globalPredictorSize_list = getRange(2048, 8192, 2048)
        self.branchPred_ras_numEntries_list = getRange(16, 40, 2)
        self.branchPred_btb_numEntries_list = [1024, 2048, 4096]
        self.numROBEntries_list = getRange(32, 256, 16)
        self.numPhysIntRegs_list = getRange(64, 256, 8)
        self.numPhysFloatRegs_list = getRange(64, 256, 8)
        self.numIQEntries_list = getRange(16, 80, 8)
        self.LQEntries_list = getRange(20, 48, 4)
        self.SQEntries_list = getRange(20, 48, 4)
        self.IntALU_list = getRange(3, 8, 1)
        self.IntMultDiv_list = getRange(1, 4, 1)
        self.FpALU_list= getRange(1, 4, 1)
        self.FpMultDiv_list = getRange(1, 4, 1)
        self.cacheline_list = [32, 64]
        self.l1i_size_list = ["16kB", "32kB", "64kB"]
        self.l1i_assoc_list = [2, 4]
        self.l1d_size_list = ["16kB", "32kB", "64kB"]
        self.l1d_assoc_list = [2, 4]
        self.l2_size_list = ["128kB", "256kB"]
        self.l2_assoc_list = [2, 4]

        self.parameter_lists = {
            0: self.cpu_clock_list,
            1: self.pipelineWidth_list,
            2: self.fetchBufferSize_list,
            3: self.fetchQueueSize_list,
            4: self.branchPred_type_list,
            5: self.branchPred_choicePredictorSize_list,
            6: self.branchPred_globalPredictorSize_list,
            7: self.branchPred_ras_numEntries_list,
            8: self.branchPred_btb_numEntries_list,
            9: self.numROBEntries_list,
            10: self.numPhysIntRegs_list,
            11: self.numPhysFloatRegs_list,
            12: self.numIQEntries_list,
            13: self.LQEntries_list,
            14: self.SQEntries_list,
            15: self.IntALU_list,
            16: self.IntMultDiv_list,
            17: self.FpALU_list,
            18: self.FpMultDiv_list,
            19: self.cacheline_list,
            20: self.l1i_size_list,
            21: self.l1i_assoc_list,
            22: self.l1d_size_list,
            23: self.l1d_assoc_list,
            24: self.l2_size_list,
            25: self.l2_assoc_list
        }

        self.searched_parameter_vector = []
        

    def generateParameters(self):
        random_parameter = [
            getRandomIndex(len(self.cpu_clock_list)),
            getRandomIndex(len(self.pipelineWidth_list)),
            getRandomIndex(len(self.fetchBufferSize_list)),
            getRandomIndex(len(self.fetchQueueSize_list)),
            getRandomIndex(len(self.branchPred_type_list)),
            getRandomIndex(len(self.branchPred_choicePredictorSize_list)),
            getRandomIndex(len(self.branchPred_globalPredictorSize_list)),
            getRandomIndex(len(self.branchPred_ras_numEntries_list)),
            getRandomIndex(len(self.branchPred_btb_numEntries_list)),
            getRandomIndex(len(self.numROBEntries_list)),
            getRandomIndex(len(self.numPhysIntRegs_list)),
            getRandomIndex(len(self.numPhysFloatRegs_list)),
            getRandomIndex(len(self.numIQEntries_list)),
            getRandomIndex(len(self.LQEntries_list)),
            getRandomIndex(len(self.SQEntries_list)),
            getRandomIndex(len(self.IntALU_list)),
            getRandomIndex(len(self.IntMultDiv_list)),
            getRandomIndex(len(self.FpALU_list)),
            getRandomIndex(len(self.FpMultDiv_list)),
            getRandomIndex(len(self.cacheline_list)),
            getRandomIndex(len(self.l1i_size_list)),
            getRandomIndex(len(self.l1i_assoc_list)),
            getRandomIndex(len(self.l1d_size_list)),
            getRandomIndex(len(self.l1d_assoc_list)),
            getRandomIndex(len(self.l2_size_list)),
            getRandomIndex(len(self.l2_assoc_list))
        ]
        return random_parameter

    def getValue(self, parameter_vector):
        parameter_value = [
            self.cpu_clock_list[parameter_vector[0]],
            self.pipelineWidth_list[parameter_vector[1]],
            self.fetchBufferSize_list[parameter_vector[2]],
            self.fetchQueueSize_list[parameter_vector[3]],
            self.branchPred_type_list[parameter_vector[4]],
            self.branchPred_choicePredictorSize_list[parameter_vector[5]],
            self.branchPred_globalPredictorSize_list[parameter_vector[6]],
            self.branchPred_ras_numEntries_list[parameter_vector[7]],
            self.branchPred_btb_numEntries_list[parameter_vector[8]],
            self.numROBEntries_list[parameter_vector[9]],
            self.numPhysIntRegs_list[parameter_vector[10]],
            self.numPhysFloatRegs_list[parameter_vector[11]],
            self.numIQEntries_list[parameter_vector[12]],
            self.LQEntries_list[parameter_vector[13]],
            self.SQEntries_list[parameter_vector[14]],
            self.IntALU_list[parameter_vector[15]],
            self.IntMultDiv_list[parameter_vector[16]],
            self.FpALU_list[parameter_vector[17]],
            self.FpMultDiv_list[parameter_vector[18]],
            self.cacheline_list[parameter_vector[19]],
            self.l1i_size_list[parameter_vector[20]],
            self.l1i_assoc_list[parameter_vector[21]],
            self.l1d_size_list[parameter_vector[22]],
            self.l1d_assoc_list[parameter_vector[23]],
            self.l2_size_list[parameter_vector[24]],
            self.l2_assoc_list[parameter_vector[25]]
        ]
        return parameter_value

    def get_parameter_list(self, index):
        if index in self.parameter_lists:
            return self.parameter_lists[index]
        else:
            return f"index {index} error"

    def updateOneParemeter(self, current_parameter_vector, update_index):
        self.searched_parameter_vector.append(copy.deepcopy(current_parameter_vector))
        update_parameter_list = self.get_parameter_list(update_index)
        search_candidate = []
        for i in range(len(update_parameter_list)):
            current_parameter_vector[update_index] = i
            if current_parameter_vector not in self.searched_parameter_vector:
                search_candidate.append(copy.deepcopy(current_parameter_vector))

        # print(f"search_candidate is {search_candidate}")
        return search_candidate


# #############################################################
def getRandomIndex(length):
    random.seed(time.time())
    return random.randint(0, length - 1)

def getRange(start, end, step):
    if step == 0:
        raise ValueError("Step interval cannot be zero.")
    final_list = []
    adjusted_end = end + (1 if step > 0 else -1)
    for i in range(start, adjusted_end, step):
        final_list.append(i)
    if step > 0 and final_list and final_list[-1] > end:
        final_list.pop()
    elif step < 0 and final_list and final_list[-1] < end:
        final_list.pop()
    return final_list

# #############################################################

def dominates(p, q):
    # print()
    return all(x <= y for x, y in zip(p, q)) and any(x < y for x, y in zip(p, q))

def paretoFront(points):
    pareto_points = []
    
    for p in points:
        # Check if p is dominated by any point in pareto_points
        is_dominated = any(dominates(q, p) for q in pareto_points)
        if not is_dominated:
            # Remove all points in pareto_points that are dominated by p
            pareto_points = [q for q in pareto_points if not dominates(p, q)]
            pareto_points.append(p)
    
    return pareto_points

def decoupleZip(point_full):
    # each point_full is [(index, (metrics1, metrics2))]
    unzipped = list(zip(*point_full))
    index = list(unzipped[0])[0]
    point = list(unzipped[1])[0]
    return index, point

def decouplePointfullList(pointfull_list):
    point_list = []
    for point_full in pointfull_list:
        index, point = decoupleZip(point_full)
        point_list.append(point)
    return point_list

def extendParetoFront(p_full, current_Pareto_front):
    _, p = decoupleZip(p_full)

    is_dominated = False
    for q_full in current_Pareto_front:
        _, q = decoupleZip(q_full)
        if dominates(q, p):
            is_dominated = True
            break
    if not is_dominated:
        # Remove all points in pareto_points that are dominated by p
        before_Pareto_front = current_Pareto_front
        current_Pareto_front = []
        for q_full in before_Pareto_front:
            _, q = decoupleZip(q_full)
            if not dominates(p, q):
                current_Pareto_front.append(q_full)
                
        current_Pareto_front.append(p_full)
    return current_Pareto_front

def getParameterValue(index, all_searched_parameter_values):
    return all_searched_parameter_values[index]

def getReferencePoint(dataset):
    max_x = max(dataset, key=lambda x: x[0])[0]
    max_y = max(dataset, key=lambda x: x[1])[1]
    
    return [max_x, max_y]

def calPHV(Pareto_optimal_set, reference_point):
    # Create a hypervolume object
    hv = pg.hypervolume(Pareto_optimal_set)
    
    # Calculate the hypervolume
    hypervolume = hv.compute(reference_point)
    
    return hypervolume

def calculate_adrs(estimated_pareto_set, true_pareto_set):
    # Ensure the input arrays are numpy arrays
    estimated_pareto_set = np.array(estimated_pareto_set)
    true_pareto_set = np.array(true_pareto_set)
    
    def euclidean_distance(point1, point2):
        return np.linalg.norm(point1 - point2)
    
    mean_true_norm = np.mean([np.linalg.norm(t) for t in true_pareto_set])
    
    def relative_distance(point, pareto_set):
        distances = np.array([euclidean_distance(point, t) for t in pareto_set])
        min_distance = np.min(distances)
        return min_distance / mean_true_norm
    
    relative_distances = np.array([relative_distance(s, true_pareto_set) for s in estimated_pareto_set])
    adrs = np.mean(relative_distances)
    
    return adrs

# #############################################################

def generate_shuffle_indices(length):
    indices = list(range(length))
    random.shuffle(indices)
    return indices

def shuffle_with_indices(origin_data, shuffle_indices):
    shuffled_list = [origin_data[i] for i in shuffle_indices]
    return shuffled_list

# #############################################################

def drawPoint(all_searched_design_points, Pareto_optimal_set):
    plt.figure(figsize=(5, 3))
    # Extract x and y coordinates
    searched_x_coords = [point[0] for point in all_searched_design_points]
    searched_y_coords = [point[1] for point in all_searched_design_points]
    plt.scatter(searched_x_coords, searched_y_coords, label='Data', c='gray', marker='+', alpha=0.5)
    # plt.scatter(searched_x_coords, searched_y_coords, color='blue')

    pareto_x_coords = [point[0] for point in Pareto_optimal_set]
    pareto_y_coords = [point[1] for point in Pareto_optimal_set]

    plt.scatter(pareto_x_coords, pareto_y_coords, label='Pareto Frontier', edgecolors='r', facecolors='none', marker='o')

    # plt.scatter(pareto_x_coords, pareto_y_coords, color='red')
    # plt.title('Searched Points')
    plt.xlabel('CPI', fontsize='large')
    # plt.ylabel('Area', fontsize='large')
    plt.ylabel('Power', fontsize='large')
    # Moving the legend outside of the plot
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize='medium', borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    # plt.savefig('figure/cpi-power.pdf', format='pdf')
    plt.show()


def drawAttnWeight(attn_weight):
    attn_weight = attn_weight.to('cpu')
    im = plt.imshow(attn_weight, cmap=plt.cm.gray_r, interpolation='nearest')

    cbar = plt.colorbar(im, orientation='vertical', fraction=0.1, label='Data Values')

    plt.title('Attention Weight Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

# #############################################################
def storeData(data_dict, dataset_name):
    prefix = "experiment_results/"
    folder_name = prefix + dataset_name
    os.makedirs(folder_name, exist_ok=True)

    for name, data in data_dict.items():
        file_path = os.path.join(folder_name, f'{name}.json')
        with open(file_path, 'w') as file:
            json.dump(data, file)


def loadData(file_name, dataset_name):
    prefix = "experiment_results/"
    folder_name = prefix + dataset_name
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


