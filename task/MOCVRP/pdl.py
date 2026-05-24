"""
Preference Distribution Learning (PDL) for the Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP).
"""

##########################################################################################
# Machine environment config
##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Imports
##########################################################################################
import os
import sys
import time
import pickle
import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")     # for problem_def
sys.path.insert(0, "../..")  # for utils / common

import hvwfg

# --- PMOCO backbone (from the original PMOCO repository) ---
from MOCVRPTester import CVRPTester as Tester
from MOCVRProblemDef import get_random_problems

# --- PDL components (shared four-layer MLP for Proxy and Generator) ---
from common.MLP import MLP


##########################################################################################
# Hyperparameters
##########################################################################################
env_params = {
    'problem_size': 20,
    'pomo_size': 20,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '.',
        'epoch': 1,
    },
    'test_episodes': 100,
    'test_batch_size': 100,
    'augmentation_enable': True,
    'aug_factor': 1,
    'aug_batch_size': 100,
}

# Path to the pre-generated MOCVRP test set (provided by PMOCO).
TEST_DATA_PATH = './movrp20_test_seed1234.pkl'


##########################################################################################
# Main PDL training procedure (Algorithm 1)
##########################################################################################
def main():
    timer_start = time.time()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    epoch = 100
    generator_batch_size = 64
    prob_size = 20
    instance_size = 100

    Model_projection = MLP(2)   # Proxy F(lambda|omega)
    generator = MLP(2)          # Generator G(lambda|phi)

    optimization = torch.optim.Adam(Model_projection.parameters(), lr=0.003)
    optimization_generator = torch.optim.Adam(generator.parameters(), lr=0.003)
    milestones = [4000, 8000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimization, milestones, 0.1)
    scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(optimization_generator, milestones, 0.1)

    ##########################################################################################
    # [Step 1] Algorithm 1, Collect training data.
    # Note: MOCVRP's get_random_problems returns (depot_xy, node_xy, node_demand).
    ##########################################################################################
    pref = torch.zeros([2])
    pref_list = torch.zeros([prob_size, instance_size, 2])
    sols_list = torch.zeros([prob_size, instance_size, 2])

    for i in range(prob_size):
        shared_depot_xy, shared_node_xy, shared_node_demand = get_random_problems(100, 20)

        # First half: random preferences.
        for j in range(instance_size // 2):
            pref[0] = torch.rand([1])
            pref[1] = 1 - pref[0]
            pref_list[i][j] = pref
            sols_list[i][j] = torch.tensor(
                tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref))

        # Second half: evenly spaced preferences for boundary coverage.
        nsols = instance_size - instance_size // 2
        a = instance_size // 2
        for j in range(nsols):
            pref[0] = 1 - 1 / (nsols - 1) * i
            pref[1] = 1 / (nsols - 1) * i
            pref_list[i][j + a] = pref
            sols_list[i][j + a] = torch.tensor(
                tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref))

        # Min-max normalization per instance (Section III).
        obj1_min, obj2_min = min(sols_list[i][:, 0]), min(sols_list[i][:, 1])
        obj1_max, obj2_max = max(sols_list[i][:, 0]), max(sols_list[i][:, 1])
        sols_list[i][:, 0] = (sols_list[i][:, 0] - obj1_min) / (obj1_max - obj1_min)
        sols_list[i][:, 1] = (sols_list[i][:, 1] - obj2_min) / (obj2_max - obj2_min)

    ##########################################################################################
    # [Step 2] Algorithm 1, Train the Proxy model F(lambda|omega).
    ##########################################################################################
    for e in range(epoch):
        optimization.zero_grad()

        index = torch.randint(low=0, high=instance_size, size=(prob_size, 1))
        index = index.expand(prob_size, 2).reshape([prob_size, 1, 2])
        input = pref_list.gather(1, index).squeeze(1)
        pred = sols_list.gather(1, index).squeeze(1)
        x = Model_projection(input)

        loss = -torch.sum(torch.cosine_similarity(x, pred)) / prob_size
        print(loss.item())
        loss.backward()
        optimization.step()
        scheduler.step()

    torch.save(Model_projection, 'Model_projection.pth')

    ##########################################################################################
    # [Step 3] Algorithm 1, Train the Generator model G(lambda|phi).
    ##########################################################################################
    data = torch.zeros([generator_batch_size, 2])
    for e in range(epoch):
        optimization_generator.zero_grad()

        # Sample a batch of preferences uniformly from the simplex.
        data[:, 0] = torch.rand([generator_batch_size])
        data[:, 1] = 1 - data[:, 0]

        # lambda' = G(lambda|phi)
        pref = generator(data)
        # F(lambda'|omega) -- predicted loss direction for the generated preference.
        x = Model_projection(pref)

        loss = -torch.sum(torch.cosine_similarity(data, x, dim=1)) / generator_batch_size
        loss.backward()
        print(loss.item())
        optimization_generator.step()
        scheduler_generator.step()

    torch.save(generator, 'Generator.pth')

    total_time = time.time() - timer_start
    print('Run Time(s): {:.4f}'.format(total_time))
    return generator


##########################################################################################
# [Step 4] Algorithm 1, Search the best mix ratio r*.
##########################################################################################
def eval_generator():
    with open(TEST_DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    shared_depot_xy = torch.tensor([[x[0]] for x in data]).cuda()
    shared_node_xy = torch.tensor([x[1] for x in data]).cuda()
    shared_node_demand = (torch.tensor([x[2] for x in data]) / float(40)).cuda()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)
    generator = torch.load('Generator.pth')

    max_ratio, max_hv = 0, 0
    n_nodes = 20
    hv_list = []

    # Sweep the candidate mix ratios r in {0.0, 0.1, ..., 0.9}.
    for j in range(10):
        # Subset 1: preferences from the learned distribution (via generator).
        n_sols = int(0.1 * j * n_nodes)
        sols = np.zeros([n_sols, 2])
        for i in range(n_sols):
            pref = torch.zeros(2)
            pref[0] = 1 - 1 / (n_sols - 1) * i
            pref[1] = 1 / (n_sols - 1) * i
            pref = generator(pref.unsqueeze(0))[0]
            sols[i] = np.array(
                tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref))

        # Subset 2: evenly spaced uniform preferences.
        n_sols = n_nodes - n_sols
        sols_2 = np.zeros([n_sols, 2])
        for i in range(n_sols):
            pref = torch.zeros(2)
            pref[0] = 1 - 1 / (n_sols - 1) * i
            pref[1] = 1 / (n_sols - 1) * i
            pref = pref / torch.sum(pref)
            sols_2[i] = np.array(
                tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref))

        # Reference point follows Table I of the paper.
        ref = np.array([40, 3])
        sols = np.vstack((sols, sols_2))
        hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
        hv_ratio = hv / (ref[0] * ref[1])
        hv_list.append(hv_ratio)

        if max_hv < hv_ratio:
            max_hv = hv_ratio
            max_ratio = j * 0.1

    return max_ratio, hv_list


if __name__ == '__main__':
    main()
    # print(eval_generator())