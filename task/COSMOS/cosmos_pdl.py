"""
COSMOS + Preference Distribution Learning (PDL) for multi-task learning (MTL).

This file integrates the PDL method into the COSMOS backbone [Ruchte & Grabocka, ICDM'21].
The PDL part corresponds to Algorithm 1 of the paper and is implemented in `train_pdl`.

External dependencies (COSMOS backbone):
    - utils.num_parameters / model_from_dataset / circle_points / dict_to_cuda
    - ..base.BaseMethod
    - hv.HyperVolume
All of the above are taken from the original COSMOS repository
(https://github.com/ruchtem/cosmos). Place this file under
`multi_objective/methods/cosmos/` of that repository so the imports resolve.
"""

import numpy as np
import torch
import torch.nn as nn

from MLP import MLP
from utils import num_parameters, model_from_dataset, circle_points
from ..base import BaseMethod
import utils
from hv import HyperVolume


##########################################################################################
# Upsampler module (original COSMOS code, unchanged)
##########################################################################################
class Upsampler(nn.Module):

    def __init__(self, K, child_model, input_dim):
        """
        In case of tabular data: append the sampled rays to the data instances (no upsampling).
        In case of image data:   use a transposed CNN for the sampled rays.
        """
        super().__init__()

        if len(input_dim) == 1:
            # tabular data
            self.tabular = True
        elif len(input_dim) == 3:
            # image data
            self.tabular = False
            self.transposed_cnn = nn.Sequential(
                nn.ConvTranspose2d(K, K, kernel_size=4, stride=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(K, K, kernel_size=6, stride=2, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Upsample(input_dim[-2:])
            )
        else:
            raise ValueError(f"Unknown dataset structure, expected 1 or 3 dimensions")

        self.child_model = child_model

    def forward(self, batch):
        x = batch['data']
        b = x.shape[0]
        a = batch['alpha'].repeat(b, 1).cuda()

        if not self.tabular:
            # use transposed convolution
            a = a.reshape(b, len(batch['alpha']), 1, 1)
            a = self.transposed_cnn(a)

        x = torch.cat((x, a), dim=1)
        return self.child_model(dict(data=x))

    def private_params(self):
        if hasattr(self.child_model, 'private_params'):
            return self.child_model.private_params()
        else:
            return []


##########################################################################################
# COSMOS method + PDL extension
##########################################################################################
class COSMOSMethod(BaseMethod):

    def __init__(self, objectives, alpha, lamda, dim, n_test_rays, **kwargs):
        """
        Instantiate the COSMOS solver.

        Args:
            objectives:   list of objectives
            alpha:        Dirichlet sampling parameter (list or float)
            lamda:        cosine similarity penalty (disabled in this PDL-integrated version)
            dim:          dimensions of the data
            n_test_rays:  number of test rays used for evaluation
        """
        self.objectives = objectives
        self.K = len(objectives)
        self.alpha = alpha
        self.n_test_rays = n_test_rays
        self.lamda = lamda

        dim = list(dim)
        dim[0] = dim[0] + self.K

        model = model_from_dataset(method='cosmos', dim=dim, **kwargs)
        self.model = Upsampler(self.K, model, dim).cuda()
        self.kwargs = kwargs
        self.n_params = num_parameters(self.model)
        print("Number of parameters: {}".format(self.n_params))

    ##########################################################################################
    # Original COSMOS training step (without the cosine similarity regularization term).
    # Removing this term is part of our setting; see Table VII of the paper for the ablation
    # showing that COSMOS achieves better performance without this regularization.
    ##########################################################################################
    def step(self, batch):
        # Step 1: sample alphas.
        if isinstance(self.alpha, list):
            batch['alpha'] = torch.from_numpy(
                np.random.dirichlet(self.alpha, 1).astype(np.float32).flatten()
            ).cuda()
        elif self.alpha > 0:
            batch['alpha'] = torch.from_numpy(
                np.random.dirichlet([self.alpha for _ in range(self.K)], 1).astype(np.float32).flatten()
            ).cuda()
        else:
            raise ValueError(f"Unknown value for alpha: {self.alpha}, expecting list or float.")

        # Step 2: compute weighted task loss.
        self.model.zero_grad()
        logits = self.model(batch)
        batch.update(logits)
        loss_total = None
        task_losses = []
        for a, objective in zip(batch['alpha'], self.objectives):
            task_loss = objective(**batch)
            loss_total = a * task_loss if not loss_total else loss_total + a * task_loss
            task_losses.append(task_loss)

        loss_total.backward()
        return loss_total.item(), 0

    def eval_step(self, batch, test_rays=None):
        self.model.eval()
        logits = []
        with torch.no_grad():
            if test_rays is None:
                test_rays = circle_points(self.n_test_rays, dim=self.K)
            for ray in test_rays:
                ray = torch.from_numpy(ray.astype(np.float32)).cuda()
                ray /= ray.sum()
                batch['alpha'] = ray
                logits.append(self.model(batch))
        return logits

    ##########################################################################################
    # PDL training procedure (Algorithm 1).
    # Trains a Proxy F(lambda|omega) and a Generator G(lambda|phi) on top of the COSMOS model.
    ##########################################################################################
    def train_pdl(self, train_loader, scores):
        self.Model_projection = MLP(2).cuda()   # Proxy F(lambda|omega)
        self.generative = MLP(2).cuda()         # Generator G(lambda|phi)

        epoch = 10000
        generator_batch_size = 64
        prob_size = self.kwargs['prob_size']
        instance_size = self.kwargs['instance_size']

        Model_projection_optimization = torch.optim.Adam(self.Model_projection.parameters(), lr=0.003)
        optimization_generator = torch.optim.Adam(self.generative.parameters(), lr=0.003)
        milestones = [4000, 8000]
        scheduler_Model_projection = torch.optim.lr_scheduler.MultiStepLR(Model_projection_optimization, milestones, 0.1)
        scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(optimization_generator, milestones, 0.1)

        pref = torch.zeros([2])
        pref_list = torch.zeros([prob_size, instance_size, 2])
        sols_list = torch.zeros([prob_size, instance_size, 2])

        ##########################################################################################
        # [Step 1] Algorithm 1, Collect training data.
        # Query the COSMOS model with sampled preferences to obtain (preference, score) pairs.
        ##########################################################################################
        # First half: random preferences.
        for j in range(instance_size // 2):
            pref = torch.zeros([2])
            pref[0] = torch.rand(1)
            pref[1] = 1 - pref[0]
            pref_list[0][j] = pref

            score_values = np.array([])
            for batch in train_loader:
                s = []
                batch = utils.dict_to_cuda(batch)
                batch['alpha'] = pref
                logits = self.model(batch)
                batch.update(logits)
                s.append([s(**batch) for s in scores])
                if score_values.size == 0:
                    score_values = np.array(s)
                else:
                    score_values += np.array(s)
            score_values /= len(train_loader)
            sols_list[0][j] = torch.tensor(score_values)

        # Second half: evenly spaced preferences for boundary coverage.
        nsols = instance_size - instance_size // 2
        a = instance_size // 2
        for j in range(nsols):
            pref[0] = 1 - 1 / (nsols - 1) * j
            pref[1] = 1 / (nsols - 1) * j
            pref_list[0][j + a] = pref

            score_values = np.array([])
            for batch in train_loader:
                s = []
                batch = utils.dict_to_cuda(batch)
                batch['alpha'] = pref
                logits = self.model(batch)
                batch.update(logits)
                s.append([s(**batch) for s in scores])
                if score_values.size == 0:
                    score_values = np.array(s)
                else:
                    score_values += np.array(s)
            score_values /= len(train_loader)
            sols_list[0][j + a] = torch.tensor(score_values)

        # Min-max normalization per instance (Section III).
        obj1_min, obj2_min = min(sols_list[0][:, 0]), min(sols_list[0][:, 1])
        obj1_max, obj2_max = max(sols_list[0][:, 0]), max(sols_list[0][:, 1])
        sols_list[0][:, 0] = (sols_list[0][:, 0] - obj1_min) / (obj1_max - obj1_min)
        sols_list[0][:, 1] = (sols_list[0][:, 1] - obj2_min) / (obj2_max - obj2_min)

        ##########################################################################################
        # [Step 2] Algorithm 1, Train the Proxy model F(lambda|omega).
        ##########################################################################################
        for e in range(epoch):
            Model_projection_optimization.zero_grad()

            index = torch.randint(low=0, high=instance_size, size=(prob_size, 1))
            index = index.expand(prob_size, 2).reshape([prob_size, 1, 2])
            input = pref_list.gather(1, index).squeeze(1).cuda()
            pred = sols_list.gather(1, index).squeeze(1).cuda()
            x = self.Model_projection(input)

            loss = -torch.sum(torch.cosine_similarity(x, pred)) / prob_size
            print(loss.item())
            loss.backward()
            Model_projection_optimization.step()
            scheduler_Model_projection.step()

        torch.save(self.Model_projection, 'Model_projection.pth')

        ##########################################################################################
        # [Step 3] Algorithm 1, Train the Generator model G(lambda|phi).
        ##########################################################################################
        for e in range(epoch):
            optimization_generator.zero_grad()

            # Sample a batch of preferences uniformly from the simplex.
            data_ = torch.zeros([generator_batch_size, 2])
            data_[:, 0] = torch.rand([generator_batch_size])
            data_[:, 1] = 1 - data_[:, 0]
            data_ = data_.cuda()

            # lambda' = G(lambda|phi), then F(lambda'|omega).
            x = self.generative(data_)
            x = self.Model_projection(x)

            loss = -torch.sum(torch.cosine_similarity(data_, x, dim=1)) / generator_batch_size
            loss.backward()
            print(loss.item())
            optimization_generator.step()
            scheduler_generator.step()

        torch.save(self.generative, 'generator.pth')

    ##########################################################################################
    # Inference: combine preferences from the learned distribution P_G (via generator) with
    # uniform preferences, using the best mix ratio r* found by `eval_generator`.
    ##########################################################################################
    def eval_step_generator(self, batch):
        self.model.eval()
        logits = []
        ratio = self.max_ratio
        with torch.no_grad():
            # Subset 1: preferences from the learned distribution.
            n_sols = int(ratio * self.n_test_rays)
            for i in range(n_sols):
                pref = torch.zeros(2)
                pref[0] = 1 - 1 / (n_sols - 1) * i
                pref[1] = 1 / (n_sols - 1) * i
                pref = self.generative(pref.unsqueeze(0).cuda())[0]
                batch['alpha'] = pref.cuda()
                logits.append(self.model(batch))

            # Subset 2: evenly spaced uniform preferences.
            n_sols = self.n_test_rays - n_sols
            test_rays = circle_points(n_sols, dim=self.K)
            for ray in test_rays:
                pref = torch.from_numpy(ray.astype(np.float32))
                pref /= pref.sum()
                batch['alpha'] = pref.cuda()
                logits.append(self.model(batch))

        return logits

    ##########################################################################################
    # [Step 4] Algorithm 1, Search the best mix ratio r*.
    ##########################################################################################
    def eval_generator(self, test_loader, scores=None):
        self.model.eval()
        volume_list = []
        max_ratio = 0
        max_volume = 0
        n_nodes = 100
        reference_point = self.kwargs['reference_point']

        # Sweep candidate mix ratios r in {0.0, 0.1, ..., 0.9}.
        for j in range(10):
            ratio = j * 0.1

            # Subset 1: preferences from the learned distribution (via generator).
            sols_1 = []
            n_sols = int(ratio * n_nodes)
            for i in range(n_sols):
                pref = torch.zeros(2)
                pref[0] = 1 - 1 / (n_sols - 1) * i
                pref[1] = 1 / (n_sols - 1) * i
                pref = self.generative(pref.unsqueeze(0).cuda())[0].cuda()

                score_values = np.array([])
                for batch in test_loader:
                    s = []
                    batch = utils.dict_to_cuda(batch)
                    batch['alpha'] = pref
                    logits = self.model(batch)
                    batch.update(logits)
                    s.append([s(**batch) for s in scores])
                    if score_values.size == 0:
                        score_values = np.array(s)
                    else:
                        score_values += np.array(s)
                score_values /= len(test_loader)
                sols_1.append(score_values[0])

            # Subset 2: evenly spaced uniform preferences.
            n_sols = n_nodes - n_sols
            test_rays = circle_points(n_sols, dim=self.K)
            sols_2 = []
            for ray in test_rays:
                pref = torch.from_numpy(ray.astype(np.float32))
                pref /= pref.sum()

                score_values = np.array([])
                for batch in test_loader:
                    s = []
                    batch = utils.dict_to_cuda(batch)
                    batch['alpha'] = pref
                    logits = self.model(batch)
                    batch.update(logits)
                    s.append([s(**batch) for s in scores])
                    if score_values.size == 0:
                        score_values = np.array(s)
                    else:
                        score_values += np.array(s)
                score_values /= len(test_loader)
                sols_2.append(score_values[0])

            sols_2 = sols_1 + sols_2
            score_values = np.array(sols_2)
            hv = HyperVolume(reference_point)
            volume = hv.compute(score_values) if score_values.shape[1] < 5 else -1
            volume = volume / (reference_point[0] * reference_point[1])

            if volume > max_volume:
                max_volume = volume
                max_ratio = ratio
            volume_list.append(volume)

        self.max_ratio = max_ratio
        print(max_ratio, volume_list)