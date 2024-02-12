#!/usr/bin/env python3
import random
from torch import nn, optim
import os, sys
sys.path.append("../../metaL-dss")
from Sine_Regression.OOD_Tasks.tasks import *
import learn2learn as l2l
import higher
import time
import csv
from matplotlib import pyplot as plt

class SyntheticMAMLModel(nn.Module):
    def __init__(self):
        super(SyntheticMAMLModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1))

    def forward(self, x):
        return self.model(x)


class FixedWeightSinewave:
    def __init__(self, ood_lv, lr=0.005, maml_lr=0.01, num_iterations=10000, shots=20, tps=100, fas=5, num_tasks=10000, weight_lr=1e-3, device='cuda'):
        self.lr = lr
        self.maml_lr = maml_lr
        self.iterations = num_iterations
        self.weight_lr = weight_lr
        self.shots = shots
        self.tps = tps
        self.fas = fas
        self.num_tasks = num_tasks
        self.task_ood_lv = ood_lv
        self.device = torch.device(device)
        self.model = SyntheticMAMLModel()
        self.model.to(self.device)
        self.meta_model = l2l.algorithms.MAML(self.model, lr=self.maml_lr, first_order=False)
        self.test_losses = []

    def fast_adapt(self, batch, learner, loss, adaptation_steps, device, task_weight):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evalutation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[0: int(data.size(0)/2)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

        # Adapt the model
        for step in range(adaptation_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            # train_error /= len(adaptation_data)
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = task_weight * loss(predictions, evaluation_labels)
        valid_error /= len(evaluation_data)
        return valid_error

    def reweight_algo(self, model, opt, train_batch_list, valid_batch_list, task_weights, lr, loss, device):
        eps = torch.nn.Parameter(task_weights.to(device), requires_grad=True)
        model.register_parameter("eps", eps)
        with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
            forward_loss = 0
            for i in range(len(train_batch_list)):
                data, labels = train_batch_list[i]
                data, labels = data.to(device), labels.to(device)
                # Separate data into adaptation/evalutation sets
                adaptation_indices = np.zeros(data.size(0), dtype=bool)
                adaptation_indices[0: int(data.size(0)/2)] = True
                adaptation_indices = torch.from_numpy(adaptation_indices)
                adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
                outputs_hat = fmodel(adaptation_data)
                forward_loss += (loss(outputs_hat, adaptation_labels)/len(adaptation_data)) * fmodel.eps[i]
            diffopt.step(forward_loss)
            backward_loss = 0
            for i in range(len(valid_batch_list)):
                data, labels = valid_batch_list[i]
                data, labels = data.to(device), labels.to(device)
                outputs_hat = fmodel(data)
                backward_loss += (loss(outputs_hat, labels)/len(data))
            grad_all = torch.autograd.grad(backward_loss, fmodel.parameters(time=0), only_inputs=True)[0]
            w_tilde = torch.clamp(task_weights - lr * grad_all, min=0)
            #norm_c = torch.sum(w_tilde)
            """
            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde
            """
        return w_tilde

    def main(self):
        opt = optim.Adam(self.meta_model.parameters(), lr=self.lr)
        loss_func = nn.MSELoss(reduction='mean')
        meta_batch_size = self.tps
        # Sine Waves
        amplitude_min = 0.1
        amplitude_max = 5
        phase_min = 0
        phase_max = np.pi
        x_min = -5
        x_max = 5
        tasks = Sine_Task_Distribution(amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max)
        weights = torch.ones(self.num_tasks)/self.num_tasks
        weights = weights.to(self.device)
        train_tasks = []
        for i in range(self.num_tasks):
            train_tasks.append(tasks.sample_task())
        ood_indices = np.arange(self.num_tasks)[0:int(self.task_ood_lv*self.num_tasks)]
        val_tasks = []
        for i in range(1024):
            val_tasks.append(tasks.sample_task())
        test_tasks = []
        for i in range(1024):
            test_tasks.append(tasks.sample_task())
        for iteration in range(self.iterations+1):
            start_it = time.time()
            opt.zero_grad()
            train_batch_indices = list(random.sample(list(np.arange(self.num_tasks)), meta_batch_size))
            valid_batch_indices = list(random.sample(list(np.arange(1024)), meta_batch_size))
            train_batch_tasks_list = []
            valid_batch_tasks_list = []
            for idx in range(meta_batch_size):
                # Compute meta-training loss
                if train_batch_indices[idx] in ood_indices:
                    batch = train_tasks[train_batch_indices[idx]].sample_outliers(self.shots*2)
                else:
                    batch = train_tasks[train_batch_indices[idx]].sample_data(self.shots * 2)
                train_batch_tasks_list.append(batch)

                # Compute meta-validation loss
                batch = val_tasks[valid_batch_indices[idx]].sample_data(self.shots * 2)
                valid_batch_tasks_list.append(batch)

            maml_reweight = l2l.algorithms.MAML(self.model, lr=self.maml_lr, first_order=False)
            maml_reweight.load_state_dict(self.meta_model.state_dict())
            tmp_weights = weights[train_batch_indices].detach().clone()
            task_weights = self.reweight_algo(maml_reweight, opt, train_batch_tasks_list, valid_batch_tasks_list, tmp_weights, self.weight_lr, loss_func, self.device).detach()
            weights[train_batch_indices] = task_weights
            #weights = weights/torch.sum(weights)
            task_weights = weights[train_batch_indices]
            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0
            for i in range(meta_batch_size):
                # Compute meta-training loss
                learner = self.meta_model.clone()
                batch = train_batch_tasks_list[i]
                evaluation_error = self.fast_adapt(batch, learner, loss_func, self.fas, self.device, task_weights[i])
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()

                learner = self.meta_model.clone()
                batch = valid_batch_tasks_list[i]
                valid_batch_tasks_list.append(batch)
                evaluation_error = self.fast_adapt(batch, learner, loss_func, self.fas, self.device, 1)
                meta_valid_error += evaluation_error.item()
            for p in self.meta_model.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()
            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta-Fixed-Weight Train Error', meta_train_error / meta_batch_size)
            print('Meta-Fixed-Weight Valid Error', meta_valid_error / meta_batch_size)
            if iteration % 10 == 0:
                meta_test_error = 0.0
                meta_batch_size = self.tps
                test_batch_indices = list(random.sample(list(np.arange(1024)), meta_batch_size))
                for task in range(meta_batch_size):
                    # Compute meta-testing loss
                    learner = self.meta_model.clone()
                    batch = test_tasks[test_batch_indices[task]].sample_data(self.shots*2)
                    evaluation_error = self.fast_adapt(batch, learner, loss_func, self.fas, self.device, 1)
                    meta_test_error += evaluation_error.item()
                print('Meta Test Error', meta_test_error / meta_batch_size)
                self.test_losses.append(meta_test_error / meta_batch_size)
            print("time used per iter:", time.time() - start_it, "for iter:", iteration)
        str_losses = [str(x) for x in self.test_losses]
        with open(r'maml_fixedwt_sine_losses' + str(self.shots) + str(self.task_ood_lv) + '.txt', 'w') as csv_file:
            csv_file.writelines(str_losses)

        torch.save(self.meta_model, 'maml_fixedwt_sine_model' + str(self.shots) + str(self.task_ood_lv) + '.pl')
        weights = weights/weights.sum()
        ood_weights = weights[ood_indices].mean()
        non_ood_weights = weights[~ood_indices].mean()

        with open(r'maml_fixedwt_wts' + str(self.shots) + str(self.task_ood_lv) + '.txt', 'w') as txt_file:
            txt_file.write(str(ood_weights))
            txt_file.write(str(non_ood_weights))