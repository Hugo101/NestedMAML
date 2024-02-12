#!/usr/bin/env python3
import random
from torch import nn, optim
import os, sys
sys.path.append("../../metaL-dss")
from Sine_Regression.OOD_Tasks.tasks import *
import learn2learn as l2l
import time
import csv


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


class MAMLSinewaveRmOOD:

    def __init__(self, ood_lv, lr=0.005, maml_lr=0.01, num_iterations=10000, shots=20, tps=100, fas=5, num_tasks=10000, device='cuda'):
        self.lr = lr
        self.maml_lr = maml_lr
        self.iterations = num_iterations
        self.shots = shots
        self.tps = tps
        self.fas = fas
        self.num_tasks = num_tasks
        self.task_ood_lv = ood_lv
        self.device = torch.device(device)
        model = SyntheticMAMLModel()
        model.to(self.device)
        self.meta_model = l2l.algorithms.MAML(model, lr=self.maml_lr, first_order=False)
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
            meta_train_error = 0.0
            meta_valid_error = 0.0
            train_batch_indices = list(random.sample(list(np.arange(self.num_tasks)), meta_batch_size))
            valid_batch_indices = list(random.sample(list(np.arange(1024)), meta_batch_size))
            train_batch_tasks_list = []
            valid_batch_tasks_list = []
            for idx in range(meta_batch_size):
                # Compute meta-training loss
                if train_batch_indices[idx] in ood_indices:
                    pass
                else:
                    batch = train_tasks[train_batch_indices[idx]].sample_data(self.shots * 2)
                    train_batch_tasks_list.append(batch)

                # Compute meta-validation loss
                batch = val_tasks[valid_batch_indices[idx]].sample_data(self.shots * 2)
                valid_batch_tasks_list.append(batch)
            if len(train_batch_tasks_list) > 0:
                for task in range(len(train_batch_tasks_list)):
                    # Compute meta-training loss
                    learner = self.meta_model.clone()
                    batch = train_batch_tasks_list[task]
                    evaluation_error = self.fast_adapt(batch,
                                                       learner,
                                                       loss_func,
                                                       self.fas,
                                                       self.device, 1)
                    evaluation_error.backward()
                    meta_train_error += evaluation_error.item()

                    # Compute meta-validation loss
                    learner = self.meta_model.clone()
                    batch = valid_batch_tasks_list[task]
                    evaluation_error = self.fast_adapt(batch,
                                                       learner,
                                                       loss_func,
                                                       self.fas,
                                                       self.device, 1)
                    meta_valid_error += evaluation_error.item()

                # Print some metrics
                print('\n')
                print('Iteration', iteration)
                print('Meta Train Error', meta_train_error / meta_batch_size)
                print('Meta Valid Error', meta_valid_error / meta_batch_size)


                # Average the accumulated gradients and optimize
                for p in self.meta_model.parameters():
                    p.grad.data.mul_(1.0 / meta_batch_size)
                opt.step()

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
        with open(r'maml_rm_ood_sine_losses' + str(self.shots) + str(self.task_ood_lv) + '.txt', 'w') as csv_file:
            csv_file.writelines(str_losses)

        torch.save(self.meta_model, 'maml_rm_ood_sine_model' + str(self.shots) + str(self.task_ood_lv) + '.pl')

