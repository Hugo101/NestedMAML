#!/usr/bin/env python3
import sys
sys.path.append("../OOD-Fixed-WT-MiniImageNet")
import random
import numpy as np
import numpy.matlib as matlib
import torch
from torch import nn
from torch import optim
from torchvision import transforms
import higher
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import time
import argparse
import os
import pickle
from _collections import defaultdict


parser = argparse.ArgumentParser(description='Learn2Learn miniImageNet Example')
parser.add_argument('--ways', type=int, default=5, metavar='N',
                    help='number of ways (default: 5)')
parser.add_argument('--meta_lr', type=float, default=0.001, metavar='LR',
                    help='meta learning rate (default: 0.001)')
parser.add_argument('--fast_lr', type=float, default=0.01, metavar='LR',
                    help='learning rate for base learner (default: 0.01)')
parser.add_argument('--meta_batch_size', type=int, default=10, metavar='N',
                    help='tasks per step (default: 32)')
parser.add_argument('--adaptation_steps', type=int, default=5, metavar='N',
                    help='steps per fast adaption (default: 5)')
parser.add_argument('--adaptation_steps_test', type=int, default=10, metavar='N',
                    help='steps per fast adaption (default: 10)')
parser.add_argument('--num_iterations', type=int, default=60000, metavar='N',
                    help='number of iterations (default: 1000)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=43, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--query_points', type=int, default=15, metavar='S',
                    help='Query points for evaluation data')
parser.add_argument('--noise_ratio', type=float, default=0.2, help='Noise Ratio')
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--weight_lr', type=float, default=0.01)
parser.add_argument('--thres', type=float, default=0.5)
parser.add_argument('--total_shots', type=int, default=25, help="Total no of shots")
parser.add_argument('--shots', type=int, default=5, help='Shots per each class')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


class MetaDataSet:

    def __init__(self, data):
        self.dataset = data
        self.indices_to_labels = self.generate_indices_to_labels()
        self.labels_to_indices = self.generate_labels_to_indices()
        self.labels = list(np.unique(self.dataset.y))

    def generate_indices_to_labels(self):
        tmp_dict = defaultdict(int)
        for i in range(len(self.dataset.y)):
            tmp_dict[i] = self.dataset.y[i]
        return tmp_dict

    def generate_labels_to_indices(self):
        tmp_dict = defaultdict(list)
        unique_classes = np.unique(self.dataset.y)
        for cls in unique_classes:
            idx = np.where(self.dataset.y == cls)[0]
            tmp_dict[cls] = list(idx)
        return tmp_dict


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, sample_indices, learner, loss, adaptation_steps, device, task_weight):
    adaptation_data, adaptation_labels = batch[0]
    adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)
    # Separate data into support/query (adaptation/evalutation) sets
    evaluation_data, evaluation_labels = batch[1]
    evaluation_data, evaluation_labels = evaluation_data.to(device), evaluation_labels.to(device)
    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels) @ (torch.ones(len(adaptation_data)).to(device))#task_weight[sample_indices[0]]
        # train_error /= len(adaptation_data)
        learner.adapt(train_error)
    #valid_error = 0
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels) @ task_weight[sample_indices[1]]
    predictions = learner(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def label_conversion(original_label):
    y_unique = torch.unique(original_label)
    original_label_2_id = {}
    for label_id in range(len(y_unique)):
        original_label_2_id[y_unique[label_id].item()] = label_id
    label_new = []
    for ele in original_label:
        label_new.append(original_label_2_id[ele.item()])
    return torch.as_tensor(label_new)


def unique_via_cpu(x, device):
    return torch.unique(x.cpu(), sorted=False).to(device)


def reweight_algo(model, opt, train_batch_list, valid_batch_list, sample_indices, task_weights, loss, device):
    eps = torch.nn.Parameter(task_weights.to(device), requires_grad=True)
    model.register_parameter("eps", eps)
    with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
        forward_loss = 0
        for i in range(len(train_batch_list)):
            adaptation_data, adaptation_labels = train_batch_list[i][0]
            #labels = label_conversion(labels)  # ADDED
            adaptation_data, adaptation_labels = adaptation_data.to(device), adaptation_labels.to(device)
            outputs_hat = fmodel(adaptation_data)
            tmp_loss = loss(outputs_hat, adaptation_labels)
            forward_loss += (tmp_loss @ fmodel.eps[sample_indices[i][0]])
        diffopt.step(forward_loss)
        backward_loss = 0
        for i in range(len(valid_batch_list)):
            support_data, support_labels = valid_batch_list[i][1]
            support_data, support_labels = support_data.to(device), support_labels.to(device)
            outputs_hat = fmodel(support_data)
            backward_loss += (loss(outputs_hat, support_labels) @ (torch.ones(len(support_data)).to(device)/len(support_data)))
        grad_all = torch.autograd.grad(backward_loss, fmodel.parameters(time=0), only_inputs=True)[0]
    return forward_loss, grad_all


def generate_no_class_imb_indices(dataset, shots, num_tasks, ways, query_points):
    indices = []
    classes = []
    for i in range(num_tasks):
        selected_classes = np.random.choice(dataset.labels, size=ways, replace=False)
        classes.append(selected_classes)
        temp_indices = []
        adaptation_indices = []
        support_indices = []
        for idx in range(ways):
            selected_indices = np.random.choice(dataset.labels_to_indices[selected_classes[idx]], size=shots[idx],
                                                replace=False)
            adaptation_indices.extend(selected_indices)
        temp_indices.append(adaptation_indices)
        for idx in range(ways):
            selected_indices = np.random.choice(list(set(dataset.labels_to_indices[selected_classes[idx]]).difference(adaptation_indices)),
                                                size=shots[idx]+query_points, replace=False)
            support_indices.extend(selected_indices)
        temp_indices.append(support_indices)
        indices.append(temp_indices)
    return classes, indices


def generate_noisy_data(dataset, noise_ratio):
    len_data = len(dataset.y)
    selected_cnt = int(noise_ratio * len_data)
    selected_lbls = np.random.choice(np.arange(len_data), size=selected_cnt, replace=False)
    noisy_lbls = np.random.choice(np.arange(64), size=selected_cnt, replace=True)
    dataset.y[selected_lbls] = noisy_lbls
    selected_lbls = list(np.unique(selected_lbls))
    return dataset, selected_lbls


def generate_no_class_imb_shots(shot, ways, num_tasks):
    shots_per_task = np.array([shot]*ways)
    shots = matlib.repmat(shots_per_task, num_tasks, 1)
    return shots


def generate_probability(num_classes, bins):
    probs_array = np.ones(num_classes)
    partitions = int(num_classes/bins)
    for i in range(bins):
        indices = np.arange(i * partitions, (i+1) * partitions)
        probs_array[indices] = 1/(i+1)
    return probs_array


def main(
        ways=args.ways,
        shots=args.shots,
        meta_lr=args.meta_lr,
        fast_lr=args.fast_lr,
        meta_batch_size=args.meta_batch_size,
        adaptation_steps=args.adaptation_steps,
        adaptation_steps_test=args.adaptation_steps_test,
        num_iterations=args.num_iterations,
        cuda=args.cuda,
        seed=args.seed,
        weight_lr=args.weight_lr,
        thres=args.thres,
        noise_frac=args.noise_ratio,
        query_num=args.query_points,):
    # flush out the arguments
    argdict = vars(args)
    print(argdict)
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # device = torch.device('cpu')
    device = torch.device('cuda')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    # Create Datasets
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='train')
    train_dataset, noisy_indices = generate_noisy_data(train_dataset, noise_frac)
    # NO NOISY DATA
    valid_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='test')
    train_dataset = MetaDataSet(train_dataset)
    valid_dataset = MetaDataSet(valid_dataset)
    test_dataset = MetaDataSet(test_dataset)
    train_shots = np.array([shots] * shots)
    valid_shots = np.array([shots] * shots)
    test_shots = np.array([shots] * shots)
    #prob_arr = generate_probability(num_classes=64, bins=4)
    train_classes, train_indices = generate_no_class_imb_indices(dataset=train_dataset, shots=train_shots,
                                                                 num_tasks=20000, ways=5, query_points=query_num)
    valid_classes, valid_indices = generate_no_class_imb_indices(dataset=valid_dataset, shots=valid_shots,
                                                                 num_tasks=2000, ways=5, query_points=query_num)
    test_classes, test_indices = generate_no_class_imb_indices(dataset=test_dataset, shots=test_shots,
                                                               num_tasks=600, ways=5, query_points=query_num)
    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    #CLUSTER_COUNT = total_shots
    # weights for each cluster
    #task_weights = torch.ones((20000, ways)) * (1 / 20000)
    weight_count = len(train_dataset.dataset.y)
    sample_wts = torch.ones(weight_count) * (1/weight_count)
    sample_wts = sample_wts.to(device)

    for iteration in range(num_iterations + 1):
        start_it = time.time()
        ##sample ID tasks and OOD tasks to a batch
        train_batch_tasks_list = []
        valid_batch_tasks_list = []
        valid_batch_classes = []
        train_batch_classes = []
        #train_batch_wts = []
        train_batch_indices = np.random.choice(np.arange(0, 20000), size=meta_batch_size, replace=False)
        for idx in train_batch_indices:
            batch = []
            train_batch_classes.append(train_classes[idx])
            #train_batch_wts.append(task_weights[train_classes[idx]])
            adaptation = []
            adaptation.append(train_dataset.dataset.x[train_indices[idx][0]].contiguous())
            adaptation.append(torch.from_numpy(train_dataset.dataset.y[train_indices[idx][0]]).contiguous().to(device))
            adaptation[1] = label_conversion(adaptation[1])
            batch.append(adaptation)
            del adaptation
            support = []
            support.append(train_dataset.dataset.x[train_indices[idx][1]].contiguous())
            support.append(torch.from_numpy(train_dataset.dataset.y[train_indices[idx][1]]).contiguous().to(device))
            support[1] = label_conversion(support[1])
            batch.append(support)
            del support
            train_batch_tasks_list.append(batch)
            #batch_clusters.append(train_shots[idx] - 1)
            #train_batch_shots.append(train_shots[idx])

        valid_batch_indices = np.random.choice(np.arange(0, 2000), size=meta_batch_size, replace=False)
        for idx in valid_batch_indices:
            valid_batch_classes.append(valid_classes[idx])
            batch = []
            adaptation = []
            adaptation.append(valid_dataset.dataset.x[valid_indices[idx][0]].contiguous())
            adaptation.append(torch.from_numpy(valid_dataset.dataset.y[valid_indices[idx][0]]).contiguous().to(device))
            adaptation[1] = label_conversion(adaptation[1])
            batch.append(adaptation)
            del adaptation
            support = []
            support.append(valid_dataset.dataset.x[valid_indices[idx][1]].contiguous())
            support.append(torch.from_numpy(valid_dataset.dataset.y[valid_indices[idx][1]]).contiguous().to(device))
            support[1] = label_conversion(support[1])
            batch.append(support)
            del support
            #valid_batch_shots.append(valid_shots[idx])
            valid_batch_tasks_list.append(batch)
        # weighted algorithm part
        maml_reweight = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        maml_reweight.load_state_dict(maml.state_dict())
        opt_rewt = optim.Adam(maml_reweight.parameters(), meta_lr)
        opt_rewt.zero_grad()
        #tmp_weights = task_weights[train_batch_indices].detach().clone()
        tmp_weights = sample_wts.detach().clone()
        train_batch_sample_indices = [train_indices[x] for x in train_batch_indices]
        valid_batch_sample_indices = [valid_indices[x] for x in valid_batch_indices]
        wt_loss, wt_grads = reweight_algo(maml_reweight, opt_rewt, train_batch_tasks_list, valid_batch_tasks_list,
                                          train_batch_sample_indices, tmp_weights, loss, device)
        grads_tensor = torch.tensor(wt_grads, device=device)
        max_grad = grads_tensor.abs().max()
        avg_grad = grads_tensor.mean()
        grads_tensor = torch.clamp(grads_tensor, min=-1 * thres, max=thres)
        #tmp_wts = torch.clamp(task_weights[train_batch_indices] - weight_lr * grads_tensor, min=0)
        tmp_wts = torch.clamp(sample_wts - weight_lr * grads_tensor, min=0)
        if (wt_loss + (torch.flatten(grads_tensor) @ torch.flatten((tmp_wts - sample_wts)))) > 0:
            sample_wts = tmp_wts
        #task_weights[train_batch_indices] = tmp_wts
        #task_weights = tmp_wts
        #task_batch_weights = task_weights[train_batch_indices]
        task_batch_weights = sample_wts
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for i in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_batch_tasks_list[i]
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               train_batch_sample_indices[i],
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               device,
                                                               task_batch_weights)
                                                               #task_batch_weights[i])
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_batch_tasks_list[i]
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               valid_batch_sample_indices[i],
                                                               learner,
                                                               loss,
                                                               adaptation_steps_test,
                                                               device,
                                                               torch.ones(len(train_dataset.dataset.y)).to(device))
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()
        print('\n')
        print('Iteration: ', iteration)
        print('Meta Train Error: ', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy: ', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error: ', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy: ', meta_valid_accuracy / meta_batch_size)
        print('Max Gradient: ', max_grad.item())
        print('Average Gradient: ', avg_grad.item())
        #print('Class Weights', task_weights.cpu().numpy())

        noisy_weights = sample_wts.detach().cpu().numpy()[noisy_indices].mean()
        non_noisy_indices = list(set(np.arange(len(train_dataset.dataset.y))).difference(set(noisy_indices)))
        non_noisy_weights = sample_wts.detach().cpu().numpy()[non_noisy_indices].mean()
        print('Noisy Weights: ', noisy_weights)
        print('Non Noisy Weights: ', non_noisy_weights)
        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)  # The operations with an underscore are inplace operations
        opt.step()
        print("time used per iter: ", time.time() - start_it, "for iter:", iteration)
        print('\n')
        NUM_TEST_POINTS = 600
        if iteration % 100 == 0:
            # test using all meta-test tasks
            acc = []
            err = []
            for t in range(NUM_TEST_POINTS):
                # Compute meta-testing loss
                learner = maml.clone()
                # batch = test_tasks.sample()
                batch = []
                adaptation = []
                adaptation.append(test_dataset.dataset.x[test_indices[t][0]].contiguous())
                adaptation.append(
                    torch.from_numpy(test_dataset.dataset.y[test_indices[t][0]]).contiguous().to(device))
                adaptation[1] = label_conversion(adaptation[1])
                batch.append(adaptation)
                del adaptation
                support = []
                support.append(test_dataset.dataset.x[test_indices[t][1]].contiguous())
                support.append(torch.from_numpy(test_dataset.dataset.y[test_indices[t][1]]).contiguous().to(device))
                support[1] = label_conversion(support[1])
                batch.append(support)
                del support
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   test_indices[t],
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps_test,
                                                                   device,
                                                                   torch.ones(len(train_dataset.dataset.y)).to(device))
                meta_test_error = evaluation_error.item()
                meta_test_accuracy = evaluation_accuracy.item()
                acc.append(meta_test_accuracy)
                err.append(meta_test_error)
            acc = np.array(acc)
            acc_mean = np.mean(acc)
            acc_95conf = 1.96 * np.std(acc) / float(np.sqrt(NUM_TEST_POINTS))
            print('Meta Test mean accu: ', acc_mean)
            result = 'acc: %.4f +- %.4f\n' % (acc_mean, acc_95conf)
            print(result)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
