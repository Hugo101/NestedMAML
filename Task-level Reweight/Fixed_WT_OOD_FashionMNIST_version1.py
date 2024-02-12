#!/usr/bin/env python3
import sys

# sys.path.append("../OOD-Fixed-WT-MiniImageNet")
import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import higher
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import time
import argparse
import os
import pickle

parser = argparse.ArgumentParser(description='Learn2Learn miniImageNet Example')

parser.add_argument('--ways', type=int, default=5, metavar='N',
                    help='number of ways (default: 5)')

parser.add_argument('--shots', type=int, default=3, metavar='N',
                    help='number of shots (default: 1)')

parser.add_argument('--query_num', type=int, default=15,
                    help='number of images per class in query set')

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

parser.add_argument('--num_iterations', type=int, default=30000, metavar='N',
                    help='number of iterations (default: 1000)')

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=43, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--download_location', type=str, default="~/data", metavar='S',
                    help='download location for train data (default : ~/data')

parser.add_argument('--ood_ratio', type=float, default=0.9)

parser.add_argument('--num_tasks', type=int, default=20000)

parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--weight_lr', type=float, default=0.01)

parser.add_argument('--thres', type=float, default=0.5)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, query_num, ways, device, task_weight):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # Separate data into support/query (adaptation/evalutation) sets
    support_indices = np.zeros(data.size(0), dtype=bool)
    # selection = np.arange(ways) * (shots * 2)
    selection = np.arange(ways) * (shots + query_num)
    for offset in range(shots):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)  # this line cannot swap with the next line
    support_indices = torch.from_numpy(support_indices)
    adaptation_data, adaptation_labels = data[support_indices], labels[support_indices]
    evaluation_data, evaluation_labels = data[query_indices], labels[query_indices]
    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        #train_error /= len(adaptation_data)
        learner.adapt(train_error)
    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = task_weight * loss(predictions, evaluation_labels)
    #valid_error /= len(evaluation_data)
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

#
# def reweight_algo(model, opt, train_batch_list, valid_batch_list, task_weights, loss, shots, ways, device):
#     eps = torch.nn.Parameter(task_weights.to(device), requires_grad=True)
#     model.register_parameter("eps", eps)
#     with higher.innerloop_ctx(model, opt) as (fmodel, diffopt):
#         forward_loss = 0
#         for i in range(len(train_batch_list)):
#             data, labels = train_batch_list[i]
#             data, labels = data.to(device), labels.to(device)
#             # Separate data into adaptation/evalutation sets
#             adaptation_indices = np.zeros(data.size(0), dtype=bool)
#             selection = np.arange(ways) * (shots * 2)
#             for offset in range(shots):
#                 adaptation_indices[selection + offset] = True
#             adaptation_indices = torch.from_numpy(adaptation_indices)
#             adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
#             outputs_hat = fmodel(adaptation_data)
#             forward_loss += (loss(outputs_hat, adaptation_labels) / len(adaptation_data)) * fmodel.eps[i]
#             # forward_loss += (loss(outputs_hat, adaptation_labels)) * fmodel.eps[i]
#         diffopt.step(forward_loss)
#         backward_loss = 0
#         for i in range(len(valid_batch_list)):
#             data, labels = valid_batch_list[i]
#             data, labels = data.to(device), labels.to(device)
#             outputs_hat = fmodel(data)
#             backward_loss += (loss(outputs_hat, labels) / len(data))
#             # backward_loss += (loss(outputs_hat, labels))
#         grad_all = torch.autograd.grad(backward_loss, fmodel.parameters(time=0), only_inputs=True)[0]
#     return forward_loss, grad_all


def flatten_params(param_list):
    l = [torch.flatten(p) for p in param_list]
    flat = torch.cat(l)
    return flat


def reweight_algo_v1(model, opt, train_batch_list, valid_batch_list, task_weights, loss, shots, query_num, ways, device):

    grads_list = []
    for i in range(len(train_batch_list)):
        # Compute meta-training loss
        learner = model.clone()
        batch = train_batch_list[i]
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           1,
                                                           shots,
                                                           query_num,
                                                           ways,
                                                           device,
                                                           1)
        grads_list.append(torch.autograd.grad(evaluation_error, learner.parameters(), only_inputs=True))

    flattened_grads = [flatten_params(grad).reshape(-1, 1) for grad in grads_list]
    flattened_grads = torch.cat(flattened_grads, dim=1)

    for i in range(len(list(model.parameters()))):
        temp_grads_tensor = torch.zeros(grads_list[0][i].shape).type(torch.float).to(device)
        for x in range(len(grads_list)):
            temp_grads_tensor += (grads_list[x][i] * task_weights[x])
        with torch.no_grad():
            list(model.parameters())[i].data.sub_((opt.param_groups[0]['lr']/len(train_batch_list)) * temp_grads_tensor)

    backward_loss = 0
    for i in range(len(valid_batch_list)):
        data, labels = valid_batch_list[i]
        data, labels = data.to(device), labels.to(device)
        support_indices = np.zeros(data.size(0), dtype=bool)
        # selection = np.arange(ways) * (shots * 2)
        selection = np.arange(ways) * (shots + query_num)
        for offset in range(shots):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)  # this line cannot swap with the next line
        evaluation_data, evaluation_labels = data[query_indices], labels[query_indices]
        outputs_hat = model(evaluation_data)
        backward_loss += (loss(outputs_hat, evaluation_labels))
    grad_all = torch.autograd.grad(backward_loss, model.parameters(), only_inputs=True)
    val_grads = flatten_params(grad_all).reshape(1, -1)
    wt_grads = -1 * opt.param_groups[0]['lr'] * torch.matmul(val_grads, flattened_grads)
    return wt_grads


ID_CLASS_COUNT = 64
OOD_CLASS_COUNT = 10


def one_hot_encoder(original_label):
    one_hot_vector = np.zeros(max(ID_CLASS_COUNT, OOD_CLASS_COUNT))
    y_unique = torch.unique(original_label)
    one_hot_vector[y_unique.int()] = 1
    return one_hot_vector


def one_hot_encoder_mod(original_label, ood_label):
    one_hot_vector = np.zeros(ID_CLASS_COUNT + OOD_CLASS_COUNT)
    y_unique = torch.unique(original_label)
    if ood_label == 0:
        y_unique = y_unique + OOD_CLASS_COUNT
    one_hot_vector[y_unique.int()] = 1
    return one_hot_vector


def main(
        ways=args.ways,
        shots=args.shots,
        query_num=args.query_num,
        meta_lr=args.meta_lr,
        fast_lr=args.fast_lr,
        meta_batch_size=args.meta_batch_size,
        adaptation_steps=args.adaptation_steps,
        adaptation_steps_test=args.adaptation_steps_test,
        num_iterations=args.num_iterations,
        cuda=args.cuda,
        seed=args.seed,
        download_location=args.download_location,
        ood_ratio=args.ood_ratio,
        weight_lr=args.weight_lr,
        thres=args.thres,
):
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
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/cxl173430', mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/cxl173430', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='~/data/cxl173430', mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    ##meta-training in-of-distribution (ID) tasks
    train_transforms = [
        NWays(train_dataset, ways),
        KShots(train_dataset, shots + query_num),
        LoadData(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)

    ##meta-training OOD tasks
    transformations_ood = transforms.Compose([
        lambda x: x.convert('RGB'),
        transforms.Resize(84),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    ood_train = l2l.data.MetaDataset(FashionMNIST(download_location,
                                                  train=True,
                                                  download=True,
                                                  transform=transformations_ood))

    ood_tasks = l2l.data.TaskDataset(ood_train,
                                     task_transforms=[
                                         l2l.data.transforms.NWays(ood_train, ways),
                                         l2l.data.transforms.KShots(ood_train, shots + query_num),
                                         l2l.data.transforms.LoadData(ood_train),
                                     ],
                                     num_tasks=10000)

    ##ID meta-validation tasks
    valid_transforms = [
        NWays(valid_dataset, ways),
        KShots(valid_dataset, shots + query_num),
        LoadData(valid_dataset),
        #ConsecutiveLabels(valid_dataset),
        #RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=2000)

    ##ID meta-testing tasks
    test_transforms = [
        NWays(test_dataset, ways),
        KShots(test_dataset, shots + query_num),
        LoadData(test_dataset),
        #ConsecutiveLabels(test_dataset),
        #RemapLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=600)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')
    CLUSTER_COUNT = 200
    pkl_filename = './' + str(CLUSTER_COUNT) + '_kmeans_org_fmnist_model.pkl'
    with open(pkl_filename, 'rb') as file:
        kmeans = pickle.load(file)
    pkl_filename = './' + str(CLUSTER_COUNT) + '_org_fmnist_clusters.pkl'
    with open(pkl_filename, 'rb') as file:
        cluster_labels = pickle.load(file)
    OOD_Labels = list(set(cluster_labels[0]))
    ID_Labels = list(set(cluster_labels[1]))
    # weights for each cluster
    cluster_2_weight = torch.ones(CLUSTER_COUNT) * (1 / CLUSTER_COUNT)
    cluster_2_weight = cluster_2_weight.to(device)

    for iteration in range(num_iterations + 1):
        start_it = time.time()
        ##sample ID tasks and OOD tasks to a batch
        train_batch_tasks_list = []
        valid_batch_tasks_list = []
        batch_clusters = []
        for idx in range(meta_batch_size):
            choice = np.random.choice([0, 1], p=[1 - ood_ratio, ood_ratio])
            if choice == 1:
                batch = ood_tasks.sample()
            else:
                batch = train_tasks.sample()
            label_one_hot = one_hot_encoder_mod(batch[1], choice)
            batch_clusters.append(kmeans.predict(label_one_hot.reshape(1, OOD_CLASS_COUNT + ID_CLASS_COUNT))[0])
            batch[1] = label_conversion(batch[1])
            train_batch_tasks_list.append(batch)
            batch = valid_tasks.sample()
            batch[1] = label_conversion(batch[1])
            valid_batch_tasks_list.append(batch)
        if shots == 1:
            batch = valid_tasks.sample()
            batch[1] = label_conversion(batch[1])
            valid_batch_tasks_list.append(batch)
        # weighted algorithm part
        maml_reweight = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
        maml_reweight.load_state_dict(maml.state_dict())
        opt_rewt = optim.Adam(maml_reweight.parameters(), meta_lr)
        opt_rewt.zero_grad()
        tmp_weights = cluster_2_weight[batch_clusters].detach().clone()
        wt_grads = reweight_algo_v1(maml_reweight, opt_rewt, train_batch_tasks_list, valid_batch_tasks_list,
                                          tmp_weights, loss, shots, query_num, ways, device)
        batch_clusters_array = np.array(batch_clusters)
        grad_list = []
        cluster_ids = []
        for clust_id in set(batch_clusters):
            idx = np.where(batch_clusters_array == clust_id)[0]
            grad = torch.sum(wt_grads[0][idx])
            grad_list.append(grad)
            cluster_ids.append(clust_id)
            #cluster_2_weight[clust_id] = torch.clamp(cluster_2_weight[clust_id] - weight_lr * grad, min=0)
        grads_tensor = torch.tensor(grad_list, device=device)
        max_grad = grads_tensor.abs().max()
        avg_grad = grads_tensor.mean()
        grads_tensor = torch.clamp(grads_tensor, min= -1 * thres, max=thres)
        tmp_wts = torch.clamp(cluster_2_weight[cluster_ids] - weight_lr * grads_tensor, min=0)
        #if (wt_loss + (grads_tensor @ (tmp_wts - cluster_2_weight[cluster_ids]))) > 0:
        cluster_2_weight[cluster_ids] = tmp_wts
        task_weights = cluster_2_weight[batch_clusters]
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
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               query_num,
                                                               ways,
                                                               device,
                                                               task_weights[i])
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_batch_tasks_list[i]
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps_test,
                                                               shots,
                                                               query_num,
                                                               ways,
                                                               device,
                                                               1)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()
        ood_wts = cluster_2_weight[OOD_Labels].mean()
        id_wts = cluster_2_weight[ID_Labels].mean()

        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        print('OOD Weights', ood_wts.item())
        print('ID Weights', id_wts.item())
        print('Max Gradient', max_grad.item())
        print('Average Gradient', avg_grad.item())

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)  # The operations with an underscore are inplace operations
        opt.step()

        print("time used per iter:", time.time() - start_it, "for iter:", iteration)

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
                batch = test_tasks[t]
                batch[1] = label_conversion(batch[1])
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps_test,
                                                                   shots,
                                                                   query_num,
                                                                   ways,
                                                                   device,
                                                                   1)
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
