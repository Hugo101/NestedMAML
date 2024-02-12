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
from torchvision.datasets import SVHN
import higher
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
import time
import argparse
import os
import pickle


parser = argparse.ArgumentParser(description='Learn2Learn miniImageNet Example')

parser.add_argument('--ood_dataset', type=str, default="fMNIST", metavar='S',
                    help='OOD dataset (default : fMNIST') #fMNIST, SVHN

parser.add_argument('--ways', type=int, default=5, metavar='N',
                    help='number of ways (default: 5)')

parser.add_argument('--shots', type=int, default=1, metavar='N',
                    help='number of shots (default: 1)')

parser.add_argument('--query_num', type=int, default=15,
                    help='number of images per class in query set')

parser.add_argument('--query_num_val', type=int, default=15,
                    help='number of images per class in query set of validation task')

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

parser.add_argument('--download_location', type=str, default="~/data/cxl173430", metavar='S',
                    help='download location for train data (default : ~/data')

parser.add_argument('--ood_ratio', type=float, default=0.9)

parser.add_argument('--num_tasks', type=int, default=20000)

parser.add_argument('--gpu_id', type=int, default=7)

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
        ood_data  = args.ood_dataset,
        ways      = args.ways,
        shots     = args.shots,
        query_num = args.query_num,
        query_num_val = args.query_num_val,
        meta_lr   = args.meta_lr,
        fast_lr   = args.fast_lr,
        meta_batch_size  = args.meta_batch_size,
        adaptation_steps = args.adaptation_steps,
        adaptation_steps_test = args.adaptation_steps_test,
        num_iterations = args.num_iterations,
        cuda = args.cuda,
        seed = args.seed,
        download_location = args.download_location,
        ood_ratio = args.ood_ratio,
        weight_lr = args.weight_lr,
        thres = args.thres,
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
    train_dataset = l2l.vision.datasets.MiniImagenet(root=download_location, mode='train')
    valid_dataset = l2l.vision.datasets.MiniImagenet(root=download_location, mode='validation')
    test_dataset  = l2l.vision.datasets.MiniImagenet(root=download_location, mode='test')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset  = l2l.data.MetaDataset(test_dataset)

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

    if ood_data == 'fMNIST':
        ood_train = l2l.data.MetaDataset(FashionMNIST(download_location,
                                                      train=True,
                                                      download=True,
                                                      transform=transformations_ood))
    if ood_data == 'SVHN':
        ood_train = l2l.data.MetaDataset(SVHN(download_location,
                                              split='train',
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
        KShots(valid_dataset, shots + query_num_val),
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

    for iteration in range(num_iterations + 1):
        start_it = time.time()
        ##sample ID tasks and OOD tasks to a batch
        train_batch_tasks_list = []
        valid_batch_tasks_list = []

        for idx in range(meta_batch_size):
            choice = np.random.choice([0, 1], p=[1 - ood_ratio, ood_ratio])
            if choice == 1:
                batch = ood_tasks.sample()
            else:
                batch = train_tasks.sample()
            batch[1] = label_conversion(batch[1])
            train_batch_tasks_list.append(batch)
            batch = valid_tasks.sample()
            batch[1] = label_conversion(batch[1])
            valid_batch_tasks_list.append(batch)
        if shots == 1:
            batch = valid_tasks.sample()
            batch[1] = label_conversion(batch[1])
            valid_batch_tasks_list.append(batch)

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
                                                               1)
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

        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

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
