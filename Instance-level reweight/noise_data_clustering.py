import sys
sys.path.append("../OOD-Fixed-WT-MiniImageNet")
import random
import numpy as np
import torch
from torchvision import transforms, models
import pretrainedmodels
import learn2learn as l2l
import time
import argparse
import os
import pickle
from _collections import defaultdict
from sklearn.cluster import KMeans
from torch.utils.data import random_split, SequentialSampler, BatchSampler, RandomSampler

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
parser.add_argument('--query_points', type=int, default=5, metavar='S',
                    help='Query points for evaluation data')
parser.add_argument('--noise_ratio', type=float, default=0.5, help='Noise Ratio')
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--weight_lr', type=float, default=0.01)
parser.add_argument('--thres', type=float, default=10)
parser.add_argument('--total_shots', type=int, default=25, help="Total no of shots")
parser.add_argument('--shots', type=int, default=1, help='Shots per each class')
parser.add_argument('--weight_init', type=float, default=0.005, help='Weight Initializaion')
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

def generate_noisy_data(dataset, noise_ratio):
    len_data = len(dataset.y)
    selected_cnt = int(noise_ratio * len_data)
    selected_lbls = np.random.choice(np.arange(len_data), size=selected_cnt, replace=False)
    noisy_lbls = np.random.choice(np.arange(64), size=selected_cnt, replace=True)
    dataset.y[selected_lbls] = noisy_lbls
    selected_lbls = list(np.unique(selected_lbls))
    return dataset, selected_lbls

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
        query_num=args.query_points,
        weight_init=args.weight_init,):
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
    train_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='train', transform=transforms.compose([transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]))
    train_dataset, noisy_indices = generate_noisy_data(train_dataset, noise_frac)
    model = models.vgg19(pretrained=True)
    #model_name = 'vgg19'
    #model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').cuda()
    dataset = torch.utils.data.TensorDataset(train_dataset.x.type(torch.float).contiguous(), torch.from_numpy(train_dataset.y).type(torch.long).contiguous())
    batchwise_indices = list(BatchSampler(SequentialSampler(dataset), 20, drop_last=False))
    model.eval()
    for idx in batchwise_indices:
        embeddings = model.features(train_dataset.x[idx].contiguous())
    print()
    # NO NOISY DATA
    #valid_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='validation')
    #test_dataset = l2l.vision.datasets.MiniImagenet(root='~/data', mode='test')

    #train_dataset = MetaDataSet(train_dataset)
    #valid_dataset = MetaDataSet(valid_dataset)
    #test_dataset = MetaDataSet(test_dataset)
    #train_shots = np.array([shots] * ways)
    #valid_shots = np.array([shots] * ways)
    #test_shots = np.array([shots] * ways)

if __name__ == '__main__':
    main()
