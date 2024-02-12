import numpy as np
import pickle, torch, math, os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

file = '21000_iteration_5W3S_svhn_oodRatio_0.6_cluster2weight.pkl'
file1 = '5W3S_svhn_oodRatio_0.6_ood_id_idx.pkl'
shot = '5-Way-3-Shot'

# file = '21000_iteration_5W5S_svhn_oodRatio_0.6_cluster2weight.pkl'
# file1 = '5W5S_svhn_oodRatio_0.6_ood_id_idx.pkl'
# shot = '5-Way-5-Shot'

ood_data = 'SVHN'
ood_level = '60'

with open(file1, 'rb') as fi:
    ood_idx, id_idx = pickle.load(fi)

#
with open(file, 'rb') as fi:
    weights = torch.load(fi, map_location='cpu')

# sum_of_weights = np.sum([num.item() for num in weights])

ood_weights = [weights[x].item() for x in ood_idx]
id_weights = [weights[x].item() for x in id_idx]


bins = np.linspace(0, math.ceil(max(weights).item()), 100)

plt.title(shot + ' ' + ood_level + '% ' + ood_data + ' OOD level', fontsize=18)
plt.ylabel('Numbers', fontsize=18)
plt.xlabel('Weights', fontsize=18)
plt.hist(ood_weights, bins, alpha=0.5, color='r', edgecolor='black', label='ood weights')
plt.hist(id_weights, bins, alpha=0.5, color='b', edgecolor='black', label='id weights')
plt.legend(loc='upper right')
plt.savefig("%s-ood%s.png" % (shot, ood_level), dpi=300, bbox_inches='tight')
plt.show()
