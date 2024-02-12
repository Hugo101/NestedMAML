import matplotlib.pyplot as plt
import numpy as np
from _collections import OrderedDict, defaultdict
import os, sys
sys.path.append("../../metaL-dss")
from sine_wave_outlier_regression.maml_rm_ood_synthetic_data import *
from sine_wave_outlier_regression.maml_synthetic_data import *
from sine_wave_outlier_regression.maml_synthetic_fixed_weight import *
from sine_wave_outlier_regression.maml_synthetic_reweight import *
from sine_wave_outlier_regression.heuristic_synthetic_data import *
from Sine_Regression.OOD_Tasks.tasks import *
import torch
import learn2learn as l2r

def model_functions_at_training(initial_model, X, y, sampled_steps, x_axis, optim=torch.optim.Adam, lr=0.01):
    model = SyntheticMAMLModel()
    model.cuda()
    model.load_state_dict(initial_model.module.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), lr)
    num_steps = max(sampled_steps)
    K = X.shape[0]
    losses = []
    outputs = {}
    for step in range(1, num_steps + 1):
        loss = criterion(model(X), y) / K
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimiser.step()
        if step in sampled_steps:
            outputs[step] = model(torch.tensor(x_axis).to(torch.device('cuda'), dtype=torch.float).view(-1, 1)).detach().cpu().numpy()
    outputs['initial'] = initial_model(torch.tensor(x_axis).to(torch.device('cuda'), dtype=torch.float).view(-1, 1)).detach().cpu().numpy()

    return outputs, losses


def plot_sampled_performance(model1, model1_name, method1,
                             model2, model2_name, method2,
                             model3, model3_name, method3,
                             model4, model4_name, method4,
                             #model5, model5_name, method5,
                             task, X, y, x_min, x_max, sampled_steps, ood_lv, k,
                             optim=torch.optim.Adam, lr=0.01):
    x_axis = np.linspace(x_min, x_max, 1000)
    outputs1, losses1 = model_functions_at_training(model1, X, y, sampled_steps, x_axis, optim, lr)
    outputs2, losses2 = model_functions_at_training(model2, X, y, sampled_steps, x_axis, optim, lr)
    outputs3, losses3 = model_functions_at_training(model3, X, y, sampled_steps, x_axis, optim, lr)
    outputs4, losses4 = model_functions_at_training(model4, X, y, sampled_steps, x_axis, optim, lr)
    #outputs5, losses5 = model_functions_at_training(model5, X, y, sampled_steps, x_axis, optim, lr)

    # plot the first figure -- sine waves
    plt.figure(1)
    plt.scatter(X.cpu().numpy(), y.cpu().numpy(), marker='*', color='purple', s=120)
    plt.plot(x_axis, task.true_function(x_axis), '-', color='r', label='ground truth')
    for step in sampled_steps:
        plt.plot(x_axis, outputs1[step], ':', color='b', label=model1_name)     # MAML
        plt.plot(x_axis, outputs2[step], '.', color='g', label=model2_name)     # MAML removed OOD
        plt.plot(x_axis, outputs3[step], '^', color='c', label=model3_name)     # Heuristic
        plt.plot(x_axis, outputs4[step], '-', color='k', label=model4_name)     # Reweighted MAML
        #plt.plot(x_axis, outputs5[step], 'v', color='y', label=model5_name)  # weighted MAML (ours)
    plt.legend(loc='best')
    plt.title("OOD Task %s, K=%s" % (ood_lv, k))
    plt.savefig("K=%s_OOD_Task_%s_Sine.png" % (k, ood_lv), dpi=300, bbox_inches='tight')

    with open('final_losses_'+str(K)+'_'+str(ood_lv)+'.txt','w') as txt_file:
        txt_file.write(model1_name+": " +str(losses1))
        txt_file.write(model2_name+": " + str(losses2))
        txt_file.write(model3_name + ": " + str(losses3))
        txt_file.write(model4_name + ": " + str(losses4))

    # plot the second figure -- val losses
    plt.figure(2)
    plt.title("Test Losses Over Time, K=%s" % (k))
    plt.plot(method1.test_losses, color='b', label=model1_name)     # MAML
    plt.plot(method2.test_losses, color='g', label=model2_name)     # MAML removed OOD
    plt.plot(method3.test_losses, color='c', label=model3_name)     # Heuristic
    plt.plot(method4.test_losses, color='k', label=model4_name)
    #plt.plot(method5.test_losses, color='y', label=model5_name)     # weighted MAML (ours)
    plt.legend(loc='best')
    plt.savefig("K=%s_OOD_Task_%s_ValLosses.png" % (k, ood_lv), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # hyperparameters
    K = [5]
    ############################################################
    # MAML
    task_ood_lv = [0.3]
    # Sine Waves
    amplitude_min = 0.1
    amplitude_max = 5
    phase_min = 0
    phase_max = np.pi
    x_min = -5
    x_max = 5

    # plotting
    sampled_steps = [10]
    ############################################################

    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)

    tasks = Sine_Task_Distribution(amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max)
    for k in K:
        for ood_lv in task_ood_lv:
            maml = MAMLSinewave(ood_lv, shots=k, fas=1, num_iterations=5000)
            maml.main()
            maml_rm_ood = MAMLSinewaveRmOOD(ood_lv, shots=k, fas=1, num_iterations=5000)
            maml_rm_ood.main()
            re_wt_maml = ReWeightSinewave(ood_lv, shots=k, fas=1, num_iterations=5000)
            re_wt_maml.main()
            fixed_wt_maml = FixedWeightSinewave(ood_lv, shots=k, fas=1, num_iterations=5000)
            fixed_wt_maml.main()
            #heuristic_maml = HeuristicSinewave(ood_lv, shots=k, fas=1, num_iterations=5000)
            #heuristic_maml.main()
            task = Sine_Task_Distribution(4, 4, phase_min, phase_max, x_min, x_max).sample_task()
            X, y = task.sample_data(k)
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))
            plot_sampled_performance(
                maml.meta_model, 'MAML', maml,
                maml_rm_ood.meta_model, 'MAML (OOD removed)', maml_rm_ood,
                #heuristic_maml.meta_model, 'Heuristic', heuristic_maml,
                re_wt_maml.meta_model, 'Weighted MAML', re_wt_maml,
                fixed_wt_maml.meta_model, 'Fixed Weight MAML', fixed_wt_maml,
                task, X, y, x_min, x_max, sampled_steps, ood_lv, k)
