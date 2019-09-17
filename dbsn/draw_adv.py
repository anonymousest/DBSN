import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

custom_lines = []

dataset = "cifar10" #'cifar100'
adv_method = "fgsm" #fgsm
suffix = "" if dataset == "cifar10" else "_" + dataset
test_dirs = ["random", "ds", "ds_dp0.2", "ds_dpth0.3", "ps", "darts", "adags_lr3_con1", "adags_lr3_decayto0.5" if dataset == "cifar10" else "adags_lr3_d205", ]

eps_list = []
acc_list = []
ent_list = []
tmp1, tmp2, tmp3 = [], [], []
for st in open('../work/ekfac' + suffix + '/log_attack_' + adv_method + '.txt').read().split("\n")[-12:-1]:
    tmp3.append(float(st.split()[5]))
    tmp1.append(1. - float(st.split()[13][1:-2])/100.)
    tmp2.append(float(st.split()[10][:-1]))
for j in range(len(tmp3)):
    assert(tmp3[j] == j/100.)
print(tmp1, tmp2)
acc_list.append(tmp1)
ent_list.append(tmp2)
for i, dir in enumerate(test_dirs):
    tmp1, tmp2, tmp3 = [], [], []
    for st in open('../work/run' + dir + suffix + '_0' + '/log_attack_' + adv_method + '.txt').read().split("\n")[-12:-1]:
        tmp3.append(float(st.split()[5]))
        tmp1.append(1. - float(st.split()[13][1:-2])/100.)
        tmp2.append(float(st.split()[10][:-1]))
    for j in range(len(tmp3)):
        assert(tmp3[j] == j/100.)
    print(tmp1, tmp2)
    acc_list.append(tmp1)
    ent_list.append(tmp2)

colores_list = ['gold', 'b', 'g', 'r', 'c', 'm', 'y', 'gray', 'k']
labels_list = ['NEK-FAC', 'Random α', 'Fixed α', 'Dropout', 'Drop-path', 'PE', 'DARTS', 'DBSN*', 'DBSN']
fig, ax = plt.subplots()

for i in range(len(acc_list)):
    ax.plot(tmp3, acc_list[i], label=labels_list[i], color=colores_list[i])
    ax.scatter(tmp3, acc_list[i], color=colores_list[i], s=16.)
    custom_lines.append(Line2D([0], [0], marker = 'o', markersize=4, linestyle='--', color=colores_list[i], lw=1))

ax.set_xlabel('Perturbation size')
ax.set_ylabel('Accuracy', color='k')
ax.tick_params('y', colors='k')
ax.grid(True)
ax.legend(custom_lines, labels_list, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0,
          fancybox=True, shadow=False, ncol=5)

ax.axis([0., max(tmp3), 0, 1.])
ax1 = ax.twinx()
for i in range(len(ent_list)):
    ax1.plot(tmp3, ent_list[i], label=labels_list[i], color=colores_list[i], linestyle='--')
    ax1.scatter(tmp3, ent_list[i], color=colores_list[i], s=16.)

ax1.set_ylabel('Entropy', color='k')
ax1.tick_params('y', colors='k')

fig.tight_layout()
plt.savefig("adv_" + dataset + '_' + adv_method + ".pdf")
