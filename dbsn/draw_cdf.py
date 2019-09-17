import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

custom_lines = []

dataset = "cifar10" #"cifar100"
suffix = "" if dataset == "cifar10" else "_" + dataset
test_dirs = ["random", "ds", "ds_dp0.2", "ds_dpth0.3", "ps", "darts", "adags_lr3_con1", "adags_lr3_decayto0.5" if dataset == "cifar10" else "adags_lr3_d205"]

entropies_list = []
entropies_list.append(np.load('../work/ekfac' + suffix + '/svhn_entropies.npy'))
for i, dir in enumerate(test_dirs):
    entropies_list.append(np.load('../work/run' + dir + suffix + '_0' + '/svhn_entropies.npy'))

colores_list = ['gold', 'b', 'g', 'r', 'c', 'm', 'y', 'gray', 'k']
labels_list = ['NEK-FAC', 'Random α', 'Fixed α', 'Dropout', 'Drop-path', 'PE', 'DARTS', 'DBSN*', 'DBSN']
fig, ax = plt.subplots()

for i in range(len(entropies_list)):
    ax.hist(entropies_list[i], bins=500, normed=True, cumulative=True, label=labels_list[i],
             histtype='step', alpha=0.8, color=colores_list[i])
    custom_lines.append(Line2D([0], [0], color=colores_list[i], lw=1))

ax.grid(True)
ax.legend(custom_lines, labels_list, loc='lower right')
ax.set_title('Empirical CDF of entropy in SVHN')
ax.axis([0., 3. if dataset == "cifar100" else 1.8, 0, 1.])
#ax.set_xlabel('Annual rainfall (mm)')
#ax.set_ylabel('Likelihood of occurrence')
fix_hist_step_vertical_line_at_end(ax)
plt.savefig("entropies_" + dataset + ".pdf")
