import sys

root=sys.argv[1]
output_dir=sys.argv[2]

from haojiepaint import *

import json
figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (6, 1.5),
}
plt.rcParams.update(figsz)
plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
plt.rcParams['axes.titlepad'] = -12  # pad is in points..


def draw(ax, nnode, mbs):
    data_file = f'{root}/logs/fig11/perf_model_{nnode}nodes_mbs{mbs}.log'

    with open(data_file, 'r') as f:
        lines = f.readlines()
        naive_est = float(lines[0])
        data = json.loads(lines[1])
        x = data['est']
        y = data['real']

    v_min = min(min(x),min(y))
    v_max = max(max(x),max(y))
    v_min = int(v_min * 0.8) // 10 * 10
    v_max = int(v_max * 1.3) // 10 * 10
    
    # from sklearn.metrics import r2_score
    # print(nnode, mbs, r2_score(y,x))
    
    x = np.array(x)
    y = np.array(y)


    ax.scatter(x, y, marker=marker_def[-1], color=color_def[7])
    
    ax.plot(np.arange(v_min,v_max), np.arange(v_min,v_max), color=color_def[4], linestyle='--')
    ax.axvline(naive_est, color=color_def[18], linestyle='-')
    
    ax.set_xlim(v_min,v_max)
    ax.set_ylim(v_min,v_max)
    
    ax.set_title(f"{nnode*8}-{mbs}")
    
    if nnode == 1 and mbs == 8:
        ax.set_xticks([0,50,100])
        ax.set_yticks([0,50,100])
    if nnode == 2:
        ax.set_xticks([50,150,250])
        ax.set_yticks([50,150,250])


fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(wspace=0.33)

draw(axes[0], 1, 4)
draw(axes[1], 1, 8)
draw(axes[2], 2, 8)
fig.supylabel('Real / ms')
fig.supxlabel('Estimation / ms',position=(0.5,-0.2))

plt.savefig(f'{output_dir}/fig11.pdf', bbox_inches='tight')
