import sys

root=sys.argv[1]
output_dir=sys.argv[2]

from haojiepaint import *

figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 2.25),
}
plt.rcParams.update(figsz)


def draw(fig, axs, nnodes, mbs, lat, layer_idx):
    import json
    
    with open(f'{root}/logs/fig13/freq100', 'r') as f:
        line = f.readlines()[0]
        data = json.loads(line)
        t0 = data['FastMoE'][1:]
        t2 = data['SmartMoE'][1:]

    L = len(t0)
    assert L == len(t2)
    
    x = np.arange(1, 1+L*10, 10)
    axs.plot(x, t0, label='static')
    
    axs.set_ylim([100,200])
    
    axs.set_xlabel('Iterations')
    axs.set_ylabel('Execution Time / ms')
    
    x = np.arange(1, 1+L*10, 10)
    axs.plot(x, t2, label='dyn. 100')
    with open(f'{root}/logs/fig13/changed_freq100', 'r') as f:
        lines = f.readlines()
        x_data = [int(v) for v in lines]
        y_data = [t2[i//10] for i in x_data]
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        axs.scatter(x_data, y_data, marker='*', color='red', zorder=999)
    
    with open(f'{root}/logs/fig13/freq10', 'r') as f:
        line = f.readlines()[0]
        data = json.loads(line)
        t0 = data['FastMoE'][1:]
        t2 = data['SmartMoE'][1:]

    L = len(t0)
    assert L == len(t2)
    
    axs.plot(x, t2, label='dyn. 10')
    with open(f'{root}/logs/fig13/changed_freq10', 'r') as f:
        lines = f.readlines()
        x_data = [int(v) for v in lines]
        y_data = [t2[i//10] for i in x_data]
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        axs.scatter(x_data, y_data, marker='*', color='red', label='change', zorder=999)
        

fig, axes = plt.subplots(1,1)
draw(fig, axes, 1, 8, 0, 1)
plt.legend(loc='upper left', ncols=4, bbox_to_anchor=(-0.01,1.2))
plt.savefig(f'{output_dir}/fig13.pdf', bbox_inches='tight')
