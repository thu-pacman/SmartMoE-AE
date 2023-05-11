import subprocess
import glob
import sys

root=sys.argv[1]
output_dir=sys.argv[2]

prefix = f'{root}/plotting/from_exec/fig12'

static_log = glob.glob(f"{prefix}/*dynamicOFF*.log")
assert len(static_log) == 1, static_log
static_log = static_log[0][:-4]

dynamic_log = glob.glob(f"{prefix}/*dynamicON*.log")
assert len(dynamic_log) == 1
dynamic_log = dynamic_log[0][:-4]

script = f'{root}/plotting/from_exec/fig12/calc.sh'

static_T = subprocess.run([script, static_log], capture_output=True).stdout.decode('utf-8').split('\n')[:-1]
dynamic_T = subprocess.run([script, dynamic_log], capture_output=True).stdout.decode('utf-8').split('\n')[:-1]

from haojiepaint import *

figsz = {
    'axes.labelsize': 13,
    'font.size': 13,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (6, 1.5),
}
plt.rcParams.update(figsz)


def draw_layer(fig, ax, names, speedup, num_layer):
    colors = color_def[1:]
    hatches = hatch_def
    L = len(names)
    width = 0.8 / L 
    
    x = np.arange(num_layer)
    
    ds_bar = None
    
    for idx, name in enumerate(names):
        b = ax.bar(x - (L - 1 - 2 * idx) * width / 2, speedup[name], width, label=name, color=colors[idx], hatch=hatches[idx] if idx > 0 else None, edgecolor='black')
    
    ax.set_xticks([])
    ax.set_ylim([0,2])
    ax.set_yticks([0,0.5,1,1.5,2])
    ax.set_ylabel('Speedup')
    ax.set_xlabel('# layers')
    ax.legend(loc='upper right',ncols=2)

fig, ax = plt.subplots(1,1)

names = ['Static', 'Adaptive']

speedup = {}
for name in names:
    speedup[name] = []


static_L = []
dynamic_L = []
for l in range(16):
    t1 = float(dynamic_T[l])
    t2 = float(static_T[l])
    dynamic_L.append(t2/t1)
    static_L.append(1)

datas = {'Static':static_L, 'Dynamic':dynamic_L}
# print(dynamic_L)
draw_layer(fig, ax, ['Static', 'Dynamic'], datas, 16)

plt.savefig(f'{output_dir}/fig12.pdf', bbox_inches='tight')
