import glob
import subprocess
import sys

root=sys.argv[1]
output_dir=sys.argv[2]

programs = ['fastmoe', 'alpa', 'smartmoe']
gpus = [16, 32]
models = ['gshard1.2', 'naive']

raw_data = []

for program in programs:
    my_data = []
    for gpu in gpus:
        prefix = f"{root}/plotting/from_exec/fig10/{program}/"
        for model in models:
            keywords = model
            log_name = glob.glob(prefix + f'*{keywords}*.log')
            if len(log_name) == 1 and gpu == 16:
                script = f"{root}/plotting/from_exec/fig10/fmoe_data.sh" 
                val = subprocess.run([script, log_name[0]], capture_output=True).stdout.decode('utf-8')[:-1]
            else:
                val = "-1"
            my_data.append(val)
    raw_data.append(my_data)

from haojiepaint import *

figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (6, 2.5),
}
plt.rcParams.update(figsz)


def draw(fig, ax, names, speedup, num_scale, num_gate):
    colors = [color_def[0],color_def[2],color_def[3]]
    hatches = hatch_def
    L = len(names)
    width = 0.8 / L 
    
    x = np.arange(num_scale * num_gate)
    
    ds_bar = None
    
    for idx, name in enumerate(names):
        b = ax.bar(x - (L - 1 - 2 * idx) * width / 2, speedup[name], width, label=name, color=colors[idx], hatch=hatches[idx] if idx > 1 else None,edgecolor='black')
    
    labels = []
    for scale in range(num_scale):
        device = 16 * (2**scale)
        for gate in range(num_gate):
            if gate == num_gate - 1:
                cap = "$+\infty$"
            else:
                cap = f"{1.2*(2**gate):.1f}"
            labels.append(f"{device}/{cap}")
    
    ax.set_xticks(x, labels)
    ax.set_ylim([0,3])
    ax.set_xticks([0,1,2,3])
    ax.set_ylabel('Speedup')
    plt.legend(loc='upper left',ncols=3)
    plt.savefig(f'{output_dir}/fig10.pdf',bbox_inches='tight')


num_scale = 2 # 16,32
num_gate = 2 # gshard-1.2,naive

fig, axes = plt.subplots(1,1)

names = ['FastMoE', 'Alpa', 'SmartMoE']

speedup = {}
for name in names:
    speedup[name] = []

for scale in range(num_scale):
    for gate in range(num_gate):
        start_col = gate + scale * num_gate
        T = {}
        for idx, name in enumerate(names):
            raw = raw_data[idx][start_col]
            if raw[0] >= 'A' and raw[0] <= 'Z':
                T[name] = 2 * T[names[0]]
            else:
                T[name] = float(raw)

        for name in names:
            speedup[name].append(T[names[0]] / T[name])

draw(fig, axes, names, speedup, num_scale, num_gate)
# print(speedup['Alpa'])
# print(speedup['SmartMoE'])
