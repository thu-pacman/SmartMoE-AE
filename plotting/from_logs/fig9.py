import glob
import subprocess
import sys

root=sys.argv[1]
output_dir=sys.argv[2]

programs = ['tutel', 'fastmoe', 'smartmoe']
gpus = [16, 32]
models = ['gshard4.8', 'naive']

raw_data = []

for program in programs:
    my_data = []
    for gpu in gpus:
        prefix = f"{root}/logs/fig9/{gpu}gpus/{program}/"
        for model in models:
            keywords = model
            log_name = glob.glob(prefix + f'*{keywords}*')
            assert len(log_name) == 1, prefix + str(log_name)
            log_name = log_name[0]
            script = f"{root}/logs/fig9/swin_data.sh" 
            val = subprocess.run([script, log_name], capture_output=True).stdout.decode('utf-8')[:-1]
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
    colors = [color_def[2],color_def[0],color_def[3]]
    hatches = hatch_def
    L = len(names)
    width = 0.8 / L 
    
    x = np.arange(num_scale * num_gate)
    
    ds_bar = None
    
    for idx, name in enumerate(names):
        ax.bar(x - (L - 1 - 2 * idx) * width / 2, speedup[name], width, label=name, color=colors[idx], hatch=hatches[idx] if idx > 1 else None, edgecolor='black')
    
    ax.set_ylim(0, 10)
    ax.set_yticks([0,2,4,6,8,10])
    
    labels = []
    for scale in range(num_scale):
        device = 16 * (2**scale)
        for gate in range(num_gate):
            if gate == num_gate - 1:
                cap = "$+\infty$"
            else:
                cap = f"4.8"
            labels.append(f"{device}/{cap}")
    
    ax.set_xticks(x, labels)
    ax.set_ylabel('Speedup')
    ax.legend(loc='upper left', ncols=3)
    plt.savefig(f'{output_dir}/fig9.pdf', bbox_inches='tight')


num_scale = 2 # 16, 32
num_gate = 2 # gshard-4.8, naive

fig, axes = plt.subplots()

names = ['Tutel', 'FastMoE', 'SmartMoE']

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
# print(speedup['SmartMoE'])
