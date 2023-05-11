import glob
import subprocess
import sys

root=sys.argv[1]
output_dir=sys.argv[2]

clusters = ['inky', 'pinky']
programs = ['fastmoe', 'deepspeed', 'fastermoe', 'smartmoe']
gpus = [[16,32], [32, 64]]
models = [['gshard1.2', 'gshard2.4', 'gshard4.8', 'naive'], ['gshard1.2', 'naive']]

raw_data = []

def is_ds(name):
    return name[:4] == 'deep'

def get_key(program, model):
    if not is_ds(program):
        return model
    else:
        if model[:6] == 'gshard':
            return 'cap-' + model[6:] + '-drop-true'
        else:
            return 'drop-false'


for program in programs:
    my_data = []
    for idx, cluster in enumerate(clusters):
        for gpu in gpus[idx]:
            prefix = f"{root}/plotting/from_exec/fig8/{program}/"
            for model in models[idx]:
                if idx == 0 and gpu == 16 and model == 'gshard1.2':
                    keywords = get_key(program, model)
                    log_name = glob.glob(prefix + f'*{keywords}*' + (".log" if not is_ds(program) else ""))
                    assert len(log_name) == 1, prefix + str(log_name)
                    log_name = log_name[0]
                    script = f"{root}/plotting/from_exec/fig8/ds_data.sh" if is_ds(program) else f"{root}/plotting/from_exec/fig8/fmoe_data.sh"

                    val = subprocess.run([script, log_name], capture_output=True).stdout.decode('utf-8')[:-1]
                else:
                    val = "-1"
                my_data.append(val)
    raw_data.append(my_data)

from haojiepaint import *

figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (14, 2),
}
plt.rcParams.update(figsz)


def draw(fig, ax, names, speedup, num_scale, num_gate):
    colors = [color_def[0],color_def[2],color_def[1],color_def[3]]
    hatches = hatch_def
    L = len(names)
    width = 0.8 / L 
    
    x = [i for i in range(num_scale * num_gate)]
    
    for i in range(8,num_scale * num_gate):
        x[i] += 0.25
    x = np.array(x)
    
    ds_bar = None
    
    for idx, name in enumerate(names):
        b = ax.bar(x - (L - 1 - 2 * idx) * width / 2, speedup[name], width, label=name, color=colors[idx], hatch=None if idx < 3 else hatches[2], edgecolor='black')
        if name == 'DeepSpeed':
            ds_bar = b
    
    ax.bar_label(ds_bar, labels=[" OOM" if v <= 0.5 else '' for v in speedup['DeepSpeed']], color='red', rotation=90)
    
    
    labels = []
    for scale in range(num_scale):
        device = 16 * (2**scale)
        for gate in range(num_gate):
            if scale < num_scale - 1:
                if gate == num_gate - 1:
                    cap = "$+\infty$"
                else:
                    cap = f"{1.2*(2**gate):.1f}"
            else:
                device = 32 if gate < 2 else 64
                if gate % 2 == 0:
                    cap = "1.2"
                else:
                    cap = "$+\infty$"
            labels.append(f"{device}/{cap}")
    ax.set_xticks(x, labels)
    ax.set_yticks([0,1,2,3])
    ax.set_xlim([-0.6,11.85])
    ax.set_ylim([0,3])
    ax.set_ylabel('Speedup')
    plt.axvline(7.625,linestyle='-', color='black')
    plt.text(9.15 + width , 3.2, 'on pinky')
    plt.text(3, 3.2, 'on inky', fontsize=12)
    plt.legend(loc='upper left', ncols=4)
    plt.savefig(f'{output_dir}/fig8.pdf', bbox_inches='tight')


num_scale = 3 # 16,32, 64
num_gate = 4 # gshard-[1.2,2.4,4.8],naive

fig, axes = plt.subplots(1,1)

names = ['FastMoE', 'DeepSpeed', 'FasterMoE', 'SmartMoE']

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
                T[name] = 10000 * T[names[0]]
            else:
                T[name] = float(raw)
        print(T)
        for name in names:
            speedup[name].append(T[names[0]] / T[name])

draw(fig, axes, names, speedup, num_scale, num_gate)
# print(speedup['SmartMoE'])