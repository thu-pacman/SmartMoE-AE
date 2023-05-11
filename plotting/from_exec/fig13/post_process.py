import sys
import glob
import json
import subprocess

def get_file(path, keywords):
    files = glob.glob(f"{path}/*{keywords}*")
    assert len(files) == 1, f"{path} {keywords} {files}"
    return files[0]

def process(log, dump, layer_idx):
    data0 = []
    data1 = []

    t_rounds = []
    with open(log, "r") as f:
        for line in f.readlines():
            if line[:6] == 'test d':
                data = line.split(' ')
                iter = int(data[12])
                round = int(data[16])
                layer = int(data[14].split(':')[0])
                if layer != layer_idx:
                    continue
                vals = data[17:22]
                t = {}
                for val in vals:
                    name, t0 = val.split('=')
                    t0 = float(t0)
                    t[name] = t0
                if round == 0:
                    t_rounds = []
                t_rounds.append(t)
                if round != 3:
                    continue

                data0.append(min([t['FastMoE']*1000 for t in t_rounds]))
                data1.append(min([t['SmartMoE']*1000 for t in t_rounds]))

                
    with open(dump, 'w') as f:
        out = json.dumps({'FastMoE':data0, 'SmartMoE':data1})
        print(out, file=f)

if __name__ == '__main__':
    log_dir = sys.argv[1]
    freq10_log = get_file(log_dir, 'FREQ10_')
    freq100_log = get_file(log_dir, 'FREQ100_')

    layer_idx = 6
    process(freq10_log, './freq10.log', layer_idx)
    process(freq100_log, './freq100.log', layer_idx)

    subprocess.run(['./star.sh', freq10_log, './freq10_star.log'])
    subprocess.run(['./star.sh', freq100_log, './freq100_star.log'])