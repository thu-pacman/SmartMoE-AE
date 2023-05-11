import json

with open('./tmp', 'r') as f:
    line = f.readlines()[0]
    d = json.loads(line)
    print(len(d['FastMoE']))
