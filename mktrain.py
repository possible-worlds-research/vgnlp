import sys
import re
from random import shuffle

def read_qa(fpath):
    typ = 'TLK'
    activities = {}
    with open(fpath, encoding='utf-8') as fin:
        for l in fin:
            if l.startswith('<a type'):
                m = re.search('idx=([^>]*)',l)
                idx = m.group(1)
                activities[idx] = f'<a type={typ}>\n'
            else:
                activities[idx]+=l
    return activities

def read_obs(fpath, object_memory=10000):
    typ = 'OBS'
    activities = {}
    memory = []
    idx = 0
    with open(fpath, encoding='utf-8') as fin:
        for l in fin:
            if l.startswith('<a type'):
                if len(memory) > 0:
                    shuffle(memory)
                    memory = memory[:object_memory]
                    activities[idx] += ''.join(memory)
                    activities[idx] += '</a>\n'
                    memory.clear()
                m = re.search('idx=([^>]*)',l)
                idx = m.group(1)
                activities[idx] = f'<a type={typ}>\n'
            if l.startswith('<e>'):
                memory.append(l.lower())
    return activities

def sample_from_dict(d, n):
    keys = list(d.keys())
    shuffle(keys)
    keys = keys[:n]
    new_d = {}
    for key in keys:
        new_d[key] = d[key]
    return new_d


memory = int(sys.argv[1])
nsamples = int(sys.argv[2])

qas = read_qa('skt/question_answers.json.skt')
obs = read_obs('obs/region_descriptions.json.obs',memory)
sample = sample_from_dict(qas,nsamples)

with open('sample_vg_qa.txt', 'w', encoding='utf-8') as tf:
    for k,v in obs.items():
        if k in sample:
            tf.write(v)
            tf.write(sample[k])
