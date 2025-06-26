from os.path import dirname, realpath, join
from random import shuffle

'''
BEGINNING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE
'''

configs = {}

def mk_idx_list():
    parent_dir = dirname(dirname(realpath(__file__)))
    ideallanguage = join(parent_dir, "data", "ideallanguage.txt")

    ids = []
    with open(ideallanguage, 'r', encoding='utf-8') as fin:
        for l in fin:
            if l.startswith('<situation id='):
                l = l.rstrip('\n').strip()
                idx = int(l.replace('<situation id=','')[:-1])
                ids.append(idx)

    n = 2000
    print(f"Found {len(ids)} situations. Picking a random {n}...")
    shuffle(ids)
    ids = ids[:n]
    for i, idx in enumerate(ids):
        ids[i] = (idx,i+1)
    return ids

configs['ids'] = mk_idx_list()
substitution_term_list = []
configs['extend_corpus'] = False
configs['apply_permutations'] = False  # This applies the permutations and writes the files 
configs['training_and_test_sets'] = True 

configs['limited'] = False
configs['limited_max_utterances'] = 5 # These make the situation be of x utterances
configs['test_mode'] = False
configs['test_max_situations'] = 3 # These make the x situations from which we extract

configs['min_referent_overlap_ratio'] = 0.7 # FOCUSED ON REFERENT SITUATION Minimum proportion of referent entities that must appear in a target situation (i.e. we apply this to referent situations, e.g. *1 if the referent situation is as such)
configs['min_target_overlap_ratio'] = 0.1 # FOCUSED ON TARGET SITUATION  Minimum proportion of target entities that must match referent entities (i.e. we apply this to all the *10 situations which we are finding similar to a referent situation *1)
configs['min_content_length'] = 1000 # Minimum number of characters in a situation's content
configs['max_content_length'] = 200000 # Maximum number of characters in a situation's content
configs['max_per_referent'] = 10 # Maximum number of similar situations to extract per referent situation (e.g. we take *10* situations similar to situation 1, *10* to situation 2)
configs['train_test_ratio'] = 0.7 # Percentage of training and testing sets
