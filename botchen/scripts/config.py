
'''
BEGINNING DYNAMICAL PARAMETERS WHICH THE USER CAN CHANGE
'''

# These are the mapping ids from the ideallangueg/visualGenome to the new ids
ids= [
        (2317468, 1), # person
        (2396154, 2), # animal
        (186, 3), # food
        (2410753, 4), # desktop 
        (1, 5), # street
        (4, 6), # bedroom

        (2412620, 7), # person
        (2412211, 8), # animal
        (713137, 9), # food
        (3, 10), # desktop
        (2357183, 11), # street
        (9, 12), # bedroom/room

        (2343232, 13), # person
        (2406947, 14), # animal
        (2343284, 15), # food
        (1515, 16), # desktop
        (2343307, 17), # street
        (2361685, 18) # bedroom/room
        ]

substitution_terms_list = [
        'car','jacket','shirt', 'man','woman', 'tree','road', 'bicycle', 
        'gym_shoe','table', 'curtain', 'sofa', 'chair', 'picture', 'teddy', 
        'desk','jean','room', 'ceiling', 'shelf', 'picture', 'monitor', 'bottle', 
        'sunset', 'mouse', 'part','cup', 'egg', 'muffin', 'plate', 'tomato', 
        'sauce', 'tea', 'spoon','mouth', 'watch','giraffe', 'branch', 
        'neck', 'eye','basket', 'ginger', 'vegetable', 'bowl', 'cheese', 
        'chopstick','grass', 'elephant', 'trunk','suit', 'belt', 'hair', 'earring'
        ]

extend_corpus = True
permutation_flag = True  # This applies the permutations and writes the files 
training_and_test_sets = True 

write_all_files = False # This makes it write files of augmented and original

limited = False
limited_max_utterances = 5 # These make the situation be of x utterances
test_mode = False
test_max_situations = 3 # These make the x situations from which we extract

if extend_corpus:
    min_referent_overlap_ratio=0.7 # FOCUSED ON REFERENT SITUATION Minimum proportion of referent entities that must appear in a target situation (i.e. we apply this to referent situations, e.g. *1 if the referent situation is as such)
    min_target_overlap_ratio=0.1 # FOCUSED ON TARGET SITUATION  Minimum proportion of target entities that must match referent entities (i.e. we apply this to all the *10 situations which we are finding similar to a referent situation *1)
    min_content_length=1000 # Minimum number of characters in a situation's content
    max_content_length=200000 # Maximum number of characters in a situation's content
    max_per_referent=10 # Maximum number of similar situations to extract per referent situation (e.g. we take *10* situations similar to situation 1, *10* to situation 2)
    train_split_ratio=0.7 # Percentage of training and testing sets
