# SPDX-FileCopyrightText: 2024 Denotation UG,
#
# SPDX-License-Identifier: AGPL-3.0-only
from os.path import dirname, realpath, join
from glob import glob
from flask import Blueprint
import click
import torch

from app import app
from app.cli.prepare import preprocess
from app.cli.statlm import train_stat_lm
from app.cli.train import run_training, ctx
from app.loaders import load_gpt_models
from app.chat.converse import generate
from app.utils import load_models_and_hyperparameters
from app import log_path
from datetime import datetime
from flask import session
from app.utils import split_by_space
from app.chat.memory_processor import write_to_short_term_memory

torch.set_num_threads(1)
training = Blueprint('training', __name__)
dir_path = dirname(dirname(dirname(realpath(__file__))))
modules_path = join(dir_path, 'app', 'modules')
data_path = join(dir_path, 'data')

@training.cli.command('run')
@click.argument('module')
@click.argument('language')
@click.argument('vocab_size')
def train(module, language, vocab_size):
    data_dir = join(data_path, module, language)
    module_dir = join(modules_path, module, language)
    preprocess(data_dir, module_dir, vocab_size)
    train_stat_lm(data_dir, module_dir)
    run_training(data_dir, module_dir)

@training.cli.command('eval')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
def chat(module, language, topk):
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")
    prompt = ''
    conversation = ''
    while True:
        prompt = input('>> ').rstrip('\n')
        if prompt == 'q':
            break
        if len(conversation.strip()) > 0:
            conversation = conversation + ' <u speaker=HUM>'+prompt+'</u> <u speaker=BOT>'
        else:
            conversation = '<u speaker=HUM>'+prompt+'</u> <u speaker=BOT>'
        with torch.no_grad():
            with ctx:
                response, _ = generate(prompt, conversation, model, encoder, decoder, 0.9, int(topk), 64)
                print('>> '+response)
                conversation = conversation + response + '</u>'
    conversation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/'))
    log_file = os.path.join(log_dir, f'user-{conversation_time}.txt')
    with open(log_file,'a') as fout:
        #print(split_by_space(conversation)[-2:])
        fout.write('\n'.join(split_by_space(conversation)[-2:])+'\n')

####################################################

from datetime import datetime
import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# Command to initiate evaluation with vectorial space in a CLI
@training.cli.command('evaluation_with_vectorial_space')
@click.argument('module')
@click.argument('language')
@click.argument('topk') # Top K responses to consider from the model
@click.argument('content_path') # Path to the content file that contains the scripts
@click.argument('situation_id', type=int) # ID of the specific conversation script to evaluate
@click.argument('print_statement', required=False)  # Optional argument to print statements live
@click.argument('log_file', required=False) # Optional argument to log output to a file
def make_evaluating_conversation(module, language, topk,
                                 content_path, situation_id,
                                 log_file = False, # If we want to store
                                 print_statement = False): # If we want to look live at the conversation
    conversation = '' # Initialize an empty string to store the conversation
    # Load the specified GPT models (encoder and decoder) based on the provided module and language
    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")
    # If log_file argument is provided, set up logging to a specific file
    if log_file:
        #conversation_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs/evaluation/'))
        log_file = os.path.join(log_dir, f'evaluation_situation{situation_id}.txt')
        with open(log_file, 'a') as file:
            file.write(f">> Loaded models {gpt_models.keys()}\n\n")
            
    # Read the content file that contains conversation scripts
    with open(content_path,'r') as content_file:
        content=content_file.read()
    # Use a regular expression to find the script for the specific situation_id
    situation_pattern = rf"<script\.{situation_id} type=CONV>.*?</script\.{situation_id}>"
    script = re.findall(situation_pattern, content, re.DOTALL) # Search for the script block
    if not script:
        print(f"No situation found with id={situation_id}")
        return None
    script = script[0]
    print(script)
    
    # Loop through each utterance in the script
    utterances = re.findall(r'<u speaker=[^>]*>\((.*?)\)</u>', script)
    for utterance in utterances:
        # Generate a response for each utterance up to 4 times
        for _ in range(4):
            if len(conversation.strip()) > 0:
                # If conversation already has content, append the next utterance to it
                conversation += f' <u speaker=HUM>{utterance}</u> <u speaker=BOT>'
            else:
                conversation = f'<u speaker=HUM>{utterance}</u> <u speaker=BOT>'
            # Generate a response from the model using the conversation context
            with torch.no_grad():
                with ctx:
                    response, _ = generate(utterance, conversation, model, encoder, decoder, 0.9, int(topk), 64)
            if print_statement:
                print(f">> Prompt: {utterance}\n>> Response: {response} \n\n")
            if log_file:
                with open(log_file, 'a') as file:
                    log_entry = f"\n>> Prompt: {utterance}\n>> Response: {response} \n"
                    file.write(log_entry)