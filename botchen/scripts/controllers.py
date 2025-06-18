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

@training.cli.command('chat_test')
@click.argument('module')
@click.argument('language')
@click.argument('topk')
@click.argument('prompt_path')
@click.argument('print_statement', required=False)
def chat_test(module, language, topk, prompt_path, print_statement=False):
    conversation = ''

    gpt_models = load_gpt_models(module, modules_path, language)
    model = gpt_models['chat'][0]
    encoder = gpt_models['chat'][1]
    decoder = gpt_models['chat'][3]
    print(f">> Loaded models {gpt_models.keys()}\n\n")

    log_dir = os.path.join(dir_path, "logs", "evaluation")

    with open(prompt_path, 'r') as content_file:
        content = content_file.read()

    match = re.search(r'prompt_(.*?)\.txt$', prompt_path)
    if match:
        format = match.group(1)
    log_file_path = os.path.join(log_dir, f'chat_test_topk{topk}_format_{format}.txt')
    with open(log_file_path, 'a') as file:
        file.write(f">> Loaded models {gpt_models.keys()}\n\n")

    def process_script(script):
        nonlocal conversation
        utterances = re.findall(r'<u speaker=[^>]*>(.*?)</u>', script)

        for utterance in utterances:
            for _ in range(4):
                prompt = f"<u speaker=HUM>{utterance}</u>\n<u speaker=BOT>"

                if len(conversation.strip()) > 0:
                    conversation += f' <u speaker=HUM>{utterance}</u> <u speaker=BOT>'
                else:
                    conversation = f'<u speaker=HUM>{utterance}</u> <u speaker=BOT>'

                with torch.no_grad():
                    with ctx:
                        response, _ = generate(prompt, conversation, model, encoder, decoder, 0.9, int(topk), 64)

                if print_statement:
                    print(f">> Prompt: {prompt}\n>> Response: {response} \n\n")
                with open(log_file_path, 'a') as file:
                    log_entry = f"\n>> Prompt: {prompt}\n>> Response: {response} \n"
                    file.write(log_entry)

    process_script(content)