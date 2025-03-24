import io
import json
import ijson
import requests
import zipfile
from tqdm import tqdm
from os.path import join
from pathlib import Path

host_url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/"

def download_vg(fpath):
    print(f">> DATA: VISUAL GENOME DOWNLOADING: {fpath}")
    response = requests.get(join(host_url, fpath), stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(fpath, mode="wb") as f:
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
    print(f">> DATA: VISUAL GENOME DOWNLOADING: {fpath} successfully downloaded.\n")


def extract_rd_for_training(fpath, activity=None):
    if activity not in ['obs']:
        print(f">> DATA: VISUAL GENOME EXTRACTION: ERROR: activity type {activity} not supported.")
        return 0

    print(f">> DATA: VISUAL GENOME EXTRACTING: {fpath} for training ({activity})")
    Path(activity).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(fpath) as zf:
        with io.TextIOWrapper(zf.open(fpath.replace('.zip','')), encoding="utf-8") as f:
         content = json.load(f)

    with open(join(activity,fpath.replace('.zip','.'+activity)), 'w', encoding="utf-8") as f:
        c = 0
        for img in content:
            idx = int(img['id'])
            if idx != c:
                if c != 0:
                    f.write(f"</a>\n")
                c = idx
                f.write(f"<a type=OBS idx={idx}>\n")
            for region in img['regions']:
                phrase = region['phrase'].rstrip('\n')
                f.write(f"<e>{phrase}</e>\n")
        f.write(f"</a>\n")
    print(f">> DATA: VISUAL GENOME EXTRACTING: extracted {c} situations.\n")


def extract_qa_for_training(fpath, activity=None):
    if activity not in ['skt']:
        print(f">> DATA: VISUAL GENOME EXTRACTION: ERROR: activity type {activity} not supported.")
        return 0

    print(f">> DATA: VISUAL GENOME EXTRACTING: {fpath} for training ({activity})")
    d = 'skill_training'
    Path(activity).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(fpath) as zf:
        with io.TextIOWrapper(zf.open(fpath.replace('.zip','')), encoding="utf-8") as f:
         content = json.load(f)

    with open(join(activity,fpath.replace('.zip','.'+activity)), 'w', encoding="utf-8") as f:
        c = 0
        for item in content:
            qas = item['qas']
            for qa in qas:
                question = qa['question']
                answer = qa['answer']
                idx = qa['image_id']
                if len(answer.split()) > 3:
                    if activity == 'skt':
                        f.write(f"<a type=SKT idx={idx}>\n")
                        f.write(f"<u speaker=HUM>{question}</u>\n<u speaker=BOT>{answer}</u>\n")
                        f.write(f"</a>\n")
                    c+=1
    print(f">> DATA: VISUAL GENOME EXTRACTING: extracted {c} QA pairs.\n")


def extract_object_descriptions_for_training(fpath, activity=None):
    if activity not in ['dsc']:
        print(f">> DATA: VISUAL GENOME EXTRACTION: ERROR: activity type {activity} not supported.")
        return 0

    print(f">> DATA: VISUAL GENOME EXTRACTING: {fpath} for training ({activity})")
    Path(activity).mkdir(parents=True, exist_ok=True)

    content = {}
    with zipfile.ZipFile(fpath) as zf:
        with io.TextIOWrapper(zf.open(fpath.replace('.zip','')), encoding="utf-8") as f:
            for record in ijson.items(f, "item"):
                for region in record["regions"]:
                    reg_id = region["region_id"]
                    objects = region["objects"]
                    phrase = region["phrase"]
                    if len(objects) > 0:
                        obj_ids = [obj["object_id"] for obj in objects]
                        print(reg_id, phrase, obj_ids)
                        content[reg_id] = {"description": phrase, "objects": obj_ids}

    with open(join(activity,fpath.replace('.zip','.'+activity)), 'w', encoding="utf-8") as f:
        c = 0
        for k,v in content.items():
            f.write(f"<a type=DSC idx={k}>\n")
            f.write(f"<u speaker=HUM>{v['objects']}</u>\n<u speaker=BOT>{v['description']}</u>\n")
            f.write(f"</a>\n")
            c+=1
    print(f">> DATA: VISUAL GENOME EXTRACTING: extracted {c} regions.\n")

qapath = 'question_answers.json.zip'
rdpath = 'region_descriptions.json.zip'
rgpath = 'region_graphs.json.zip'
download_vg(qapath)
download_vg(rdpath)
download_vg(rgpath)
extract_qa_for_training(qapath,'skt')
extract_rd_for_training(rdpath,'obs')
extract_object_descriptions_for_training(rgpath, 'dsc')
