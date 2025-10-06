import re

from typing import Optional, List
from collections import defaultdict

from schemas.schemas import InputDataBase, InputDataPut

categories = [
    "Cardiovascular disorder",
    "Neurodevelopmental disorder",
    "Others",
]

hpo_data = defaultdict(dict)
is_entry = False
hpo_id = ""
hpo_name = ""

with open("data/hp.obo", "r") as fh:
    for line in fh:
        if line.startswith("[Term]"):
            is_entry = True
            continue

        if not is_entry:
            continue
        
        if line.startswith("id: "):
            hpo_id = line.rstrip().lstrip("id: ")
        if line.startswith("name: "):
            hpo_name = line.rstrip().lstrip("name: ")
            hpo_data[hpo_id]["name"] = hpo_name
        if line.startswith("def: "):
            match = re.search(r'"(.*?)"', line)
            if match: # not always exist
                hpo_def = match.group(1)
                hpo_data[hpo_id]["def"] = hpo_def

def preprocess_input(data: InputDataBase):
    for hpo_id in data.hpo_ids:
        if hpo_id not in hpo_data:
            raise ValueError(f"Input hpo {hpo_id} is not valid")
    
    preprocessed_input = {"hpo_ids": sorted(data.hpo_ids)}
    return preprocessed_input

def preprocess_input_put(data: InputDataPut):
    for hpo_id in data.hpo_ids:
        if hpo_id not in hpo_data:
            raise ValueError(f"Input hpo {hpo_id} is not valid")

    if data.category and data.category not in categories:
        raise ValueError(f"{category} is not a valid category")
    
    preprocessed_input = dict()
    preprocessed_input["hpo_ids"] = sorted(data.hpo_ids)
    preprocessed_input["category"] = data.category
    return preprocessed_input

def make_query(hpo_ids: List[str]):
    system_prompt = "Predict disease category for the following symptoms:\n"
    system_prompt += ", ".join([hpo_data[hpo_id]["name"] for hpo_id in hpo_ids])
    return system_prompt

