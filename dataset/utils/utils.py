import json
import os
import os.path as osp

def read_json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

def save_json(json_path,json_dict):
    with open(json_path,"w") as fp:
        json.dump(json_dict,fp,indent=4,separators=(",",": "))
