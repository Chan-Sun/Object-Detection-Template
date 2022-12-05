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

def read_txt(txt_path):
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            # print(line)

def write_txt(txt_path,txt_list):    
    with open(txt_path,"w") as f:
        f.writelines(txt_list)

def read_folder(folder_path):
    return os.listdir(folder_path)