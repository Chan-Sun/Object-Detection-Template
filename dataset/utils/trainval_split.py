import json
import numpy as np
import copy
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="Range of seeds")
    parser.add_argument("--anno_path", type=str, default="",
                        help="annotation filepath")    
    parser.add_argument("--save_path", type=str, default="",
                        help="annotation filepath")      
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    coco_anno = args.anno_path
    save_path = args.save_path
    np.random.seed(args.seed)
    os.makedirs(save_path,exist_ok=True)
    with open(coco_anno,"r") as load_json:
        load_json = json.load(load_json)

    image_num = len(load_json["images"])
    image_id_list = [i for i in range(1,image_num+1)]
    all_list = dict(trainval=[],test=[])
    np.random.shuffle(image_id_list)
    split_point = image_num//5
    all_list["test"].extend(image_id_list[:split_point])
    # all_list["val"].extend(image_id_list[split_point:split_point*3])
    # all_list["train"].extend(image_id_list[split_point*3:])
    all_list["trainval"].extend(image_id_list[split_point:])

    category = load_json["categories"]

    for i in all_list:
        train_dict = {"images": [],
                        "type": "instances",
                        "annotations": [],
                        "categories": category}
        for images in load_json["images"]:
            if images["id"] in all_list[i]:
                train_dict["images"].append(images)
        for anno in load_json["annotations"]:
            if anno["image_id"] in all_list[i]:
                train_dict["annotations"].append(anno)
        save_file = os.path.join(save_path,i+'.json')
        with open(save_file, 'w') as fp:
            json.dump(train_dict, fp,indent=4)