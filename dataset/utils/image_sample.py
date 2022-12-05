from utils import read_json,save_json
from random import sample 
json_path = "/home/sunchen/Projects/GoerTek/dataset/PCB-Images/Annotations/trainval.json"

json_file = read_json(json_path=json_path)
category_info = json_file["categories"]
image_info = json_file["images"]
anno_info = json_file["annotations"]
num = 20
sample_id_list = []
class_list = [cat["name"] for cat in category_info]
cat2imgid = {cat:[] for cat in class_list}
for img in image_info:
    file_name = img["file_name"][:-4]
    img_cat = file_name[3:-3]
    id = img["id"]
    cat2imgid[img_cat].append(id)
for cat, id_list in cat2imgid.items():
    a = sample(id_list,num)
    sample_id_list.extend(a)
img_list = []
anno_list = []
for idx,img in enumerate(json_file["images"]):
    if img["id"] in sample_id_list:
        img_list.append(img)
for idx,anno in enumerate(json_file["annotations"]):
    if anno["image_id"] in sample_id_list:
        anno_list.append(anno)
json_file = {
    "images":img_list,
    "annotations":anno_list,
    "categories":category_info    
}
save_json("/home/sunchen/Projects/GoerTek/dataset/PCB-Images/Annotations/SSTrain.json",json_file)