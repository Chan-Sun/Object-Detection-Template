#%%%
root_path="/home/sunchen/Projects/det_project_template/main"
import sys
sys.path.append(root_path)

from mmdet.datasets import build_dataset
from main import *
from utils import save_json

image_dir = ""
dataset_json_path = ""
output_dir = ""
min_area_ratio=0.1
ignore_negative_samples=True
output_images_dir = ""
sliced_coco_name = ""

from sahi.slicing import slice_coco

# assure slice_size is list
slice_size_list = 1024
if isinstance(slice_size_list, (int, float)):
    slice_size_list = [slice_size_list]
overlap_ratio = 0

# slice coco dataset images and annotations
print("Slicing step is starting...")
for slice_size in slice_size_list:
    # in format: train_images_512_01
    
    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=dataset_json_path,
        image_dir=image_dir,
        output_coco_annotation_file_name="",
        output_dir=output_images_dir,
        ignore_negative_samples=ignore_negative_samples,
        slice_height=slice_size,
        slice_width=slice_size,
        min_area_ratio=min_area_ratio,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        out_ext=".jpg",
        verbose=False,
    )
    output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + ".json")
    save_json(output_coco_annotation_file_path,coco_dict)
    print(f"Sliced dataset for 'slice_size: {slice_size}' is exported to {output_dir}")
