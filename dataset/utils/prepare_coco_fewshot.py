import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
                        help="Range of seeds")
    parser.add_argument("--anno_path", type=str, default='',
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = args.anno_path
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        for c in ID2CLASS.keys():
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            for shots in [1, 2, 3, 5, 10, 30]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f,indent=4, separators=(',', ': '))


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('/home/dlsuncheng/Dataset/NEU-DET/COCO_Annotation/', 'cocosplit', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: "crazing",
        2: "inclusion",
        3: "patches",
        4: "pitted_surface",
        5: "rolled-in_scale",
        6: "scratches",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
