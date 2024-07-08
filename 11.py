#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from pathlib import Path
import torchaudio
import IPython.display as ipd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "spokencoco_data_path",
    metavar="spokencoco-data-path",
    type=Path,
    help="path to SpokenCOCO dataset.",
)
parser.add_argument(
    "caltech_data_path",
    metavar="caltech-data-path",
    type=Path,
    help="path to Caltech dataset.",
)
args = parser.parse_args()

dataset = np.load(Path("data/splited_me_dataset.npz"), allow_pickle=True)
test = dataset['test'].item()
unseen_test = dataset['unseen_test'].item()

seen_image_to_dataset = {}
seen_dataset_to_image = {}
unseen_image_to_dataset = {}

category_ids = np.load(Path("data/coco_dataset.npz"), allow_pickle=True)['category_ids'].item()
    
annot = {}
segments_dir = Path(args.spokencoco_data_path) / Path('panoptic_annotations_trainval2017/annotations')
for fn in segments_dir.rglob('*.json'):

    with open(fn, 'r') as f:
        l = json.load(f)
    
    for entry in l['annotations']:
        annot[entry['image_id']] =  []
        
        for e in entry['segments_info']:
            id = e['category_id']
            if id in category_ids: annot[entry['image_id']].append(category_ids[id])

for fn in segments_dir.rglob(f'*.png'):
    a = Path(fn).parent.stem
    b = Path(fn).parent.parent.stem
    if a != b: continue
    num = int(fn.stem)
    
    for c in annot[num]: 
        name = f'{c}_masked_{num}'
        new_fn = Path(f'data/images/{name}.jpg')
        
        if c in images:
            if new_fn not in images[c]: continue
            if new_fn not in unseen_image_to_dataset: 
                unseen_image_to_dataset[new_fn] = 'mscoco'
                
                
        if c in test:
            if new_fn not in test[c]['images']: continue
            if new_fn not in seen_image_to_dataset: 
                seen_image_to_dataset[new_fn] = 'mscoco'
            if 'mscoco' not in seen_dataset_to_image: seen_dataset_to_image['mscoco'] = {}
            if c not in seen_dataset_to_image['mscoco']: seen_dataset_to_image['mscoco'][c] = []
            seen_dataset_to_image['mscoco'][c].append(new_fn)
            


caltech = np.load(Path('data/caltech_101_dataset.npz'), allow_pickle=True)['names'].item()

for im in Path(args.caltech_data_path).rglob('**/*.jpg'):
    
    raw_c = im.parent.stem
    
    if raw_c in caltech: 
        c = caltech[raw_c]
        num = int(im.stem.split('_')[-1])
        name = f'{c}_masked_{num}'
        new_fn = Path(f'data/images/{c}_masked_{num}.jpg')

        if c in images: 
            if new_fn not in images[c]: continue
            if new_fn not in unseen_image_to_dataset: 
                unseen_image_to_dataset[new_fn] = 'caltech'

        elif c in test:
            if new_fn not in test[c]['images']: continue
            if new_fn not in seen_image_to_dataset: 
                seen_image_to_dataset[new_fn] = 'caltech'
            if 'caltech' not in seen_dataset_to_image: seen_dataset_to_image['caltech'] = {}
            if c not in seen_dataset_to_image['caltech']: seen_dataset_to_image['caltech'][c] = []
            seen_dataset_to_image['caltech'][c].append(new_fn)


seen_classes = []
with open(Path('data/seen.txt'), 'r') as f:
    for line in f:
        seen_classes.append(line.strip())

unseen_classes = []
with open(Path('data/unseen.txt'), 'r') as f:
    for line in f:
        unseen_classes.append(line.strip())


num_episodes = 1000

episodes = {}
for i_num in tqdm(range(num_episodes)):
    
    episodes[i_num] = {'familiar_test':{}, 'novel_test': {}}
    
    for n, novel_class in enumerate(unseen_classes):
        
        novel_im = np.random.choice(images[novel_class], 1, replace=False)[0] #novel_class = np.random.choice(unseen_classes, 1, replace=False)[0]
        novel_aud = np.random.choice(unseen_test[novel_class]['english'], 1, replace=False)[0]
        
        d_n = unseen_image_to_dataset[Path(novel_im)]
    
        if 'novel' not in episodes[i_num]['novel_test']: episodes[i_num]['novel_test']['novel'] = []
        episodes[i_num]['novel_test']['novel'].append((novel_class, novel_im, novel_aud, d_n))
        
        fam_c = np.random.choice(list(seen_dataset_to_image[d_n].keys()), 1, replace=False)[0]

        seen_im = np.random.choice(seen_dataset_to_image[d_n][fam_c], 1, replace=False)[0]
        seen_aud = np.random.choice(test[fam_c]['english'], 1, replace=False)[0]
        
        if 'known' not in episodes[i_num]['novel_test']: episodes[i_num]['novel_test']['known'] = []
        episodes[i_num]['novel_test']['known'].append((fam_c, seen_im, seen_aud, d_n))
        if d_n != seen_image_to_dataset[Path(seen_im)]: print('novel ffs')


    for fam_1_class in seen_classes:
        
        seen_im = np.random.choice(test[fam_1_class]['images'], 1, replace=False)[0]
        seen_aud = np.random.choice(test[fam_1_class]['english'], 1, replace=False)[0]
        
        if 'test' not in episodes[i_num]['familiar_test']: episodes[i_num]['familiar_test']['test'] = []
        d_f = seen_image_to_dataset[Path(seen_im)]
        episodes[i_num]['familiar_test']['test'].append((fam_1_class, seen_im, seen_aud, d_f))

        fam_c = np.random.choice(list(set(seen_dataset_to_image[d_f].keys()) - set([fam_1_class])), 1, replace=False)[0]
        if fam_c == fam_1_class: print('ffs')

        seen_im = np.random.choice(seen_dataset_to_image[d_f][fam_c], 1, replace=False)[0]
        seen_aud = np.random.choice(test[fam_c]['english'], 1, replace=False)[0]
        
        if 'other' not in episodes[i_num]['familiar_test']: episodes[i_num]['familiar_test']['other'] = []
        episodes[i_num]['familiar_test']['other'].append((fam_c, seen_im, seen_aud, d_f))   
        if d_f != seen_image_to_dataset[Path(seen_im)]: print('familiar ffs')  

np.savez_compressed(
    Path("data/episodes"), 
    episodes=episodes
    )

for i in episodes:
    print(episodes[i])
    break

episodes = np.load(Path('data/episodes.npz'), allow_pickle=True)['episodes'].item()

for n in range(len(episodes[1]['familiar_test']['test'])):
    print(episodes[1]['familiar_test']['test'][n][-1], episodes[1]['familiar_test']['other'][n][-1])

for n in range(len(episodes[0]['novel_test']['novel'])):
    print(episodes[0]['novel_test']['novel'][n][-1], episodes[0]['novel_test']['known'][n][-1])