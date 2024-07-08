#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2024
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
import json
import numpy as np
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_path",
    metavar="data-path",
    type=Path,
    help="path to ImageNet dataset.",
)
args = parser.parse_args()

f = open(args.data_path / Path('LOC_synset_mapping.txt'), 'r')

class_key = {}
class_key_classes_to_keys = {}

for line in f:
    p = line.lower().split()
    key = p[0]
    class_key[key] = []
    for w in p[1:]:
        word = w.split(',')[0]
        # word = p[1]
    
        class_key[key].append(word)

    if word not in class_key_classes_to_keys: class_key_classes_to_keys[word] = []
    class_key_classes_to_keys[word].append(key)


image_dir = args.data_path / Path('Data')


fns = list(image_dir.rglob('*.JPEG'))

annot_dir = args.data_path / Path('Annotations')

maps = {}
for fn in list(annot_dir.rglob(f'*.xml')):
    id = Path(fn).parent.stem
    name = fn.stem
    if name not in maps: maps[name] = fn


for fn in fns:
    name = fn.stem
    if fn.parent.stem == 'train': print(fn, name)
    if name in maps: 
        print(name, fn, maps[name])
        break


classes = set()
with open(Path('data/unseen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
        
with open(Path('data/seen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
# print(classes)
# print(class_key_classes_to_keys.keys())
print(len(classes.intersection(set(class_key_classes_to_keys.keys()))))
print(sorted(classes.intersection(set(class_key_classes_to_keys.keys()))))
key = np.load(Path("data/image_class_key.npz"), allow_pickle=True)['key'].item()

images = {}
for f in tqdm(fns):

    id = f.stem.split('_')[0]
    if id not in class_key: continue
    cs = class_key[id]
#     if c not in names: 
#         if c not in images and c not in vito:
#             choice = input(f'{count} {c}')
#             if choice == 'y': 
#                 if c not in images: 
#                     images[c] = []
#                 names[c] = c
#             elif choice == 'r': 
#                 n = input('new name')
#                 if n not in images: 
#                     images[n] = []
#                 names[c] = n
#                 vito.append(c)
#             else: vito.append(c)
#             count += 1
    for c in cs:
        if c in key:
            if key[c] in classes:
                if key[c] not in images: images[key[c]] = []
                name = f.stem
                if name not in maps: continue
                if (name, f, maps[name]) not in images[key[c]]: images[key[c]].append((name, f, maps[name]))
        else:
            if c in classes:
                if c not in images: images[c] = []
                name = f.stem
                if name not in maps: continue
                if (name, f, maps[name]) not in images[c]: images[c].append((name, f, maps[name]))
    

for c in images:
    if len(images[c]) == 0: print(c, len(images[c]))

print(len(images))

for i, c in enumerate(sorted(images)):
    print(i, c, len(images[c]))

np.savez_compressed(
    Path("data/imagenet"), 
    images=images,
    key_to_class=class_key,
    class_to_key=class_key_classes_to_keys
    )
