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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_path",
    metavar="data-path",
    type=Path,
    help="path to Caltech dataset.",
)
args = parser.parse_args()


image_dir = args.data_path #/ Path('caltech-101/101_ObjectCategories')

image_dir.is_dir()

fns = list(image_dir.rglob('**/*.jpg'))

print(fns[0])
count = 0

classes = set()
with open(Path('data/unseen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
        
with open(Path('data/seen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
key = np.load(Path("data/image_class_key.npz"), allow_pickle=True)['key'].item()

images = {}
count = 1
for f in fns:
    c = f.parent.stem.lower()
    
    if c in key:

        if key[c] in classes:

            if key[c] not in images: images[key[c]] = []
            
            name = str(int(f.stem.split('_')[-1]))
            images[key[c]].append((name, f))
    else:
        if c in classes:

            if c not in images: images[c] = []
            
            name = str(int(f.stem.split('_')[-1]))
            images[c].append((name, f))



for i, c in enumerate(sorted(images)):
    print(i, c, len(images[c]))

np.savez_compressed(
    Path("data/caltech_101_dataset"), 
    images=images,
    names=key
    )