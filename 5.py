#_________________________________________________________________________________________________
#
# Author: Leanne Nortje
# Year: 2023
# Email: nortjeleanne@gmail.com
#_________________________________________________________________________________________________

from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import torchaudio
import json
from PIL import Image, ImageOps
from matplotlib import image
from matplotlib import pyplot
import scipy.io as sio
from PIL import ImageDraw
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "spokencoco_data_path",
    metavar="spokencoco-data-path",
    type=Path,
    help="path to SpokenCOCO dataset.",
)
args = parser.parse_args()

bounding_boxes = True

audio = np.load(Path("data/english.npz"), allow_pickle=True)['audio'].item()
images = np.load(Path("data/images.npz"), allow_pickle=True)['images'].item()

image_words = set(list(images.keys()))

word_classes = set(list(audio.keys()))

classes = image_words.intersection(word_classes)#).intersection(keywords)

if Path('data/images').is_dir() is False: Path('data/images').mkdir()
if Path('data/english_words').is_dir() is False: Path('data/english_words').mkdir()

category_ids = np.load(Path("data/coco_dataset.npz"), allow_pickle=True)['category_ids'].item()
category_words = {}
for id in category_ids:
    category_words[category_ids[id]] = id
    print(category_ids[id], id)


segments_dir = Path(args.spokencoco_data_path) / Path('panoptic_annotations_trainval2017/annotations')
    
annot = {}
for fn in segments_dir.rglob('*.json'):
    print(fn)
    with open(fn, 'r') as f:
        l = json.load(f)
    
    for entry in l['annotations']:
        annot[entry['image_id']] =  []
        
        for e in entry['segments_info']:
            annot[entry['image_id']].append({'id': e['id'], 'category_id': e['category_id'], 'bb_box': e['bbox']})
print(len(annot))

s = set()
dataset = {}
checks = set()

masked_classes = set()

for c in tqdm(sorted(classes)):

    if c not in dataset: dataset[c] = {'images': [], 'audio': []}
    
    for name, im_fn, m in images[c]: 

        new_fn = Path('data/images') / Path(c+'_'+str(name)+'.jpg')

        shutil.copy(im_fn, new_fn)
        dataset[c]['images'].append(new_fn)
        new_fn = Path('data/images') / Path(c+'_masked_'+str(name)+'.jpg')
        # if new_fn.is_file(): continue

        if m.suffix == '.mat':
            masked_classes.add(c)
            checks.add(str(im_fn).split('/')[-4])
            

            if new_fn.is_file() is False:

                image = Image.open(im_fn)
                n = im_fn.stem.split('_')[-1]

                im_seg = sio.loadmat(m)

                bb_box = im_seg['box_coord'][0]
                x = im_seg['obj_contour'][0, :] + bb_box[2]
                y = im_seg['obj_contour'][1, :] + bb_box[0]

                s.add(image.size)
                mask = Image.new("RGBA", image.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(mask)
                coord = []
                for num in range(len(y)):
                    coord.append((x[num], y[num]))
                draw.polygon(coord, fill=(255, 255, 255, 255))
                masked_image = Image.composite(image, mask, mask).convert("RGB") 

                
                pyplot.axis('off')
                pyplot.imshow(masked_image)
                pyplot.savefig(new_fn)
                # dataset[c]['images'].append(new_fn)
                # pyplot.show()
                pyplot.close()

        elif m.suffix == '.png':
            masked_classes.add(c)
            checks.add(str(im_fn).split('/')[-3])

            if new_fn.is_file() is False:

                image = np.asarray(Image.open(im_fn).convert('RGB')).copy()
                n = im_fn.stem.split('_')[-1]
                cat_ids = np.asarray(Image.open(m)).astype(np.int32)
                cat_ids = cat_ids[:, :, 0] + (cat_ids[:, :, 1]*256) + (cat_ids[:, :, 2]*256*256)

                unique_ids = list(np.unique(np.asarray(cat_ids)))
                decode_key = {}
                for a in annot[name]:
                    if a['id'] in unique_ids and a['category_id'] in category_ids:
                        if a['category_id'] not in decode_key: decode_key[a['category_id']] = []
                        decode_key[a['category_id']].append(a['id'])

                mask = np.ones((image.shape[0], image.shape[1], image.shape[2]))
                cat_id = category_words[c]
                for id in decode_key[cat_id]:
                    mask[cat_ids == id] = 0
                image[mask == 1] = 255
                image = Image.fromarray(image)

                invert_im = ImageOps.invert(image)
                imageBox = invert_im.getbbox()
                trimed = image.crop(imageBox)

                if trimed.size[0] >= 120 and trimed.size[1] >= 120:
                    pyplot.figure(figsize=(15, 5))
                    pyplot.axis('off')
                    pyplot.imshow(trimed)
                    pyplot.savefig(new_fn,bbox_inches='tight')
                    pyplot.close()
                # dataset[c]['images'].append(new_fn)
        # break

print(checks)
np.savez_compressed(
    Path("data/english_me_dataset"), 
    dataset=dataset
    )

with open(Path('data/concepts_filtered.txt'), 'w') as file_for_writing:
    dutch_vocab = set()
    french_vocab = set()
    translations = {}
    with open(Path('data/concepts.txt'), 'r') as f:
        for line in f:
            e, d, f = line.split()
            if e in masked_classes:
                file_for_writing.write(f'{e}\t{d}\t{f}\n')