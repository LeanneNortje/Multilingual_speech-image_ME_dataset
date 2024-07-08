
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
    help="path to MSCOCO dataset.",
)
args = parser.parse_args()


segments_dir = args.data_path / Path('panoptic_annotations_trainval2017/annotations')

print(segments_dir.is_dir())

classes = set()
with open(Path('data/unseen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
        
with open(Path('data/seen.txt'), 'r') as f:
    for line in f:
        classes.add(line.strip())
key = np.load(Path("data/image_class_key.npz"), allow_pickle=True)['key'].item()

images = {}
annot = {}
category_ids = {}
count = 1
for fn in segments_dir.rglob('*.json'):

    with open(fn, 'r') as f:
        l = json.load(f)
    
    for entry in l['categories']:
        c = entry['name']
        id = entry['id']

        if c in key:
            if key[c] in classes:
                if key[c] not in images: images[key[c]] = []
                category_ids[id] = c
        else:
            if c in classes:
                if c not in images: images[c] = []
                category_ids[id] = c

    
    for entry in l['annotations']:
        annot[entry['image_id']] =  entry['segments_info']
print(len(annot))

from tqdm import tqdm
segs = {}
for fn in tqdm(list(segments_dir.rglob('*.png'))):
    a = Path(fn).parent.stem
    b = Path(fn).parent.parent.stem
    if a != b: continue
    id = int(Path(fn).stem)
    
    segs[id] = set()
    seg = np.asarray(Image.open(fn)).astype(np.int32)
    seg_ids = seg[:, :, 0] + (seg[:, :, 1]*256) + (seg[:, :, 2]*256*256)
    seg_ids = list(np.unique(seg_ids))

    for e in annot[id]:
        if e['id'] in seg_ids:
            cat_id = e['category_id']
            
            if cat_id in category_ids:
                segs[id].add(category_ids[cat_id])

print(list(segs.keys())[0], len(segs))
for name in ['train2017', 'val2017']:
    for fn in Path(args.data_path).rglob(f'{name}/*.jpg'):

        id = int(Path(fn).stem)
        
        if id in segs: 
            for c in segs[id]:
                
                if c not in images: images[c] = []
                if (id, fn) not in images[c]: images[c].append((id, fn))
                


for c in sorted(images):
    print(c, len(images[c]))

# segs = {}
# for fn in segments_dir.rglob('*.png'):
#     a = Path(fn).parent.stem
#     b = Path(fn).parent.parent.stem
#     if a != b: continue
#     id = int(Path(fn).stem)
#     segs[id] = fn

# for c in images:
    
#     for name, img_fn in images[c]:

#         ids = [a['id'] for a in annot[name]]
#         pyplot.figure()
#         image = Image.open(img_fn)
#         pyplot.axis('off')
#         pyplot.imshow(image)
#         pyplot.show()
        
#         pyplot.figure()
#         seg = Image.open(segs[name])#.convert('RGB')
#         pyplot.axis('off')
#         pyplot.imshow(seg)
#         pyplot.show()
        
#         seg = np.asarray(seg).astype(np.int32)
#         cat_ids = seg[:, :, 0] + (seg[:, :, 1]*256) + (seg[:, :, 2]*256*256)

#         unique_ids = list(np.unique(np.asarray(cat_ids)))

#         decode_key = {}
#         for a in annot[name]:

#             if a['id'] in unique_ids and a['category_id'] in category_ids:
#                 if a['category_id'] not in decode_key: decode_key[a['category_id']] = []
#                 decode_key[a['category_id']].append(a['id'])

#         for cat_id in decode_key:

#             new_image = np.asarray(image).copy()
#             im_seg = np.asarray(cat_ids)
#             mask = np.ones((new_image.shape[0], new_image.shape[1], new_image.shape[2]))
#             for id in decode_key[cat_id]:
#                 mask[im_seg == id] = 0
#             new_image[mask == 1] = 255
#             pyplot.axis('off')
#             pyplot.imshow(Image.fromarray(new_image))
#             pyplot.title(category_ids[cat_id])
#             pyplot.show()


# for i, c in enumerate(sorted(images)):
#     print(i, c, len(images[c]))

np.savez_compressed(
    Path("data/coco_dataset"), 
    images=images,
    category_ids=category_ids
    )

# for id in category_ids:
#     print(category_ids[id], id)