#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import torchaudio
import json
from PIL import Image, ImageOps
from matplotlib import image
from matplotlib import pyplot

audio = np.load(Path("data/english.npz"), allow_pickle=True)['audio'].item()
images = np.load(Path("data/images.npz"), allow_pickle=True)['images'].item()
dataset = np.load(Path("data/english_me_dataset.npz"), allow_pickle=True)['dataset'].item()
print(len(dataset.keys()))

image_words = set(list(images.keys()))

word_classes = set(list(audio.keys()))

classes = list(dataset.keys()) #(image_words.union(word_classes)).intersection(keywords)

category_ids = np.load(Path("data/coco_dataset.npz"), allow_pickle=True)['category_ids'].item()
category_words = {}
for id in category_ids:
    category_words[category_ids[id]] = id
    print(category_ids[id], id)

eng_dir = Path('data/english_words')
eng_dir.mkdir(parents=True, exist_ok=True)
s = set()
# dataset = {}
import scipy.io as sio
from PIL import ImageDraw
for c in tqdm(sorted(classes)):
    
    b = False
#     if len(images[c]) == 0 or len(audio[c]) == 0: continue
    if c not in dataset: dataset[c] = {'images': [], 'english': []}
    for aud_fn, start, stop in audio[c]: 
        new_fn = eng_dir / Path(c + '_' +str(Path(aud_fn).stem) +'.wav')
        
        offset = int(start * 16000)
        frames = int(np.abs(stop - start) * 16000)
        aud, sr = torchaudio.load(aud_fn, frame_offset=offset, num_frames=frames)
        if aud.size(1) != 0 and aud.size(1) > 400:
            torchaudio.save(new_fn, aud, sr)
            dataset[c]['audio'].append(new_fn)

np.savez_compressed(
    Path("data/english_me_dataset"), 
    dataset=dataset
    )