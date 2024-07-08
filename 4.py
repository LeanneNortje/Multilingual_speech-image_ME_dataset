from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import torchaudio
import json
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "flickr_data_path",
    metavar="flickr-data-path",
    type=Path,
    help="path to flickr dataset.",
)
parser.add_argument(
    "buckeye_data_path",
    metavar="buckeye-data-path",
    type=Path,
    help="path to Buckeye dataset.",
)
parser.add_argument(
    "librispeech_data_path",
    metavar="librispeech-data-path",
    type=Path,
    help="path to librispeech dataset.",
)
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
parser.add_argument(
    "imagenet_data_path",
    metavar="imagenet-data-path",
    type=Path,
    help="path to ImageNet dataset.",
)
args = parser.parse_args()


word_wavs = {}
boundaries = {}
flickr_boundaries_fn = Path(args.flickr_data_path) / Path('flickr_8k.ctm')
flickr_audio_dir = flickr_boundaries_fn.parent / "wavs"

with open(flickr_boundaries_fn, 'r') as file:
    for line in tqdm(file):
        parts = line.strip().split()
        name = parts[0].split(".")[0]
        number = parts[0].split("#")[-1]
        wav = flickr_audio_dir / Path(name + "_" + number + ".wav")
        if wav.is_file():
            word = parts[-1].lower()
            if word not in word_wavs: word_wavs[word] = []
            word_wavs[word].append(wav)
            if wav not in boundaries: boundaries[wav] = {}
            boundaries[wav][word] = (float(parts[2]), float(parts[2]) + float(parts[3]))
        else: print(wav)

words = set(word_wavs.keys())

buckeye_speech_audio_dir = Path(args.buckeye_data_path)
wavs = list(buckeye_speech_audio_dir.rglob(f'*.wav'))
all_wavs = {}
for w in wavs:
    all_wavs[w.stem] = w

buckeye_word_wavs = {}
buckeye_boundaries = {}
buckeye_boundaries_fn = Path(args.buckeye_data_path) / Path('english.wrd')

with open(buckeye_boundaries_fn, 'r') as file:
    for line in tqdm(file):
        id, start, stop, word = line.split()
        wav = all_wavs[id]
        if wav.is_file() and word != 'SIL':
            word = word.lower()
            if word not in buckeye_word_wavs: buckeye_word_wavs[word] = []
            buckeye_word_wavs[word].append(wav)
            if wav not in buckeye_boundaries: buckeye_boundaries[wav] = {}
            buckeye_boundaries[wav][word] = (float(start), float(stop))

buckeye_words = set(buckeye_word_wavs.keys())

libri_speech_audio_dir = Path(args.librispeech_data_path)
wavs = list(libri_speech_audio_dir.rglob(f'*.flac'))
all_wavs = {}
for w in wavs:
    all_wavs[w.stem] = w

libri_speech_word_wavs = {}
libri_speech_boundaries = {}
libri_speech_boundaries_fn = Path(args.librispeech_data_path) / Path('words.txt')

file = open(libri_speech_boundaries_fn, 'r')
for line in tqdm(file):
    parts = line.strip().split()
    name = parts[0]

    wav = all_wavs[name]
    if wav.is_file():
        word = parts[-1]
        if word not in libri_speech_word_wavs: libri_speech_word_wavs[word] = []
        libri_speech_word_wavs[word].append(wav)
        if wav not in libri_speech_boundaries: libri_speech_boundaries[wav] = {}
        libri_speech_boundaries[wav][word] = (float(parts[1]), float(parts[2]))
    else: print(wav)

libri_speech_words = set(libri_speech_word_wavs.keys())

word_classes = (libri_speech_words.union(buckeye_words)).union(words)

audio = {}
for wav in tqdm(libri_speech_boundaries):
    for word in libri_speech_boundaries[wav]:
        if word not in audio: audio[word] = []
        audio[word].append((wav, libri_speech_boundaries[wav][word][0], libri_speech_boundaries[wav][word][1]))
        
for wav in tqdm(buckeye_boundaries):
    for word in buckeye_boundaries[wav]:
        if word not in audio: audio[word] = []
        audio[word].append((wav, buckeye_boundaries[wav][word][0], buckeye_boundaries[wav][word][1]))
        
for wav in tqdm(boundaries):
    for word in boundaries[wav]:
        if word not in audio: audio[word] = []
        audio[word].append((wav, boundaries[wav][word][0], boundaries[wav][word][1]))

libri_speech_boundaries = {}
buckeye_boundaries = {}
boundaries = {}

caltech_images = np.load(Path("data/caltech_101_dataset.npz"), allow_pickle=True)['images'].item()
coco_images = np.load(Path("data/coco_dataset.npz"), allow_pickle=True)['images'].item()
imagenet_images = np.load(Path("data/imagenet.npz"), allow_pickle=True)['images'].item()

caltech_words = set(list(caltech_images.keys()))

coco_words = set(list(coco_images.keys()))

imagenet_words = set(list(coco_images.keys()))

image_words = (caltech_words.union(coco_words)).union(imagenet_words)

classes = image_words.intersection(word_classes)

segments_dir = Path(args.spokencoco_data_path) / Path('panoptic_annotations_trainval2017/annotations')

segs = {}
for fn in segments_dir.rglob('*.png'):
    a = Path(fn).parent.stem
    b = Path(fn).parent.parent.stem
    if a != b: continue
    id = int(Path(fn).stem)
    segs[id] = fn
    
annot = {}
for fn in segments_dir.rglob('*.json'):
    print(fn)
    with open(fn, 'r') as f:
        l = json.load(f)
    
    for entry in l['annotations']:
        annot[entry['image_id']] =  entry['segments_info']
print(len(annot))


names = np.load(Path("data/caltech_101_dataset.npz"), allow_pickle=True)['names'].item()

images = {}
for c in tqdm(caltech_images):
    if c not in images: images[c] = []
    for name, im_fn in caltech_images[c]:
        n = im_fn.stem.split('_')[-1]
        m = list((Path(args.caltech_data_path) / Path(f'caltech-101/Annotations')).rglob(f'{im_fn.parent.stem}*/annotation_{n}.mat'))
        if len(m) == 1:
            m = m[0]  
            images[c].append((name, im_fn, m))

for c in tqdm(coco_images):
    if c not in images: images[c] = []
    for name, im_fn in coco_images[c]:
        m = segs[name]  
        images[c].append((name, im_fn, m))
        
for c in tqdm(imagenet_images):
    if c not in images: images[c] = []
    for name, im_fn, m in imagenet_images[c]:
        images[c].append((name, im_fn, m))


category_ids = np.load(Path("data/coco_dataset.npz"), allow_pickle=True)['category_ids'].item()
category_words = {}
for id in category_ids:
    category_words[category_ids[id]] = id
    print(category_ids[id], id)

np.savez_compressed(
    Path("data/english"), 
    audio=audio
    )

np.savez_compressed(
    Path("data/images"), 
    images=images
    )

# for c in audio:
#     print(c, len(audio[c]))