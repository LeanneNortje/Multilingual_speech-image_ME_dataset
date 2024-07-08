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
from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import gzip
import librosa
from pydub import AudioSegment
import IPython.display as ipd


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "mls_french_data_path",
    metavar="mls-french-data-path",
    type=Path,
    help="path to MLS French dataset.",
)
parser.add_argument(
    "cv_french_data_path",
    metavar="cv-french-data-path",
    type=Path,
    help="path to Common Voice French dataset.",
)
args = parser.parse_args()


dutch_vocab = set()
french_vocab = set()
translations = {}
with open(Path('data/concepts.txt'), 'r') as f:
    for line in f:
        english, dutch, french = line.split()
        dutch_vocab.add(dutch)
        french_vocab.add(french)
        if english not in translations:
            translations[english] = {
                'dutch': dutch,
                'french': french
            }

dataset = np.load(Path("data/english_me_dataset.npz"), allow_pickle=True)['dataset'].item()

mls_fn = Path(args.mls_french_data_path) / Path("words.txt")

print(mls_fn.is_file())

read_in = set()
french_me_dataset = {}
aud_dir = mls_fn.parent

names = {}

for wav in tqdm(list(aud_dir.rglob(f'**/*.opus'))):
    name = wav.stem
    if name not in names:
        names[name] = wav
    else:
        print(wav, names[name])

with open(mls_fn, 'r') as f:
    for line in tqdm(f):
        name = line.split()[0]
        start = float(line.split()[1])
        end = float(line.split()[2])
        word = line.split()[-1]
        
        if word in french_vocab:
            wav = names[name]
            read_in.add(word)
            aud, sr = torchaudio.load(wav)
            offset = int(start * sr)
            dur = int((end - start) * sr)
            aud, sr = torchaudio.load(wav, frame_offset=offset, num_frames=dur)
            
            new_fn = Path('data/french_words')
            new_fn.mkdir(parents=True, exist_ok=True)
            new_fn = new_fn / Path(f'{word}_{name}.wav')
            if sr != 16000:
                aud = torchaudio.functional.resample(aud, orig_freq=sr, new_freq=16000)
                sr = 16000
#             print(aud.size())
            torchaudio.save(new_fn, aud, sr)
#             print(word)
#             aud = aud.squeeze().numpy()
#             ipd.display(ipd.Audio(aud, rate=sr))

            if word not in french_me_dataset: french_me_dataset[word] = []
            french_me_dataset[word].append(new_fn)


cv_fn = Path(args.cv_french_data_path) / Path("fr/words.txt")
print(cv_fn.is_file())

temp_fn = 'temp.wav'

aud_dir = cv_fn.parent / 'clips'
with open(cv_fn, 'r') as f:
    for line in tqdm(f):
        wav = aud_dir / Path(line.split()[0] + '.mp3')
        word = line.split()[-1]
        if word in french_vocab:
            read_in.add(word)
            
            sound = AudioSegment.from_mp3(wav)
            sound.export(temp_fn, format="wav")
            aud, sr = torchaudio.load(temp_fn)

            offset = int(float(line.split()[1]) * sr)
            dur = int((float(line.split()[2]) - float(line.split()[1])) * sr)
            
            aud, sr = torchaudio.load(temp_fn, frame_offset=offset, num_frames=dur)

            name = wav.stem.split('_')[-1]
            new_fn = Path('data/french_words')
            new_fn.mkdir(parents=True, exist_ok=True)
            new_fn = new_fn / Path(f'{word}_{name}.wav')
            if sr != 16000:
                aud = torchaudio.functional.resample(aud, orig_freq=sr, new_freq=16000)
                sr = 16000
#             print(aud.size())
            torchaudio.save(new_fn, aud, sr)
#             print(word)
#             aud = aud.squeeze().numpy()
#             ipd.display(ipd.Audio(aud, rate=sr))

            if word not in french_me_dataset: french_me_dataset[word] = []
            french_me_dataset[word].append(new_fn)

print(french_vocab - read_in)

french = {}
for e_w in dataset:
    if e_w not in translations: continue
    f_w = translations[e_w]['french']
    if f_w not in french: french[f_w] = {'audio': []}
    french[f_w]['audio'] = french_me_dataset[f_w]
#     french[f_w]['images'] = dataset[e_w]['images']

for c in french:
    if len(french[c]['audio']) == 0:
#         or len(french[c]['images']) == 0: 
        print(c)

np.savez_compressed(
    Path("data/french_me_dataset"), 
    dataset=french
    )