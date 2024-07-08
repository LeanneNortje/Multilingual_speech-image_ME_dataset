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
    "mls_dutch_data_path",
    metavar="mls-dutch-data-path",
    type=Path,
    help="path to MLS Dutch dataset.",
)
parser.add_argument(
    "cgn_dutch_data_path",
    metavar="cgn-dutch-data-path",
    type=Path,
    help="path to CGN Dutch dataset.",
)
parser.add_argument(
    "cv_dutch_data_path",
    metavar="cv-dutch-data-path",
    type=Path,
    help="path to Common Voice Dutch dataset.",
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

cv_fn = Path(args.cv_dutch_data_path) / Path('nl/words.txt')
print(cv_fn.is_file())

cgn_dir = Path(args.cgn_dutch_data_path)
print(cgn_dir.is_dir())

temp_fn = 'temp.wav'

read_in = set()
aud_dir = cv_fn.parent / 'clips'
dutch_me_dataset = {}
# sr = 32000
with open(cv_fn, 'r') as f:
    for line in tqdm(f):
        wav = aud_dir / Path(line.split()[0] + '.mp3')
#         print(wav)
        word = line.split()[-1]
        if word in dutch_vocab:

            read_in.add(word)

            name = wav.stem.split('_')[-1]
            new_fn = Path('data/dutch_words')
            new_fn.mkdir(parents=True, exist_ok=True)
            new_fn = new_fn / Path(f'{word}_{name}.wav')
            
            if new_fn.is_file() is False:
                
                sound = AudioSegment.from_mp3(wav)
                sound.export(temp_fn, format="wav")
                aud, sr = torchaudio.load(temp_fn)
                
                start = float(line.split()[1])
                end = float(line.split()[2])
                offset = int(start * sr)
                dur = int((end - start) * sr)

                aud, sr = torchaudio.load(temp_fn, frame_offset=offset, num_frames=dur)
            
                if sr != 16000:
                    aud = torchaudio.functional.resample(aud, orig_freq=sr, new_freq=16000)
                    sr = 16000
                torchaudio.save(new_fn, aud, sr)
            
            if word not in dutch_me_dataset: dutch_me_dataset[word] = []
            dutch_me_dataset[word].append(new_fn)


aud_dir = cgn_dir / Path('data')

for fn in tqdm(cgn_dir.rglob("**/nl/*.wav")):
    
    reading = []

    wav = fn#aud_dir / fn.parent.parent.stem / fn.parent.stem / Path(fn.stem.split('.')[0] + '.wav')
    name = fn.stem.split('.')[0]
    text = aud_dir / Path('annot/text/wrd') / fn.parent.parent.stem / fn.parent.stem / Path(fn.stem.split('.')[0] + '.wrd')

    if text.is_file() is False: continue
    file = open(text, 'rb')
    for line in file:
        
        line = line.decode('latin-1')

        if "\"" not in line:
            reading.append(line)
        else: 
            line = line.split('\"')[1]
            if line in dutch_vocab:
                word = line
                read_in.add(word)
                
                new_fn = Path('data/dutch_words')
                new_fn.mkdir(parents=True, exist_ok=True)
                new_fn = new_fn / Path(f'{word}_{name}.wav')
                
                if new_fn.is_file() is False:
                    aud, sr = torchaudio.load(wav)
                    start = float(reading[-2])
                    end = float(reading[-1])
                    offset = int(start * sr)
                    dur = int((end - start) * sr)
                    aud, sr = torchaudio.load(wav, frame_offset=offset, num_frames=dur)
                    if sr != 16000:
                        aud = torchaudio.functional.resample(aud, orig_freq=sr, new_freq=16000)
                        sr = 16000
                    torchaudio.save(new_fn, aud, sr)
                reading = []
                if word not in dutch_me_dataset: dutch_me_dataset[word] = []
                dutch_me_dataset[word].append(new_fn)

mls_fn = Path(args.mls_dutch_data_path) / Path("words.txt")
print(mls_fn.is_file())

aud_dir = mls_fn.parent
names = {}

for wav in tqdm(list(aud_dir.rglob(f'**/*.opus'))):
    name = wav.stem
    if name not in names:
        names[name] = wav
    else:
        print(wav, names[name])


with open(mls_fn, 'r') as f:
#     print(len(f))
    for line in tqdm(f):
        name = line.split()[0]
        start = float(line.split()[1])
        end = float(line.split()[2])
        word = line.split()[-1]
        
        if word in dutch_vocab:
            wav = names[name]
            read_in.add(word)
    
            new_fn = Path('data/dutch_words')
            new_fn.mkdir(parents=True, exist_ok=True)
            new_fn = new_fn / Path(f'{word}_{name}.wav')
            
            if new_fn.is_file() is False:
                aud, sr = torchaudio.load(wav)
                offset = int(start * sr)
                dur = int((end - start) * sr)
                aud, sr = torchaudio.load(wav, frame_offset=offset, num_frames=dur)
                if sr != 16000:
                    aud = torchaudio.functional.resample(aud, orig_freq=sr, new_freq=16000)
                    sr = 16000
                torchaudio.save(new_fn, aud, sr)
            
            if word not in dutch_me_dataset: dutch_me_dataset[word] = []
            dutch_me_dataset[word].append(new_fn)
            

print(dutch_vocab - read_in)

for c in dutch_me_dataset:
    if len(dutch_me_dataset[c]) == 0: 
        dutch.remove(c)
        print(c)

dataset = np.load(Path("data/english_me_dataset.npz"), allow_pickle=True)['dataset'].item()

dutch = {}
for e_w in dataset:
    if e_w not in translations: continue
    d_w = translations[e_w]['dutch']
    if d_w not in dutch_me_dataset: continue
    if d_w not in dutch: dutch[d_w] = {'audio': []}
    dutch[d_w]['audio'] = dutch_me_dataset[d_w]
#     dutch[d_w]['images'] = dataset[e_w]['images']

for c in dutch:
    if len(dutch[c]['audio']) == 0:
#         or len(dutch[c]['images']) == 0: 
        print(c)

np.savez_compressed(
    Path("data/dutch_me_dataset"), 
    dataset=dutch
    )

print(dutch_me_dataset.keys())