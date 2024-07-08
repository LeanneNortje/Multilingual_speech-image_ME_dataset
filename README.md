Before you get started download the following speech and image datasets at the links provided. Make sure to extract the datasets to have the correct structure or change the code accordingly.

# Image datasets:
* ## Caltech-101
  
  [Download](https://data.caltech.edu/records/mzrjq-6wc02)
  
  Structure:

```bash
├── MACOSX
│   ├── ._caltech-101
│   ├── caltech-101
│   │   ├── ._101_ObjectCategories.tar.gz
│   │   ├── ._Annotations.tar
│   │   ├── ._show_annotation.m
├── caltech-101
│   ├── 101_ObjectCategories.tar.gz
│   ├── Annotations.tar
│   ├── show_annotation.m
```

* ## ImageNet
  
  Structure:

```bash
├── ILSVRC
│   ├── Annotations
│   ├── Data
│   │   ├── test 
│   │   │   ├── ILSVRC2012_test_00033334.JPEG
│   │   │   ├── ILSVRC2012_test_00050001.JPEG
│   │   │   ├── ...
│   │   │   ├── ILSVRC2012_test_00066668.JPEG
│   │   │   ├── ILSVRC2012_test_00083335.JPEG
│   │   ├── train
│   │   │   ├── n01917289_968.JPEG
│   │   │   ├── n02104029_956.JPEG
│   │   │   ├── ...
│   │   │   ├── n02342885_1950.JPEG
│   │   │   ├── n02834397_34349.JPEG 
│   │   ├── val 
│   │   │   ├── ILSVRC2012_val_00007143.JPEG
│   │   │   ├── ILSVRC2012_val_00014286.JPEG
│   │   │   ├── ...
│   │   │   ├── ILSVRC2012_val_00021429.JPEG
│   │   │   ├── ILSVRC2012_val_00028572.JPEG
│   ├── LOC_synset_mapping.txt
```
  
To download the content for the ```train```, ```val```, ```test``` and ```Annotation``` folder, go [here]([https://data.caltech.edu/records/mzrjq-6wc02](https://image-net.org/download-images.php)).
  
For the training samples, download the ```Training images (Task 1 & 2)``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
For the validation images, download the ```Validation images (all tasks)``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
For the testing images, download the ```Test images (all tasks).``` zip file under the 2012 Images of ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
The Annotation files can be downloaded under ```Training bounding box annotations (Task 1 & 2 only) ``` under the bounding boxes of the 2012 ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)```.
Lastly, download the ```LOC_synset_mapping.txt``` file [here](https://github.com/formigone/tf-imagenet/blob/master/LOC_synset_mapping.txt).

* ## MSCOCO

Download the MSCOCO data [here](https://cocodataset.org/#download). We used the 2017 splits. Therefore, download ```2017 Train images [118K/18GB]```, ```2017 Val images [5K/1GB]```, and ```2017 Test images [41K/6GB]```. Also, download the ```2017 Panoptic Train/Val annotations [821MB]```. Extract all these in a MSCOCO folder named ```<any name for the MSCOCO dataset folder>``` so that your dataset directory looks like this:

```bash
├── <any name for the MSCOCO dataset folder>
│   ├── panoptic_annotations_trainval2017
│   ├── train2017
│   ├── val017
│   ├── test2017
```

# Speech datasets

* ## Buckeye

Download the corpus [here](https://buckeyecorpus.osu.edu/).
  
* ## Flckr Audio

Download the corpus [here](https://groups.csail.mit.edu/sls/downloads/flickraudio/).
  
* ## LibriSpeech

Download the LibriSpeech corpus [here](https://www.openslr.org/12) and specifically download the following:

- [ ] ```train-clean-100.tar.gz ``` 
- [ ] ```train-clean-360.tar.gz``` 
- [ ] ```train-other-500.tar.gz```
- [ ] ```dev-clean.tar.gz```
- [ ] ```dev-other.tar.gz```
- [ ] ```test-clean.tar.gz```
- [ ] ```test-other.tar.gz```

Extract all these in a MSCOCO folder named ```<any name for the LibriSpeech corpus folder>``` so that your dataset directory looks like this:

```bash
├── <any name for the LibriSpeech corpus folder>
│   ├── train-clean-100
│   ├── train-clean-360
│   ├── train-other-500
│   ├── dev-clean
│   ├── dev-other
│   ├── test-clean
│   ├── test-other
```
To get forced alignments required for testing, use the [Montreal forced aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/). 
Or simply use the words.txt file in the releases and paste it in the ```<any name for the LibriSpeech corpus folder>``` folder.

# Dataset generation

Setup the image part of the dataset py running the scripts in the exact order:
```
python 1_process_Caltech.py /path/to/caltech-101
python 2_process_ImageNet.py /path/to/ImageNET
python 3_process_COCO.py /path/to/MSCOCO
python 4.py /path/to/flickr_audio /path/to/buckeye /path/to/librispeech /path/to/spokencoco /path/to/caltech-101 /path/to/imagenet
python 5.py /path/to/spokencoco
python 6.py 
python 7.py
python 8.py
```
Open ```9.ipynb``` and ```10.ipynb``` in jupyter notebook since you have to manually choose dev and test images for familiar classes in ```9.ipynb``` and for novel classes in ```10.ipynb```.

Now generate the episodes:
```
python 11.py
```
