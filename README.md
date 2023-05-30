# EasyMMS

A simple Python package to easily use [Meta's Massively Multilingual Speech (MMS) project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms). 

[![PyPi version](https://badgen.net/pypi/v/easymms)](https://pypi.org/project/easymms/)
[![wheels](https://github.com/abdeladim-s/easymms/actions/workflows/wheels.yml/badge.svg)](https://github.com/abdeladim-s/easymms/actions/workflows/wheels.yml)
<a target="_blank" href="https://colab.research.google.com/github/abdeladim-s/easymms/blob/main/examples/EasyMMS.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

The [current MMS code](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/asr/infer/mms_infer.py) is using subprocess to call another Python script, which is not very convenient to use, and might lead to several [issues](https://github.com/facebookresearch/fairseq/issues/5117).
This package is created to address those problems and to wrap up the project in an API to easily integrate it with other projects. 
<!-- TOC -->
* [Installation](#installation)
* [Quickstart](#quickstart)
  * [ASR](#asr)
  * [ASR with Alignment](#asr-with-alignment)
  * [TTS](#tts)
  * [LID](#lid)
* [API reference](#api-reference)
* [License](#license)
* [Disclaimer & Credits](#disclaimer--credits)
<!-- TOC -->
# Installation

1. You will need [ffmpeg](https://ffmpeg.org/download.html) for audio processing

2. Also, if you want to use the [`Alignment` model](https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep):
* you will need `perl` to use [uroman](https://github.com/isi-nlp/uroman).
Check the [perl website]([perl](https://www.perl.org/get.html)) for installation instructions on different platforms.
* You will need a nightly version of `torchaudio`:
```shell
pip install -U --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```
* You might need [sox](https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html) as well.

3. Install `easymms` from Pypi
```bash
pip install easymms
```
or from source 
```bash
pip install git+https://github.com/abdeladim-s/easymms
```
4. `Fairseq` has not included the `MMS` project yet in the released PYPI version, so until the next release, you will need to install `fairseq` from source:
```shell
pip uninstall fairseq && pip install git+https://github.com/facebookresearch/fairseq
```

# Quickstart

## ASR 
You will need first to download the model weights, you can find and download all the supported models from [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr).


```python
from easymms.models.asr import ASRModel

asr = ASRModel(model='/path/to/mms/model')
files = ['path/to/media_file_1', 'path/to/media_file_2']
transcriptions = asr.transcribe(files, lang='eng', align=False)
for i, transcription in enumerate(transcriptions):
    print(f">>> file {files[i]}")
    print(transcription)
```

## ASR with Alignment

```python 
from easymms.models.asr import ASRModel

asr = ASRModel(model='/path/to/mms/model')
files = ['path/to/media_file_1', 'path/to/media_file_2']
transcriptions = asr.transcribe(files, lang='eng', align=True)
for i, transcription in enumerate(transcriptions):
    print(f">>> file {files[i]}")
    for segment in transcription:
        print(f"{segment['start_time']} -> {segment['end_time']}: {segment['text']}")
    print("----")
```

## Alignment model only

```python 
from easymms.models.alignment import AlignmentModel
    
align_model = AlignmentModel()
transcriptions = align_model.align('path/to/wav_file.wav', 
                                   transcript=["segment 1", "segment 2"],
                                   lang='eng')
for transcription in transcriptions:
    for segment in transcription:
        print(f"{segment['start_time']} -> {segment['end_time']}: {segment['text']}")
```

## TTS
Coming Soon

## LID 
Coming Soon

# API reference
You can check the [API reference documentation](https://abdeladim-s.github.io/easymms/) for more details.

# License
Since the models are [released under the CC-BY-NC 4.0 license](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md#license). 
This project is following the same [License](./LICENSE).

# Disclaimer & Credits
This project is not endorsed or certified by Meta AI and is just simplifying the use of the MMS project. 
<br/>
All credit goes to the authors and to Meta for open sourcing the models.
<br/>
Please check their paper [Scaling Speech Technology to 1000+ languages](https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/) and their [blog post](https://ai.facebook.com/blog/multilingual-model-speech-recognition/).
