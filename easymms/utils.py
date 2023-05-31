#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utils functions
"""

import json
import logging
import os
import tarfile
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import requests
from git import Repo
from tqdm import tqdm

from easymms import constants

logger = logging.getLogger(__name__)


def download_file(url: str, download_dir=None, chunk_size=1024) -> str:
    """
    Helper function to download models and other required files
    :param url: URL of the file
    :param download_dir: Where to store the file
    :param chunk_size: size of the download chunk

    :return: Absolute path of the downloaded model
    """

    os.makedirs(download_dir, exist_ok=True)
    file_name = os.path.basename(url)
    file_path = Path(download_dir) / file_name
    # check if the file is already there
    if file_path.exists():
        logging.info(f"File '{file_name}' already exists in {download_dir}")
    else:
        # download it
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get('content-length', 0))

        progress_bar = tqdm(desc=f"Downloading File {file_name} ...",
                            total=total,
                            unit='iB',
                            unit_scale=True,
                            unit_divisor=1024)

        try:
            with open(file_path, 'wb') as file, progress_bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    progress_bar.update(size)
            logging.info(f"Model downloaded to {file_path.absolute()}")
        except Exception as e:
            # error download, just remove the file
            os.remove(file_path)
            raise e
    return str(file_path.absolute())


def download_and_extract(url, extract_to='.'):
    """
    Downloads and unzips a zip folder
    Will be used to download uroman source code
    :param url: the file URL
    :param extract_to: extract path
    :return: None
    """
    logger.info(f"Downloading file '{url}' and extracting to '{extract_to}' ...")
    http_response = urlopen(url)
    file_name = os.path.basename(url)
    if file_name.endswith('.zip'):
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)
    else:
        tar = tarfile.open(BytesIO(http_response.read()), "r:gz")
        tar.extractall(path=extract_to)
        tar.close()


def clone(repo_url, destination_folder: Path):
    if destination_folder.exists():
        logger.info(f"{destination_folder} already exists")
    else:
        logger.info(f"Cloning from {repo_url} tp {destination_folder} ...")
        Repo.clone_from(repo_url, str(destination_folder.resolve()))
    return destination_folder
def get_uroman():
    clone(constants.UROMAN_URL, constants.UROMAN_DIR)
    return constants.UROMAN_DIR / 'bin'

def get_lang_info(lang: str) -> dict:
    """
    Returns more info about a language,
    :param lang: the ISO 693-3 language code
    :return: dict of info
    """
    with open(constants.MMS_LANGS_FILE) as f:
        data = json.load(f)
        return data[lang]


def get_transcript_segments(transcript: str, t_type: str = 'segment', max_segment_len: int = 24):
    """
    A helper function to fragment the transcript to segments
    <Quick implementation, Not perfect (barely works), needs improvements>

    :param transcript: results of the ASR model
    :param t_type: one of [`word`, `segment`, `char`]
    :param max_segment_len: the maximum length of the segment
    :return: list of segments
    """
    res = []
    if t_type == 'word':
        res = transcript.split()
    elif t_type == 'char':
        s = ''
        for c in transcript:
            if c == ' ':
                continue
            if len(s) >= max_segment_len:
                res.append(s)
                s = c
            else:
                s += c
        res.append(s)
    else:
        s = ''
        words = transcript.strip().split()
        for word in words:
            new_word = s + ' ' + word
            if len(new_word) > max_segment_len:
                res.append(s.strip())
                s = word
            else:
                s = new_word
        res.append(s)

    return res
