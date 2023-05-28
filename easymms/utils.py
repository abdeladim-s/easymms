#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" """
import json
import logging
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

from easymms.constants import MMS_LANGS_FILE

logger = logging.getLogger(__name__)

def download_and_unzip(url, extract_to='.'):
    """
    Downloads and unzips a zip folder
    Will be used to download uroman source code
    :param url: the file URL
    :param extract_to: extract path
    :return: None
    """
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


def get_lang_info(lang: str) -> dict:
    """
    Returns more info about a language,
    :param lang: the ISO 693-3 language code
    :return: dict of info
    """
    with open(MMS_LANGS_FILE) as f:
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

if __name__ == '__main__':
    print(get_lang_info('ara'))
