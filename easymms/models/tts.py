#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains a class definition to use the TTS models from [Meta's Massively Multilingual Speech (MMS) project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
"""

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2023,"

import json
import os
import sys
import tarfile
import logging
from pathlib import Path
from typing import List, Tuple
import torch
import soundfile as sf
import site
sys.path.append(str(Path(site.getsitepackages()[0]) / 'fairseq'))

import easymms.utils as easymms_utils
from easymms._logger import set_log_level
from easymms import constants

logger = logging.getLogger(__name__)


class TTSModel:
    """
    TTS class model

    Example usage:

    ```python
    from easymms.models.tts import TTSModel

    tts = TTSModel('eng')
    res = tts.synthesize("This is a simple example")
    tts.save(res)
    ```
    """

    def __init__(self,
                 lang: str,
                 model_dir: str = None,
                 log_level: int = logging.INFO):
        """
        Use a TTS model by its language ISO ID.
        The model will be downloaded automatically

        :param lang: TTS model language
        :param log_level: log level
        """
        self.log_level = log_level
        self.lang = lang
        set_log_level(log_level)
        # check if models_dir is provided
        if model_dir is not None:
            # verify if all files exist
            model_dir_path = Path(model_dir)
        else:
            model_dir_path = self._download_tts_model_files(lang)

        self.cp = model_dir_path / "G_100000.pth"
        assert self.cp.exists(), f"G_100000.pth not found in {str(model_dir_path)}"
        self.config = model_dir_path / "config.json"
        assert self.config.exists(), f"config.json not found in {str(model_dir_path)}"
        self.vocab = model_dir_path / "vocab.txt"
        assert self.vocab.exists(), f"vocab.txt not found in {str(model_dir_path)}"
        self.uroman_dir_path = None

        self._setup()

    def _setup(self) -> None:
        """
        Helper function to setup different required packages
        :return: None
        """
        # clone Uroman
        easymms_utils.clone(constants.UROMAN_URL, constants.UROMAN_DIR)
        # clone VITS
        easymms_utils.clone(constants.VITS_URL, constants.VITS_DIR)
        # add VITS to path
        sys.path.append(str(constants.VITS_DIR.resolve()))
        # build monotonic ext
        monotonic_ext_dir = constants.VITS_DIR / 'monotonic_align' / 'monotonic_align'
        Path.mkdir(monotonic_ext_dir, exist_ok=True)
        if len(list(monotonic_ext_dir.iterdir())) > 0:
            # extension is probably already there
            return
        else:
            from distutils.core import run_setup
            os.chdir(str(constants.VITS_DIR / 'monotonic_align'))
            run_setup(str(constants.VITS_DIR / 'monotonic_align' / 'setup.py'), script_args=['build_ext', '--inplace'],
                      stop_after='run')

    @staticmethod
    def _download_tts_model_files(lang: str):
        """
        Downloads and extracts model files into the TTS directory (::: constants.TTS_DIR)
        :param lang:
        :return:
        """
        model_dir = constants.TTS_DIR / lang
        url = constants.TTS_MODELS_BASE_URL + f"{lang}.tar.gz"
        compressed_file = easymms_utils.download_file(url, str(constants.TTS_DIR))
        # extract it
        logger.info(f"Extracting {compressed_file} to {constants.TTS_DIR}")
        tar = tarfile.open(compressed_file, "r:gz")
        tar.extractall(path=constants.TTS_DIR)
        tar.close()
        return model_dir

    @staticmethod
    def get_supported_langs() -> List[str]:
        """
        Helper function to get supported ISO 693-3 languages by the TTS models
        Source <https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html>
        :return: list of supported languages
        """
        with open(constants.MMS_LANGS_FILE) as f:
            data = json.load(f)
            return [key for key in data if data[key]['TTS']]

    def synthesize(self, txt: str, device=None):
        """
         Synthesizes the text provided as input.

        :param txt: Text
        :param lang: Language
        :param device: Pytorch device (cpu/cuda)
        :return: Tuple(data, sample_rate)
        """
        from utils import get_hparams_from_file, load_checkpoint
        from models import SynthesizerTrn
        from vits.utils import get_hparams_from_file
        try:
            from fairseq.examples.mms.tts.infer import TextMapper
        except ImportError:
            from examples.mms.tts.infer import TextMapper

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            assert device in ['cpu', 'cuda']

        text_mapper = TextMapper(str(self.vocab))
        hps = get_hparams_from_file(str(self.config))
        net_g = SynthesizerTrn(
            len(text_mapper.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net_g.to(device)
        _ = net_g.eval()
        g_pth = self.cp
        logger.info(f"loading {g_pth} ...")
        _ = load_checkpoint(g_pth, net_g, None)
        logger.info(f"text: {txt}")
        is_uroman = hps.data.training_files.split('.')[-1] == 'uroman'
        if is_uroman:
            uroman_pl = str(self.uroman_dir_path / "uroman.pl")
            txt = text_mapper.uromanize(txt, uroman_pl)
            logger.info(f"uroman text: {txt}")
        txt = txt.lower()
        txt = text_mapper.filter_oov(txt, lang=self.lang)
        stn_tst = text_mapper.get_text(txt, hps)
        # inference
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            hyp = net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()

        return hyp, hps.data.sampling_rate

    def save(self, tts_data: Tuple, out_file='out.wav') -> Path:
        """
        Saves the results of the `synthesize` function to a file

        :param tts_data: tts_data: a tuple of `wav data array` and `sample rate`
        :param out_file: output file path

        :return: out_file absolute path
        """
        set_log_level(self.log_level)
        logger.info(f"Saving audio file to {out_file}")
        sf.write(out_file, tts_data[0], tts_data[1])
        return out_file
