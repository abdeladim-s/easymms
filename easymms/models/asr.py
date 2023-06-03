#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains a class definition to use the ASR models from [Meta's Massively Multilingual Speech (MMS) project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
"""

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2023,"


import os
import atexit
import json
import re
import sys
import tempfile
import logging
from pathlib import Path
from typing import Union, List
import torch
from omegaconf import OmegaConf
from pydub import AudioSegment
from pydub.utils import mediainfo

# fix importing from fairseq.examples
# import site
# sys.path.append(str(Path(site.getsitepackages()[0]) / 'fairseq'))
# try:
#     from fairseq.examples.speech_recognition.new.infer import hydra_main
# except ImportError:
#     from examples.speech_recognition.new.infer import hydra_main


from easymms import utils as easymms_utils
from easymms._logger import set_log_level
from easymms.models.alignment import AlignmentModel
from easymms import constants

logger = logging.getLogger(__name__)


class ASRModel:
    """
    MMS ASR class model

    Example usage:

    ```python
    from easymms.models.asr import ASRModel

    asr = ASRModel(model='/path/to/mms/model')
    files = ['path/to/media_file_1', 'path/to/media_file_2']
    transcriptions = asr.transcribe(files, lang='eng', align=False)
    for i, transcription in enumerate(transcriptions):
        print(f">>> file {files[i]}")
        print(transcription)
    ```
    """

    def __init__(self,
                 model: str,
                 log_level: int = logging.INFO):
        """
        :param model: path to the asr model <https://github.com/facebookresearch/fairseq/tree/main/examples/mms#asr>
        :param log_level: log level
        """
        set_log_level(log_level)
        self.cfg = constants.CFG.copy()
        self.model = Path(model)
        self.cfg['common_eval']['path'] = str(self.model.resolve())

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_dir_path = Path(self.tmp_dir.name)
        atexit.register(self._cleanup)

        self.wer = None

    def _cleanup(self) -> None:
        """
        cleans up the temp_dir
        :return: None
        """
        self.tmp_dir.cleanup()

    def _setup_tmp_dir(self, media_files: List[str]) -> None:
        """
        Sets up the tmp dir with the required files
        Taken from https://github.com/facebookresearch/fairseq/blob/main/examples/mms/asr/infer/mms_infer.py
        :param media_files: list of media files
        :return: None
        """
        logger.info(f"Setting up tmp dir: {self.tmp_dir}")
        with open(self.tmp_dir_path / "dev.tsv", "w") as fw:
            fw.write("/\n")
            for audio in media_files:
                fw.write(f"{audio[0]}\t{audio[1]}\n")
        with open(self.tmp_dir_path / "dev.uid", "w") as fw:
            fw.write(f"{audio}\n" * len(media_files))
        with open(self.tmp_dir_path / "dev.ltr", "w") as fw:
            fw.write("d u m m y | d u m m y\n" * len(media_files))
        with open(self.tmp_dir_path / "dev.wrd", "w") as fw:
            fw.write("dummy dummy\n" * len(media_files))

    def _prepare_media_files(self, media_files: List[str]) -> List[str]:
        """
        Takes the list of media files, converts them to wav and sets the sample rate to 16k
        :param media_files: list of media files
        :return: list of paths to processed media files
        """
        files = [Path(file) for file in media_files]
        res = []
        for file in files:
            logger.info(f"Preparing file {file}")
            assert file.exists(), FileNotFoundError(file)
            info = mediainfo(file)
            if info['format_name'] == 'wav' and info['sample_rate'] == '16000':
                res.append((str(file.resolve()), info['duration_ts']))
            else:
                seg = AudioSegment.from_file(file, format=info['format_name'])
                processed_file = seg.set_frame_rate(16000)
                ds = processed_file.frame_count()
                save_file_at = self.tmp_dir_path / f"{file.stem}.wav"
                processed_file.export(save_file_at, format='wav')
                res.append((str(save_file_at.resolve()), int(ds)))
        return res

    def transcribe(self,
                   media_files: List[str],
                   lang: str = 'eng',
                   device: str = None,
                   align: bool = False,
                   timestamps_type: str = 'segment',
                   max_segment_len: int = 27,
                   cfg: dict = None) -> Union[List[str], List[dict]]:
        """
        Transcribes a list of media files provided as inputs

        :param media_files: list of media files (video/audio), in whichever format supported by ffmpeg
        :param lang: the language of the media
        :param device: Pytorch device (`cuda`, `cpu` or `tpu`)
        :param align: if True the alignment model will be used to generate the timestamps, otherwise you will get raw text from the MMS model
        :param timestamps_type: Once of (`segment`, `word` or `char`) if `align` is set to True, this will be used to fragment the raw text
        :param max_segment_len: the maximum length of the fragmented segments
        :param cfg: configuration dict in case you want to use a custom configuration, see [CFG](#Constants.CFG)

        :return: List of transcription text in the same order as input files
        """
        processed_files = self._prepare_media_files(media_files)
        cwd = os.getcwd()
        # clone Fairseq
        easymms_utils.clone(constants.FAIRSEQ_URL, constants.FAIRSEQ_DIR)
        fairseq_dir = str(constants.FAIRSEQ_DIR.resolve())
        sys.path.append(fairseq_dir)
        os.chdir(fairseq_dir)
        # import
        from examples.speech_recognition.new.infer import hydra_main
        try:
            from fairseq.data.data_utils_fast import (
                batch_by_size_fn,
                batch_by_size_vec,
                batch_fixed_shapes_fast,
            )
        except ImportError:
            # we need to build the extension
            logger.info("Bulding required extensions, this may take a while ...")
            from distutils.core import run_setup
            run_setup(str((constants.FAIRSEQ_DIR / 'setup.py').resolve()), script_args=['build_ext', '--inplace'],
                      stop_after='run')


        self._setup_tmp_dir(processed_files)
        # edit cfg
        if cfg is None:
            self.cfg['task']['data'] = self.cfg['decoding']['results_path'] = str(self.tmp_dir_path.resolve())
            self.cfg['dataset']['gen_subset'] = f'{lang}:dev'
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                else:
                    device = 'cpu'
            if device == 'cuda':
                pass  # default
            elif device == 'cpu':
                self.cfg['common']['cpu'] = True
            if device == 'tpu':
                self.cfg['common']['tpu'] = True
            cfg = OmegaConf.structured(self.cfg)

        self.wer = hydra_main(cfg)
        # get results: will just read from hypo.word as I don't want to change fairseq repo to get the hypo array
        hypo_file = self.tmp_dir_path / constants.HYPO_WORDS_FILE
        res = []
        with open(hypo_file) as hw:
            hypos = hw.readlines()
            outputs = self._reorder_decode(hypos)
            transcripts = [line[1].strip() for line in outputs]
        if align:
            align_model = AlignmentModel()
            for i in range(len(transcripts)):
                media_file = processed_files[i][0]
                transcript = easymms_utils.get_transcript_segments(transcripts[i], timestamps_type, max_segment_len=max_segment_len)
                segments = align_model.align(media_file=media_file,
                                             transcript=transcript,
                                             lang=lang,
                                             device=device)
                res.append(segments)
        else:
            res = transcripts

        os.chdir(cwd)
        return res

    @staticmethod
    def get_supported_langs() -> List[str]:
        """
        Helper function to get supported ISO 693-3 languages by the ASR model
        Source <https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html>
        :return: list of supported languages
        """
        with open(constants.MMS_LANGS_FILE) as f:
            data = json.load(f)
            return [key for key in data if data[key]['ASR']]

    @staticmethod
    def _reorder_decode(hypos):
        """
        Helper method to reorder the `hypos`, see @bekarys0504 comment in
        [#5](https://github.com/abdeladim-s/easymms/issues/5)

        :param hypos: hypos list
        :return: ordered hypos
        """
        outputs = []
        for hypo in hypos:
            idx = int(re.findall("\(None-(\d+)\)$", hypo)[0])
            hypo = re.sub("\(\S+\)$", "", hypo).strip()
            outputs.append((idx, hypo))
        outputs = sorted(outputs)
        return outputs