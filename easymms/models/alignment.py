#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains a simple class to use the AlignmentModel model from the MMS project
More info can be found here: https://github.com/facebookresearch/fairseq/tree/main/examples/mms/data_prep
"""
import logging
import shutil
import sys
from pathlib import Path
from typing import List
import torch
from easymms import utils as easymms_utils
from easymms._logger import set_log_level
from easymms import constants
from easymms.constants import PACKAGE_DATA_DIR, ALIGNMENT_MODEL_URL, ALIGNMENT_DICTIONARY_URL, UROMAN_URL, \
    UROMAN_DIR



from torchaudio.models import wav2vec2_model

logger = logging.getLogger(__name__)


class AlignmentModel:
    """
    MMS Alignment algorithm
    Example usage:

    ```python
    from easymms.models.alignment import AlignmentModel

    align_model = AlignmentModel()
    transcriptions = align_model.align('/home/su/code/easymms/assets/eng_1.mp3',
                                       transcript=["segment 1", "segment 2"],
                                       lang='eng')
    for transcription in transcriptions:
        for segment in transcription:
            print(f"{segment['start_time']} -> {segment['end_time']}: {segment['text']}")
    ```
    """
    def __init__(self,
                 model: str = None,
                 dictionary: str = None,
                 uroman_dir: str = None,
                 log_level: int = logging.INFO):

        set_log_level(log_level)
        assert shutil.which("perl") is not None, "To use the alignment algorithm you will need uroman " \
                                                 "<https://github.com/isi-nlp/uroman> which is written in perl " \
                                                 "please install perl first <https://www.perl.org/get.html>"
        if uroman_dir is not None:
            self.uroman_dir_path = Path(uroman_dir)
        else:
            self.uroman_dir_path = easymms_utils.get_uroman()

        if model is not None:
            self.model_path = Path(model)
        else:
            self.model_path = Path(PACKAGE_DATA_DIR) / "ctc_alignment_mling_uroman_model.pt"

        if dictionary is not None:
            self.dictionary_path = Path(dictionary)
        else:
            self.dictionary_path = Path(PACKAGE_DATA_DIR) / "ctc_alignment_mling_uroman_model.dict"

        self.model, self.dictionary = self._load_model_dict()

        # clone Fairseq
        easymms_utils.clone(constants.FAIRSEQ_URL, constants.FAIRSEQ_DIR)
        sys.path.append(str(constants.FAIRSEQ_DIR.resolve()))

    def _load_model_dict(self):
        """
        Modified from <https://github.com/facebookresearch/fairseq/blob/main/examples/mms/data_prep/align_utils.py>
        to store the models in a consistent directory

        :return: None
        """
        # model_path_name = self.model
        logger.info(f"Loading AlignmentModel model {str(self.model_path.resolve())}")
        if self.model_path.exists():
            logger.info(f"Using model {self.model_path}")
        else:
            logging.info("Downloading alignment model ...")
            torch.hub.download_url_to_file(
                ALIGNMENT_MODEL_URL,
                str(self.model_path.resolve()),
            )
            assert self.model_path.exists()
        state_dict = torch.load(self.model_path, map_location="cpu")

        model = wav2vec2_model(
            extractor_mode="layer_norm",
            extractor_conv_layer_config=[
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
            ],
            extractor_conv_bias=True,
            encoder_embed_dim=1024,
            encoder_projection_dropout=0.0,
            encoder_pos_conv_kernel=128,
            encoder_pos_conv_groups=16,
            encoder_num_layers=24,
            encoder_num_heads=16,
            encoder_attention_dropout=0.0,
            encoder_ff_interm_features=4096,
            encoder_ff_interm_dropout=0.1,
            encoder_dropout=0.0,
            encoder_layer_norm_first=True,
            encoder_layer_drop=0.1,
            aux_num_out=31,
        )
        model.load_state_dict(state_dict)
        model.eval()

        if self.dictionary_path.exists():
            logger.info("Dictionary path already exists.")
        else:
            logger.info("Downloading dictionary ...")
            torch.hub.download_url_to_file(
                ALIGNMENT_DICTIONARY_URL,
                str(self.dictionary_path.resolve()),
            )
            assert self.dictionary_path.exists()
        with open(self.dictionary_path) as f:
            dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}
        return model, dictionary

    def align(self,
              media_file: str,
              transcript: List[str],
              lang: str,
              device: str = None) -> List[dict]:
        """
        Takes a media file, transcription segments and the lang and returns a list of dicts in the following format
        [{
            'start_time': ...
            'end_time': ...,
            'text': ...,
            'duration': ...
        }, ...]

        :param media_file: the path of the media file, should be wav
        :param transcript: list of segments
        :param lang: language ISO code
        :param device: 'cuda' or 'cpu'
        :return: list of transcription timestamps
        """
        # import
        import os
        cwd = os.getcwd()
        os.chdir(constants.FAIRSEQ_DIR)
        from examples.mms.data_prep.align_and_segment import get_alignments
        from examples.mms.data_prep.align_utils import get_uroman_tokens, get_spans
        from examples.mms.data_prep.text_normalization import text_normalize
        os.chdir(cwd)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.model.to(device)
        res = []
        logger.info(f"Aligning file {media_file} ...")
        norm_transcripts = [text_normalize(line.strip(), lang) for line in transcript]
        tokens = get_uroman_tokens(norm_transcripts, str(self.uroman_dir_path.resolve()), lang)

        segments, stride = get_alignments(
            media_file,
            tokens,
            model,
            self.dictionary,
            use_star=False,
        )
        # Get spans of each line in input text file
        spans = get_spans(tokens, segments)

        for i, t in enumerate(transcript):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end
            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000
            res.append({'start_time': audio_start_sec,
                        'end_time': audio_end_sec,
                        'text': t,
                        'duration': audio_end_sec - audio_start_sec})

        return res
