#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test the ASRModel, the AlignmentModel will be tested implicitly
"""

from unittest import TestCase

from easymms.models.asr import ASRModel


class TestASR(TestCase):
    asr = ASRModel('../models_dir/mms1b_fl102.pt')
    eng_files = ['../assets/eng_1.mp3', '../assets/eng_2.flac']
    ara_files = ['../assets/ara_1.ogg']

    def test_transcribe_eng(self):
        res = self.asr.transcribe(self.eng_files, align=False, lang='eng')
        self.assertIsNotNone(res)
        self.assertIsInstance(res, list)
        self.assertEqual(len(self.eng_files), len(res))

    def test_transcribe_ara(self):
        """
        Just to try testing with another lang other than `eng`
        :return:
        """
        res = self.asr.transcribe(self.ara_files, align=True, lang='ara')
        self.assertIsNotNone(res)
        self.assertIsInstance(res, list)
        self.assertEqual(len(self.ara_files), len(res))

