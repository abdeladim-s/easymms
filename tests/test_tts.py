#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from unittest import TestCase

from easymms.models.tts import TTSModel


class TestTTSModelEng(TestCase):
    lang = 'eng'
    tts = TTSModel(lang)

    def test_synthesize(self):
        res = self.tts.synthesize("This is a simple example")
        self.assertIsInstance(res, tuple)
        self.assertEquals(len(res), 2)

    def test_save(self):
        res = self.tts.synthesize("This is a simple example")
        out_file = self.tts.save(res)
        self.assertIsNotNone(out_file)
        self.assertTrue(Path(out_file).exists())

class TestTTSModelAra(TestCase):
    lang = 'ara'
    tts = TTSModel(lang)

    def test_synthesize(self):
        res = self.tts.synthesize("هذا مثال بسيط")
        self.assertIsInstance(res, tuple)
        self.assertEquals(len(res), 2)

    def test_save(self):
        res = self.tts.synthesize("هذا مثال بسيط")
        out_file = self.tts.save(res)
        self.assertIsNotNone(out_file)
        self.assertTrue(Path(out_file).exists())
