#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, logging, torch
import numpy as np
from collections import Counter
from . general_utils import get_minibatches
from parser_transitions import minibatch_parse
from tqdm import tqdm

class Config(object):
    language = 'english'

def load_and_preprocess_data(reduced=True):
    config = Config()

if __name__ == '__main__':
    pass