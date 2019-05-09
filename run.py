#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from torch import nn, optim
from tqdm import tqdm
from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter
import os, pickle, math, time, torch

if __name__ == "__main__":
    # Set debug to False when training on entire corpus
    # debug = True
    debug = False
    
    # assert(torch.__version__ == "1.0.0"), "Please install torch version 1.0.0"
    
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
    
    