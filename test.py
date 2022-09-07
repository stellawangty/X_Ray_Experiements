import os
import numpy as np
import time
import sys
import argparse
from ChexnetTrainer import ChexnetTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default='dataset/test_1.txt', help='pathFileTest')
    parser.add_argument('--class_count', default=14, help='nnClassCount')
    parser.add_argument('--batch_size', type=int, default=16, help='trBatchSize')
    parser.add_argument('--crop_size', type=int, default=224, help='imgtransCrop')
    parser.add_argument('--time_stamp', default='', help='timestampLaunch')
    parser.add_argument('--checkpoint', default=None, help='if not None loads the model and continues training')
    opt = parser.parse_args()
    print(opt)

    ChexnetTrainer.test('',
                        None,
                        opt.test_file,
                        None,
                        None,
                        opt.class_count,
                        opt.batch_size,
                        None,
                        None,
                        opt.crop_size,
                        None,
                        opt.checkpoint,
                        )

