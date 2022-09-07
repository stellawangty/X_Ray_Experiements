import os
import numpy as np
import time
import sys
import argparse
from ChexnetTrainer import ChexnetTrainer

# ---- Test the trained network
# ---- pathDirData - path to the directory that contains images
# ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
# ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
# ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
# ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
# ---- nnClassCount - number of output classes
# ---- trBatchSize - batch size
# ---- trMaxEpoch - number of epochs
# ---- transResize - size of the image to scale down to (not used in current implementation)
# ---- transCrop - size of the cropped image
# ---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
# ---- checkpoint - if not None loads the model and continues training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', help='pathDirData, the absolute path of root directory of chexnet')
    parser.add_argument('--train_file', default='dataset/train.txt', help='pathFileTrain')
    parser.add_argument('--val_file', default='dataset/val.txt', help='pathFileVal')
    parser.add_argument('--model_name', default='DENSE-NET-121', help='nnArchitecture')
    parser.add_argument('--isTrained', default=True, help='nnIsTrained')
    parser.add_argument('--class_count', default=14, help='nnClassCount')
    parser.add_argument('--batch_size', type=int, default=16, help='trBatchSize')
    parser.add_argument('--epochs', type=int, default=100, help='trMaxEpoch')
    parser.add_argument('--resize', type=int, default=256, help='imgtransResize')
    parser.add_argument('--crop_size', type=int, default=224, help='imgtransCrop')
    parser.add_argument('--model_path', default=None, help='pathModel')
    parser.add_argument('--time_stamp', default='', help='timestampLaunch')
    parser.add_argument('--model_save_dir', default='output', help='model_save_dir')
    parser.add_argument('--checkpoint', default=None, help='if not None loads the model and continues training')

    opt = parser.parse_args()
    print(opt)

    opt.time_stamp = time.strftime("%Y%m%d_%H%M%S")
    opt.model_path = 'm_' + opt.time_stamp + '.pth.tar'
    opt.data_path = os.path.abspath('.')
    # opt.data_path = '/home/q/Downloads/Fw_batch_download_zips/chexnet'

    print('Training NN architecture = ', opt.model_name)
    ChexnetTrainer.train(opt.data_path,
                         opt.train_file,
                         opt.val_file,
                         opt.model_name,
                         opt.isTrained,
                         opt.class_count,
                         opt.batch_size,
                         opt.epochs,
                         opt.resize,
                         opt.crop_size,
                         opt.time_stamp,
                         opt.checkpoint,
                         opt.model_save_dir
                         )

