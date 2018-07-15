import argparse


class Config(object):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--image_per_gpu", default=1)
    parser.add_argument("--epoch", default=300)
    args = parser.parse_args()
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    gpu_count = args.gpu_count