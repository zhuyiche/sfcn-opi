import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--image_per_gpu", default=1)
    parser.add_argument("--epoch", default=300)
    parser.add_argument("--det_loss_weight", default=1)
    args = parser.parse_args()
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    gpu_count = args.gpu_count
    gpu = args.gpu
    detection_loss_weight = args.det_loss_weight
