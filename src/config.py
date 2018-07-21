import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--image_per_gpu", default=1)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--det_loss_weight", default=1)
    parser.add_argument("--type", default='focal')
    parser.add_argument("--model_loss", default='focal', type=str)
    parser.add_argument("--summary", default=True)
    parser.add_argument("--loss_backend", default='keras')
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--test_img", default=15, type=int)
    args = parser.parse_args()
    backbone = args.backbone
    loss_backend = args.loss_backend
    summary = args.summary
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    gpu_count = args.gpu_count
    gpu = args.gpu
    detection_loss_weight = args.det_loss_weight
    model_loss = args.model_loss
    type = args.type
    test_img = args.test_img
