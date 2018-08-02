import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--extend", type=bool, default=True)
    parser.add_argument("--gpu_count", default=1, type=int)
    parser.add_argument("--gpu1", default='0', type=str)
    parser.add_argument("--gpu2", default='1', type=str)
    parser.add_argument("--gpu3", default='2', type=str)
    parser.add_argument("--gpu4", default='3', type=str)
    parser.add_argument("--image_per_gpu", type=int, default=1)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--det_loss_weight", default=1, type=int)
    parser.add_argument("--type", default='focal')
    parser.add_argument("--model_loss", default='focal_double', type=str)
    parser.add_argument("--summary", default=True, type=bool)
    parser.add_argument("--loss_backend", default='keras')
    parser.add_argument("--backbone", default='resnet50', type=str)
    parser.add_argument("--test_img", default=15, type=int)
    parser.add_argument("--data", default='crop', type=str)
    parser.add_argument("--det_weight", type=float, default=0.1)
    args = parser.parse_args()
    det_weight = args.det_weight
    backbone = args.backbone
    data=args.data
    loss_backend = args.loss_backend
    summary = args.summary
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    gpu_count = args.gpu_count
    gpu1 = args.gpu1
    gpu2 = args.gpu2
    gpu3 = args.gpu3
    gpu4 = args.gpu4
    detection_loss_weight = args.det_loss_weight
    model_loss = args.model_loss
    type = args.type
    test_img = args.test_img
    extend_program = args.extend
