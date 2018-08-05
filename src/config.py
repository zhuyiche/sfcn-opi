import argparse


class Config(object):
    """
    Manually setting configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_count", default=1, type=int, help='set gpu count above 1 if you want to train multi gpus.')
    parser.add_argument("--gpu1", default='0', type=str)
    parser.add_argument("--gpu2", default='1', type=str)
    parser.add_argument("--gpu3", default='2', type=str)
    parser.add_argument("--gpu4", default='3', type=str)
    parser.add_argument("--image_per_gpu", type=int, default=1, help='set batch size depending on gpu.')
    parser.add_argument("--epoch", default=300, type=int, help='number of epoch to train, same for each process.')
    parser.add_argument("--type", default='focal')
    parser.add_argument("--summary", default=True, type=bool, help='check if we use the summary')
    parser.add_argument("--backbone", default='resnet50', type=str, help='current only support for resnet50')
    parser.add_argument("--test_img", default=15, type=int)
    parser.add_argument("--lr", default=0.01, type=int, help='learning rate for model')
    args = parser.parse_args()
    backbone = args.backbone
    loss_backend = args.loss_backend
    summary = args.summary
    image_per_gpu = args.image_per_gpu
    epoch = args.epoch
    gpu_count = args.gpu_count
    lr = args.lr
    gpu1 = args.gpu1
    gpu2 = args.gpu2
    gpu3 = args.gpu3
    gpu4 = args.gpu4
    type = args.type
    test_img = args.test_img
    extend_program = args.extend
