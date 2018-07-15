import argparse


class Config(object):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0)
    args = parser.parse_args()
