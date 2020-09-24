import argparse
from smp_manifold_learning.differentiable_models.utils import create_dir_if_not_exist


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("-d", "--dir_path", default='../plot/ecmnn/', type=str)


if __name__ == '__main__':
    args = parser.parse_args()
    dir_path = args.dir_path
    create_dir_if_not_exist(dir_path)