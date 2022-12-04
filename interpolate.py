import argparse
from create_parser import create_parser
from exp import Exp

import warnings
warnings.filterwarnings('ignore')

def main():
    args = create_parser().parse_args()

    if not args.resume_path:
        raise ValueError('Missing location of pre-train model')

    exp = Exp(args)
    exp.interpolate()

if __name__ == '__main__':
    main()