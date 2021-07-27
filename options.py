import argparse
from pathlib import Path


def get_parser(name='Base Options') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(name)
    parser.add_argument('--export_dir', type=Path, required=True, help='export directory')
    parser.add_argument('--propagation_iters', default=10, type=int, help='test epochs')
    parser.add_argument('--number_parts', type=int, default=15)
    parser.add_argument('--minimum_points_per_patch', type=int, default=21)
    parser.add_argument('--curvature_threshold', default=0.0, type=float)
    parser.add_argument('--pc', type=Path, required=True, help='pc to read')
    parser.add_argument('--models', nargs='+', type=Path, default=[], help='path to trained models')
    parser.add_argument('--iters', default=100, type=int, help='iters to optimize')
    parser.add_argument('--diffuse', action='store_true')
    parser.add_argument('--weighted_prop', action='store_true')
    parser.add_argument('--estimate_normals', action='store_true')
    parser.add_argument('--n', type=int, default=30, help='size of knn for normal estimation')

    return parser


def export_options(opts):
    def args_to_str(args):
        d = args.__dict__
        txt = ''
        for k in d.keys():
            txt += f'{k}: {d[k]}\n'
        return txt.strip('\n')
    txt = args_to_str(opts)
    with open(opts.export_dir / 'opts.txt', 'w+') as file:
        file.write(txt)