import torch
import util
import argparse
import field_utils
from pathlib import Path


def run(opts):
    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load input pc', count=False):
        input_pc = util.xyz2tensor(open(opts.input, 'r').read(), append_normals=False).to(device)

    with MyTimer('load reference pc', count=False):
        input_reference = util.xyz2tensor(open(opts.reference, 'r') \
                                           .read()).to(device)

    if input_pc.shape[-1] == 3 and opts.estimate_normals:
        with MyTimer('estimating normals'):
            input_pc = util.estimate_normals(input_pc, max_nn=opts.n)

    with MyTimer('calculating field'):
        input_pc = field_utils.reference_field(input_reference, input_pc)

    with MyTimer('export referenced normals', count=False):
        util.export_pc(input_pc.transpose(1, 0), opts.output)

    MyTimer.print_total_time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n', type=int, default=30, help='size of knn for normal estimation')
    parser.add_argument('--estimate_normals', action='store_true', help='estimate normal using pca,'
                                                                        ' or use the field for normal direction'
                                                                        ' as well as orientation ')
    opts = parser.parse_args()
    run(opts)
