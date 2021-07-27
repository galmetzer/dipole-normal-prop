from options import get_parser
import options
from pathlib import Path
from field_utils import *
torch.manual_seed(1)


def run(opts):
    export_path: Path = opts.export_dir
    export_path.mkdir(exist_ok=True)

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    MyTimer = util.timer_factory()
    with MyTimer('load pc', count=False):
        input_pc = util.xyz2tensor(open(opts.pc, 'r').read()).to(device)

    if opts.estimate_normals:
        with MyTimer('Estimating normals'):
            input_pc = util.estimate_normals(input_pc, max_nn=30)

    input_pc, transform = util.Transform.trans(input_pc)

    with MyTimer('propagating field'):
        strongest_field_propagation_points(input_pc, diffuse=opts.diffuse, starting_point=0)

    with MyTimer('fix global orientation'):
        if measure_mean_potential(input_pc) < 0:
            # if average global potential is negative, flip all normals
            input_pc[:, 3:] *= -1

    with MyTimer('exporting result', count=False):
        util.export_pc(transform.inverse(input_pc).transpose(0, 1), export_path / f'final_result.xyz')

    MyTimer.print_total_time()


if __name__ == '__main__':
    opts = get_parser().parse_args()

    opts.export_dir.mkdir(exist_ok=True, parents=True)

    options.export_options(opts)
    run(opts)
