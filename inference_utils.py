import torch
from pathlib import Path
from collections import namedtuple
from models.pointcnn import PointCNN


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


def txt2opts(path: Path):
    attr = ['pool']
    opts_dict = {}
    opts = open(path, 'r').read()
    for line in opts.split('\n'):
        line = line.replace(' ', '')
        tokens = line.split(':')
        if tokens[0] in attr:
            if tokens[0] == 'pool':
                val = float(tokens[1])
            else:
                val = tokens[1]

            opts_dict[tokens[0]] = val

    Opts = namedtuple('Opts', opts_dict)
    return Opts(**opts_dict)


def load_model_from_file(file: Path, device):
    opts_file = file.with_suffix('.txt')
    model_opts = txt2opts(opts_file)
    model = PointCNN(model_opts, 6, 16).to(device)
    model.load_state_dict(torch.load(file))
    model.eval()
    return model


def voting_policy(probs):
    probs = torch.stack(probs, dim=0).mean(dim=0)
    return probs < 0.5, probs


def fix_n_filter(input_pc, patch_indices, threshold):
    def criterion(patch):
        x = input_pc[patch]
        temp = x[:, :3] - x.mean(dim=0)[None, :3]
        cov = (temp.transpose(0, 1) @ temp) / x.shape[0]
        e, v = torch.symeig(cov, eigenvectors=True)
        n = v[:, 0]
        return (e[0] / ((e[1] + e[2] / 2))).item() > threshold, n

    new_patches = []
    for i, patch in enumerate(patch_indices):
        flag, n = criterion(patch)
        if flag:
            new_patches.append((i, patch))
        else:
            sign = (input_pc[patch, 3:] * n[None, :]).sum(dim=-1) > 0
            sign = sign.float() * 2 - 1
            input_pc[patch, 3:] = input_pc[patch, 3:] * sign[:, None]

    return new_patches


