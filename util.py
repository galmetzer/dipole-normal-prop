import torch
import numpy as np
from typing import List
import time


def gen_grid(n=10):
    index = torch.arange(0, n ** 3)
    z = index % n
    xy = index // n
    y = xy % n
    x = xy // n
    pts = torch.stack([x, y, z], dim=1).float()
    pts = pts / n
    pts -= 0.5
    pts *= 2
    return pts


def orient_center(pred):
    cent = pred[:, :3].mean(dim=0)
    ref = pred[:, :3] - cent
    flip_mask = (ref * pred[:, 3:]).sum(dim=-1) < 0
    pred[flip_mask, 3:] *= -1
    return pred


def export_pc(pc, dest):
    txt = '\n'.join(map(lambda x: ' '.join(map(lambda y: str(y.item()), x)), pc.transpose(0, 1)))
    txt.strip()

    with open(dest, 'w+') as file:
        file.write(txt)


def xyz2tensor(txt, append_normals=True):
    pts = []
    for line in txt.split('\n'):
        line = line.strip()
        spt = line.split(' ')
        if 'nan' in line:
            continue
        if len(spt) == 6:
            pts.append(torch.tensor([float(x) for x in spt]))
        if len(spt) == 3:
            t = [float(x) for x in spt]
            if append_normals:
                t += [0.0 for _ in range(3)]
            pts.append(torch.tensor(t))

    rtn = torch.stack(pts, dim=0)
    return rtn


def divide_pc(pc_in: torch.Tensor, n_part: int, ranges=(-1.5, 1.5),
              min_patch=0) -> (List[torch.Tensor], List[torch.Tensor]):
    '''
    divide a pc into voxel parts
    Args:
        pc_in: input pc (N X (3/6))
        n_part: number of parts in each axis i.e. total num_parts ** 3 parts
        ranges: range of the bounding box the pc is in
        min_patch: join patches with less than min_path points

    Returns: List[torch.Tensor] a list of indices corresponding to each part

    '''
    def mask_to_index(mask, n):
        return torch.arange(n)[mask]

    def bounds(t):
        l = edge_len * t + ranges[0]
        return l, l + edge_len

    pc = pc_in[:, :3]
    num_points = pc.shape[0]
    indices = []
    ijk = []
    edge_len = (ranges[1] - ranges[0]) / (n_part)
    for i in range(n_part + 1):
        x1, x2 = bounds(i)
        x_mask = (x1 < pc[:, 0]) * (pc[:, 0] <= x2)
        for j in range(n_part + 1):
            y1, y2 = bounds(j)
            y_mask = (y1 < pc[:, 1]) * (pc[:, 1] <= y2)
            for k in range(n_part + 1):
                z1, z2 = bounds(k)
                z_mask = (z1 < pc[:, 2]) * (pc[:, 2] <= z2)

                total_mask = x_mask * y_mask * z_mask
                if total_mask.long().sum() > 0:
                    indices.append([mask_to_index(total_mask, num_points)])
                    ijk.append([torch.tensor([i, j, k])])
    indices, ijk = merge_nodes(pc_in[:, :3], indices, ijk, min_patch)
    return indices


def merge_nodes(pts, indices, ijk, min_patch):
    def find_dij(i, ijk1, ijks):
        min_ijks = -1
        for other_index, ijk2 in enumerate(ijks):
            if other_index != i:
                for i_sub in range(len(ijk1)):
                    for j_sub in range(len(ijk2)):
                        dij = (ijk1[i_sub] - ijk2[j_sub])  # look for neighbors
                        if (dij.abs() <= 1).all():
                            min_ijks = other_index
                            break


        return min_ijks

    remaining_small_patches = True
    count = 0
    max_recursive_merges = 10
    while remaining_small_patches and count < max_recursive_merges:
        remaining_small_patches = False
        count += 1
        for i in range(len(ijk)):
            if len(indices[i]) > 0 and len(indices[i][0]) < min_patch:
                if len(ijk[i]) > 0:
                    min_j = find_dij(i, ijk[i], ijk)
                    if min_j != -1:
                        indices[min_j][0] = torch.cat([indices[min_j][0], indices[i][0]])
                        for t in range(len(ijk[i])):
                            ijk[min_j].append(ijk[i][t])
                        indices[i] = []
                        ijk[i] = []
                        if len(indices[min_j][0]) < min_patch:
                            remaining_small_patches = True

    if count == max_recursive_merges:
        print('recursive merge failed to merge some patches')

    new_indices = []
    new_ijk = []
    for i in range(len(ijk)):
        if len(ijk[i]) > 0 and len(indices[i][0]) >= min_patch:
            new_indices.append(torch.cat(indices[i]))
            new_ijk.append(ijk[i])

    return new_indices, new_ijk


def pca_eigen_values(x: torch.Tensor):
    temp = x[:, :3] - x.mean(dim=0)[None, :3]
    cov = (temp.transpose(0, 1) @ temp) / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    n = v[:, 0]
    return e[0:1], n


def rotate_to_principle_components(x: torch.Tensor, scale=True):
    temp = x[:, :3] - x.mean(dim=0)[None, :3]
    cov = temp.transpose(0, 1) @ temp / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)

    # rotate xyz
    rotated = x[:, :3]@v
    if scale:
        # scale to unit var on for the larger eigen value
        rotated = rotated / torch.sqrt(e[2])

    # if x contains normals rotate the normals as well
    if x.shape[1] == 6:
        rotated = torch.cat([rotated, x[:, 3:]@v], dim=-1)
    return rotated


def estimate_normals_torch(inputpc, max_nn):
    from torch_cluster import knn_graph
    knn = knn_graph(inputpc[:, :3], max_nn, loop=False)
    knn = knn.view(2, inputpc.shape[0], max_nn)[0]
    x = inputpc[knn][:, :, :3]
    temp = x[:, :, :3] - x.mean(dim=1)[:, None, :3]
    cov = temp.transpose(1, 2) @ temp / x.shape[0]
    e, v = torch.symeig(cov, eigenvectors=True)
    n = v[:, :, 0]
    return torch.cat([inputpc[:, :3], n], dim=-1)


def estimate_normals(inputpc, max_nn=30, keep_orientation=False):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        xyz = np.array(inputpc[:, :3].cpu())
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=max_nn))
        normals = np.array(pcd.normals)
        inputpc_unoriented = torch.cat((inputpc[:, :3], torch.Tensor(normals).to(inputpc.device)), dim=1)
        if keep_orientation:
            flip = (inputpc[:, 3:] * inputpc_unoriented[:, 3:]).sum(dim=-1) < 0
            inputpc_unoriented[flip, 3:] *= -1
    except ModuleNotFoundError:
        inputpc_unoriented = estimate_normals_torch(inputpc, max_nn)

    return inputpc_unoriented


class Transform:
    def __init__(self, pc: torch.Tensor, ttype='reg'):
        if ttype == 'reg':
            self.center = pc[:, :3].mean(dim=0)
            self.scale = (pc[:, :3].max(dim=0)[0] - pc[:, :3].min(dim=0)[0]).max()
        elif ttype == 'bb':
            self.center = pc[:, :3].mean(dim=0)
            pc_tag = pc[:, :3] - self.center
            d = pc[:, :3].sum(dim=-1)
            a, b = d.argmin(), d.argmax()
            line = pc_tag[b] - pc_tag[a]
            self.scale = line.norm()
            mid_points = (pc_tag[a] + pc_tag[b]) / 2
            self.center += mid_points

    def apply(self, pc: torch.Tensor) -> torch.Tensor:
        pc = pc.clone()
        pc[:, :3] -= self.center[None, :]
        pc[:, :3] = pc[:, :3] / self.scale
        return pc

    def inverse(self, pc: torch.Tensor) -> torch.Tensor:
        pc = pc.clone()
        pc[:, :3] = pc[:, :3] * self.scale
        pc[:, :3] += self.center[None, :]
        return pc

    @staticmethod
    def trans(pc: torch.Tensor, ttype='reg'):
        T = Transform(pc, ttype=ttype)
        return T.apply(pc), T


def timer_factory():
    class MyTimer(object):
        total_count = 0

        def __init__(self, msg='', count=True):
            self.msg = msg
            self.count = count

        def __enter__(self):
            self.start = time.clock()
            if self.msg:
                print(f'started: {self.msg}')
            return self

        def __exit__(self, typ, value, traceback):
            self.duration = time.clock() - self.start
            if self.count:
                MyTimer.total_count += self.duration
            if self.msg:
                print(f'finished: {self.msg}. duration: {MyTimer.convert_to_time_format(self.duration)}')

        @staticmethod
        def print_total_time():
            print('\n ----- \n')
            print(f'total time: {MyTimer.convert_to_time_format(MyTimer.total_count)}')

        @staticmethod
        def convert_to_time_format(sec):
            sec = round(sec, 2)
            if sec < 60:
                return f'{sec} [sec]'

            minutes = int(sec / 60)
            remaining_seconds = sec - (minutes * 60)
            remaining_seconds = round(remaining_seconds, 2)
            return f'{minutes}:{remaining_seconds} [min:sec]'

    return MyTimer

