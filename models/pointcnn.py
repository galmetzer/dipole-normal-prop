import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
import torch.nn as nn
from torch_geometric.nn import fps, global_mean_pool, knn, knn_graph, knn_interpolate
from models.XConv import XConv
from torch_geometric.data.data import Data
import util as utils


class PointCNN(torch.nn.Module):
    def __init__(self, opts, input_feat, num_classes):
        super(PointCNN, self).__init__()
        self.opts = opts
        self.num_classes = num_classes
        self.input_features = input_feat
        self.encoder = PointCnnEncoderPool(self.input_features, pooling_ratio=self.opts.pool)

        self.output_channels = 2
        self.decoder = PointCnnDecoderPool(self.output_channels)

    def forward(self, data_in: torch.Tensor, batch_index=None):
        '''
        data: a batch of input, torch.Tensor or torch_geometric.data.Data type
            - torch.Tensor: (batch_size, 3, num_points), as common batch input
        '''

        n_points = data_in.shape[0]
        if batch_index is None:
            batch_index = torch.zeros(n_points).long().to(data_in.device)

        pos = data_in
        data = Data()
        data.x, data.pos, data.batch = pos, pos[:, :3].detach(), batch_index

        for i in range(batch_index.max() + 1):
            data.x[batch_index == i, :3] -= data.x[batch_index == i, :3].mean(dim=0)
            data.x[batch_index == i] = utils.rotate_to_principle_components(data.x[batch_index == i])


        if not hasattr(data, 'x'):
            data.x = None

        latent = self.encoder(data)
        x = self.decoder(*latent)

        return x.view(n_points, self.output_channels)


class PointCnnEncoderPool(nn.Module):
    def __init__(self, input_feat, pooling_ratio=0.5):
        super(PointCnnEncoderPool, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.conv1 = XConv(input_feat, 32, dim=3, kernel_size=16, hidden_channels=None)
        self.conv2 = XConv(
            32, 64, dim=3, kernel_size=20, hidden_channels=None, dilation=1)
        self.conv3 = XConv(
            64, 128, dim=3, kernel_size=20, hidden_channels=None, dilation=1)
        self.conv4 = XConv(
            128, 256, dim=3, kernel_size=20, hidden_channels=None, dilation=1)


    def forward(self, data_in):
        x, pos, batch = data_in.x, data_in.pos, data_in.batch
        pos1, batch1 = pos.clone(), batch.clone()
        x = F.relu(self.conv1(x, pos.clone(), batch))

        idx = fps(pos, batch, ratio=self.pooling_ratio, random_start=True)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        pos2, batch2 = pos.clone(), batch.clone()
        x = F.relu(self.conv2(x, pos.clone(), batch))

        idx = fps(pos, batch, ratio=self.pooling_ratio, random_start=True)
        x, pos, batch = x[idx], pos[idx], batch[idx]
        x = F.relu(self.conv3(x, pos.clone(), batch))

        x = F.relu(self.conv4(x, pos, batch))
        return (x, pos, batch), (pos2, pos1), (batch2, batch1)


class PointCnnDecoderPool(nn.Module):
    def __init__(self, input_feat):
        super(PointCnnDecoderPool, self).__init__()
        self.conv1 = XConv(256, 128, dim=3, kernel_size=20, hidden_channels=None)
        self.conv2 = XConv(
            128, 64, dim=3, kernel_size=20, hidden_channels=None, dilation=1)
        self.conv3 = XConv(
            64, 32, dim=3, kernel_size=20, hidden_channels=None, dilation=1)
        self.lin4 = Lin(32, input_feat)

    def forward(self, data_in, poss, batchs):
        x, pos, batch = data_in

        def upsamp(i, x, pos, batch):
            x = knn_interpolate(x, pos, poss[i], batch_x=batch, batch_y=batchs[i], k=16, num_workers=1)
            pos = poss[i]
            batch = batchs[i]
            return x, pos, batch

        x = F.relu(self.conv1(x, pos.clone(), batch))

        x, pos, batch = upsamp(0, x, pos, batch)
        x = F.relu(self.conv2(x, pos.clone(), batch))

        x, pos, batch = upsamp(1, x, pos, batch)
        x = F.relu(self.conv3(x, pos.clone(), batch))

        x = self.lin4(x)
        return x




