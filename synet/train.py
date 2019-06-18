import datetime
import os
import pickle
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import sigmoid
from torch.utils.data import Dataset
import torch.nn.init as init
from tqdm import tqdm

from synet.evaluation.evl import *
from synet.models.synet import Block, SYnet
from synet.config import b_size, num_labels, freq_dim, epoch_size,\
    audio_data_path


date = datetime.datetime.now()


def get_row_avg_std(data):
    print('Get std and average')
    common_sum = 0
    square_sum = 0

    # remove zero padding
    tfle = 0
    for i in range(len(data)):
        tfle += (data[i].sum(0) != 0).astype('int').sum()
        common_sum += data[i].sum(-1)
        square_sum += (data[i] ** 2).sum(-1)

    common_avg = common_sum / tfle
    square_avg = square_sum / tfle
    std = np.sqrt(square_avg - common_avg ** 2)

    return common_avg, std


def get_weighted_loss(data, beta):
    class_prior = data[:].sum(0) / data[:].sum()
    mean_prior = class_prior.mean()

    loss_weight = (
        (mean_prior / class_prior) * (1 - mean_prior) / (1 - class_prior)
        ) ** beta

    return loss_weight


def model_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)

    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

    elif classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))


def load_audio(test_index_save_path):
    tr_data = h5py.File(audio_data_path, 'r')

    data_length = len(tr_data['x'])
    np.random.seed(19941115)
    index = np.random.permutation(data_length)
    train_index = list(np.sort(index[1000:]))
    test_index = list(np.sort(index[:1000]))

    Xtr = tr_data['x'][train_index]
    Ytr = tr_data['y'][train_index]
    Xva = tr_data['x'][test_index]
    Yva = tr_data['y'][test_index]

    avg, std = get_row_avg_std(Xtr)
    loss_weight = get_weighted_loss(Ytr, beta=1)

    test_index_save_path = os.path.join(test_index_save_path, 'test_index.pkl')

    with open(test_index_save_path, 'wb') as f:
        pickle.dump(test_index, f)

    return Xtr, Ytr, Xva, Yva, avg, std, loss_weight


class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.Y = data[1]

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()

        return mX, mY

    def __len__(self):

        return len(self.X)


class Net(nn.Module):
    def __init__(
            self, freq_dim, num_labels, avg=None, std=None):
        super(Net, self).__init__()

        self.model = SYnet(freq_dim, num_labels)

        if avg is not None:
            self.avg = nn.Parameter(torch.from_numpy(avg).float(),
                                    requires_grad=False
                                    )

        else:
            self.avg = nn.Parameter(torch.zeros(freq_dim).float(),
                                    requires_grad=False
                                    )

        if std is not None:
            self.std = nn.Parameter(torch.from_numpy(std).float(),
                                    requires_grad=False
                                    )

        else:
            self.std = nn.Parameter(torch.ones(freq_dim).float(),
                                    requires_grad=False
                                    )

    def forward(self, x):
        batch, freq_dim, length = x.size()
        avg = self.avg.view(1, freq_dim, 1).expand(batch, -1, length)
        std = self.std.view(1, freq_dim, 1).expand(batch, -1, length)
        x = (x - avg) / std

        macro_out, micro_out = self.model(x)

        return macro_out, micro_out


def mm_loss(target, macro_out, micro_out, loss_weight):
    loss_weight = torch.from_numpy(loss_weight).float().to(device)
    loss = 0
    loss_weight = loss_weight.view(1, -1).repeat(target.size(0), 1)

    loss_fn = torch.nn.BCEWithLogitsLoss(weight=loss_weight)
    loss += loss_fn(macro_out, target)

    for att_sc, det in micro_out:
        det_size = det.size()

        fl_target = target.view(
            det_size[0], det_size[1], 1).repeat(1, 1, det_size[2])

        loss_weight = loss_weight.view(
            det_size[0], det_size[1], 1
            ).repeat(1, 1, det_size[2])

        loss_weight = att_sc * loss_weight
        loss_weight = loss_weight.detach()

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=loss_weight)
        loss += loss_fn(det, fl_target)

    return loss


class Trainer:
    def __init__(self, model, lr, epoch, save_fn, device, loss_weight):
        self.epoch = epoch
        self.lr = lr
        self.save_fn = save_fn
        self.loss_weight = loss_weight
        self.model = model.to(device)

    def tester(self, loader, size, th, generate_th):
        all_pred = np.zeros((size, num_labels))
        all_tar = np.zeros((size, num_labels))

        self.model.eval()

        with torch.no_grad():
            ds = 0
            for idx, _input in enumerate(loader):

                data = _input[0].to(device)
                target = _input[1].to(device)

                pred, frame_pred = self.model(data)

                all_tar[ds: ds + len(target)] = target.cpu().numpy()
                all_pred[ds: ds + len(target)] = sigmoid(pred).cpu().numpy()
                ds += len(target)

            if generate_th:
                va_th, evl_metrics, va_out = evl_for_th(all_tar, all_pred, th)

                return va_th, evl_metrics, va_out

            else:
                evl_metrics, va_out = evl(all_tar, all_pred, th)

        return evl_metrics, va_out

# random.shuffle not return value, that a problem!!
    def downsampling(self, data, target):
        happys = []
        aggressives = []
        sads = []
        calms = []

        for i, y in enumerate(target):
            if y[0] == 1:
                happys.append(i)

            if y[1] == 1:
                aggressives.append(i)

            if y[2] == 1:
                sads.append(i)

            if y[3] == 1:
                calms.append(i)

        label_lens = [len(x) for x in [happys, aggressives, sads, calms]]
        sample_size = min(label_lens)
        random.shuffle(happys)
        random.shuffle(aggressives)
        random.shuffle(sads)
        random.shuffle(calms)

        sample_indexs = happys[:sample_size] + aggressives[:sample_size] +\
            sads[:sample_size] + calms[:sample_size]
        random.shuffle(sample_indexs)

        data = data[sample_indexs]
        target = target[sample_indexs]

        return data, target

    def fit(self, tr_loader, va_loader):
        save_dict = {}

        def lambda_for_optim(epoch):
            return self.lr ** ((epoch / 50) + 1) / self.lr

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr, momentum=0.9,
            weight_decay=1e-4
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_for_optim)

        for e in range(1, self.epoch + 1):
            loss_total = 0

            with tqdm(total=len(tr_loader)) as t:
                t.set_description('Epoch %2d/%2d' % (e, self.epoch))

                for batch_idx, _input in enumerate(tr_loader):
                    data = _input[0]
                    target = _input[1]

                    # data, target = self.downsampling(data, target)
                    data = data.to(device)
                    target = target.to(device)
                    macro_out, micro_out = self.model(data)

                    loss = mm_loss(
                        target,
                        macro_out,
                        micro_out,
                        self.loss_weight
                        )

                    loss_total += loss.data
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t.update(1)
                    t.set_postfix(loss='%.4f' % loss.data)

            scheduler.step()

            va_th, evl_metrics, va_out = self.tester(
                tr_loader, len(tr_loader.dataset), [], True)

            evl_metrics, va_out = self.tester(
                va_loader, len(va_loader.dataset), va_th, False)

            print(np.around(evl_metrics[:, 0], decimals=3))
            save_dict['state_dict'] = self.model.state_dict()
            save_dict['va_out'] = va_out
            save_dict['va_th'] = va_th
            save_dict['evl_metrics'] = evl_metrics
            torch.save(save_dict, os.path.join(self.save_fn, '_e_%d' % (e)))


def main():
    # load data
    t_kwargs = {'batch_size': b_size, 'num_workers': 0,
                'pin_memory': False, 'drop_last': True}

    v_kwargs = {'batch_size': b_size, 'num_workers': 0, 'pin_memory': False}

    out_model_fn = os.path.join(
        './trained_model',
        '%d%d%d' % (date.year, date.month, date.day)
        )

    if not os.path.exists(out_model_fn):
        os.makedirs(out_model_fn)

    Xtr, Ytr, Xva, Yva, avg, std, loss_weight = load_audio(
        test_index_save_path=out_model_fn
        )

    trdata = [Xtr, Ytr]
    vadata = [Xva, Yva]

    tr_loader = torch.utils.data.DataLoader(
        Data2Torch(trdata),
        shuffle=True,
        **t_kwargs
        )

    va_loader = torch.utils.data.DataLoader(Data2Torch(vadata), **v_kwargs)
    print('finishing data building...')

    # load model
    model = Net(
        freq_dim=freq_dim,
        num_labels=num_labels,
        avg=avg,
        std=std
        )

    model.apply(model_init)
    print(model)
    print('finishing model loading...')

    # start training
    trer = Trainer(
        model, 0.01, epoch_size, out_model_fn, device, loss_weight
        )

    trer.fit(tr_loader, va_loader)

    print(out_model_fn)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main()
