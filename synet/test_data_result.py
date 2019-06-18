import os
import pickle

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from synet.models.synet import Block, SYnet
from synet.config import freq_dim, num_labels, model_use_date, \
    model_use_epoch, audio_data_path
from synet.evaluation.evl import *
from synet.train import Net, Data2Torch, model_init


def load_test_data(model_path):
    # load audio data
    data = h5py.File(audio_data_path, 'r')
    # load test index
    test_index_path = os.path.join(model_path, 'test_index.pkl')

    with open(test_index_path, 'rb') as f:
        test_index = pickle.load(f)

    Xte = data['x'][test_index]
    Yte = data['y'][test_index]

    return Xte, Yte

    print('finish dataset loading...')


def model_loading(model_path, model_epoch, device):
    save_dict = torch.load(
        os.path.join(model_path, '_e_%d' % (model_epoch)),
        map_location=device
    )

    model = Net(
        freq_dim=freq_dim,
        num_labels=num_labels,
        avg=None,
        std=None
        ).to(device)

    model.apply(model_init)
    model.load_state_dict(save_dict['state_dict'])
    va_th = save_dict['va_th']

    return model, va_th

    print('finishing model loading')


def main():
    Xte, Yte = load_test_data(model_path=model_path)

    model, va_th = model_loading(
        model_path=model_path,
        model_epoch=model_use_epoch,
        device=device
    )
    # for confusion matrix
    m_tar = []
    m_pre = []

    # predict configure
    v_kwargs = {'batch_size': 10, 'num_workers': 10, 'pin_memory': True}

    loader = torch.utils.data.DataLoader(
        Data2Torch([Xte, Yte]),
        **v_kwargs
        )

    all_pred = np.zeros((Xte.shape[0], num_labels))
    all_tar = np.zeros((Xte.shape[0], num_labels))

    model.eval()

    with torch.no_grad():
        ds = 0
        for idx, _input in enumerate(loader):

            data, target = _input[0].to(device), _input[1].to(device)
            macro_out, micro_out = model(data)

            tar = target.data.cpu().numpy()
            pre = torch.sigmoid(macro_out).data.cpu().numpy()

            for i, (t, p) in enumerate(zip(tar, pre)):
                if t.sum() == 1:

                    m_pre.append(np.argmax(p))
                    m_tar.append(np.argmax(t))

            all_tar[ds: ds + len(target)] = tar
            all_pred[ds: ds + len(target)] = pre

            ds += len(target)
    # print threshold choose by training data's result
    print('threshold choose by training data')
    evl_matrix, va_out = evl(all_tar, all_pred, va_th)
    print(confusion_matrix(m_tar, m_pre))
    print('acc: %f' % accuracy_score(m_tar, m_pre))
    print(np.around(evl_matrix[:, 0], decimals=3))
    print(np.around(evl_matrix[:, 1], decimals=3))
    print(np.around(evl_matrix[:, 2], decimals=3))

    # print threshold choose by testing data's result
    print('threshold choosed by testing data (cheating)')
    va_th, evl_matrix, out = evl_for_th(all_tar, all_pred, [])
    print(np.around(evl_matrix[:, 0], decimals=3))
    print(np.around(evl_matrix[:, 1], decimals=3))
    print(np.around(evl_matrix[:, 2], decimals=3))


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = './trained_model/%s' % (model_use_date)
    main()
