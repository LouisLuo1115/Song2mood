import torch.nn as nn
import torch.nn.functional as F

batchNorm_momentum = 0.1


class Block(nn.Module):
    def __init__(self, inp, out):

        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm1d(inp)
        self.conv1 = nn.Conv1d(inp, out, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out)
        self.conv2 = nn.Conv1d(out, out, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(out)
        self.sk = nn.Conv1d(inp, out, 1, padding=0)

    def forward(self, x):
        out = self.bn1(x)
        bn1 = F.relu(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)

        return out, bn1


class SYnet(nn.Module):
    def __init__(self, freq_dim, num_labels):
        super(SYnet, self).__init__()
        self.freq_dim = freq_dim
        self.nc = num_labels
        self.kernel_size = 3
        self.padding_size = 1

        self.bn = nn.BatchNorm2d(self.freq_dim)

        self.conv1 = nn.Conv1d(
            self.freq_dim,
            self.freq_dim,
            self.kernel_size,
            padding=self.padding_size
            )

        self.block1 = Block(self.freq_dim, self.freq_dim*2)
        self.block2 = Block(self.freq_dim*2, self.freq_dim*3)
        self.block3 = Block(self.freq_dim*3, self.freq_dim*3)
        self.bnf = nn.BatchNorm1d(self.freq_dim*3)

        self.kernel_size = 1
        self.padding_size = 0

        self.det = nn.Conv1d(
            self.freq_dim*3,
            self.nc,
            self.kernel_size,
            padding=self.padding_size
            )

        self.att = nn.Conv1d(
            self.freq_dim*3,
            self.freq_dim*3,
            self.kernel_size,
            padding=self.padding_size
            )

        self.dropout = nn.Dropout(.0)

        hidden_layer = 512*2

        # linear
        self.den1 = nn.Linear(self.freq_dim*3, hidden_layer)
        self.den2 = nn.Linear(hidden_layer, hidden_layer)
        self.dbn1 = nn.BatchNorm1d(hidden_layer)
        self.dbn2 = nn.BatchNorm1d(hidden_layer)
        self.pred = nn.Linear(hidden_layer, self.nc)

    def nn_apl(self, x):

        return F.avg_pool1d(x, x.size()[2]).view(x.size()[0], -1)

    def nn_att(self, input, att):
        att_out = F.softmax(att(input), dim=2)
        att_sc = att_out.sum(dim=1).view(input.size(0), 1, input.size(2))
        att_sc = att_sc.repeat(1, self.nc, 1)

        return att_sc

    def forward(self, x):
        conv1 = self.conv1(x)

        b1, bnb1 = self.block1(conv1)
        mp1 = F.max_pool1d(b1, 3)

        b2, bnb2 = self.block2(mp1)
        mp2 = F.max_pool1d(b2, 3)

        b3, b3bn = self.block3(mp2)
        bf = F.relu(self.bnf(b3))

        # global average pooling
        gap = self.nn_apl(bf)

        # DNN
        den1 = F.relu(self.dbn1(self.den1(self.dropout(gap))))
        den2 = F.relu(self.dbn2(self.den2(self.dropout(den1))))
        y_cls = self.pred(self.dropout(den2))

        # attention
        att = self.nn_att(bf, self.att)
        det = self.det(bf)

        # ensemble prediction
        att_ens = F.softmax(att, dim=2)
        y_att = (det * att_ens).sum(-1)
        y_ens = (y_cls + y_att) / 2

        return y_ens, [[att, det]]
