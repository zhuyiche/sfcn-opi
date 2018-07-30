import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2
import scipy.io as sio
from config import Config
from metric import non_max_suppression, get_metrics
from util import load_data

epsilon = 1e-7
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith('src'):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'CRCHistoPhenotypes_2016_04_28', 'cls_and_det')
CROP_DATA_DIR = os.path.join(ROOT_DIR, 'crop_cls_and_det')
TENSORBOARD_DIR = os.path.join(ROOT_DIR, 'tensorboard_logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'model_weights')


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Identity_block_2(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(Identity_block_2, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Convolution_block_2(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1):
        super(Convolution_block_2, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(inplanes, outplanes, stride=2),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            conv3x3(outplanes, outplanes),
            nn.BatchNorm2d(outplanes)
        )
        self.residual_conv = nn.Sequential(
            conv3x3(inplanes, outplanes, stride=1),
            nn.BatchNorm2d(outplanes)
        )
        self.relu = nn.ReLU(inplace = True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv(x)

        residual = self.residual_conv(residual)
        out += residual
        out = self.relu(out)
        return out


class ResNet_Block(nn.Module):
    def __init__(self, inplanes, outplanes, downsample=False):
        super(ResNet_Block, self).__init__()
        if downsample:
            self.conv1 = Convolution_block_2(inplanes, outplanes)
        else:
            self.conv1 = Identity_block_2(inplanes, outplanes)
        self.conv_repeat = nn.Sequential(
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
            Identity_block_2(outplanes, outplanes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_repeat(out)
        return out


class FCN_36_Module(nn.Module):
    def __init__(self):
        super(FCN_36_Module, self).__init__()
        self.resnet = nn.Sequential(
            conv3x3(2, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResNet_Block(32, 32, downsample=False),
            ResNet_Block(32, 64, downsample=True),
            ResNet_Block(64, 128, downsample=True),
            ResNet_Block(128, 256, downsample=True)
        )

    def forward(self, x):
        out = self.resnet(x)
        return out


def MyMetrics(model):
    path = './CRCHisto'
    tp_num = 0
    gt_num = 0
    pred_num = 0
    precision = 0
    recall = 0
    f1_score = 0

    for i in range(81, 101):
        filename = os.path.join(path, 'img' + str(i) + '.bmp')
        if os.path.exists(filename):
            gtpath = './CRCHistoPhenotypes_2016_04_28/Detection'
            imgname = 'img' + str(i)
            img = misc.imread(filename)
            img = misc.imresize(img, (256, 256))
            img = img - 128.
            img = img / 128.
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = np.transpose(img, (0, 3, 1, 2))
            img = torch.Tensor(img).cuda()
            result = model(img)
            result = result.cpu().detach().numpy()
            result = np.transpose(result, (0, 2, 3, 1))[0]
            result = np.exp(result)
            result = result[:, :, 1]
            result = misc.imresize(result, (500, 500))
            result = result / 255.
            boxes = non_max_suppression(result)
            matname = imgname + '_detection.mat'
            matpath = os.path.join(gtpath, imgname, matname)
            gt = sio.loadmat(matpath)['detection']
            pred = []
            for i in range(boxes.shape[0]):
                x1 = boxes[i, 0]
                y1 = boxes[i, 1]
                x2 = boxes[i, 2]
                y2 = boxes[i, 3]
                cx = int(x1 + (x2 - x1) / 2)
                cy = int(y1 + (y2 - y1) / 2)
                pred.append([cx, cy])
            p, r, f1, tp = get_metrics(gt, pred)
            tp_num += tp
            gt_num += gt.shape[0]
            pred_num += np.array(pred).shape[0]
    precision = tp_num / (pred_num + epsilon)
    recall = tp_num / (gt_num + epsilon)
    f1_score = 2 * (precision * recall / (precision + recall + epsilon))

    return precision, recall, f1_score


def gpu_cuda(model, train, valid, weight):
    model_gpu = model.cuda()
    train_gpu = train.cuda()
    valid_gpu = valid.cuda()
    weight_gpu = weight.cuda()
    return model_gpu, train_gpu, valid_gpu, weight_gpu


def data_prepare(batch_size, print_img_shape=True):
    train_imgs, train_det_masks, train_cls_masks = load_data(data_path=DATA_DIR, type='train',
                                                             reshape_size=(512, 512))
    valid_imgs, valid_det_masks, valid_cls_masks = load_data(data_path=DATA_DIR, type='validation',
                                                             reshape_size=(512, 512))
    test_imgs, test_det_masks, test_cls_masks = load_data(data_path=DATA_DIR, type='test',
                                                          reshape_size=(512, 512))
    if print_img_shape:
        print('Image shape print below: ')
        print('train_imgs: {}, train_det_masks: {}'.format(train_imgs.shape, train_det_masks.shape))
        print('valid_imgs: {}, valid_det_masks: {}'.format(valid_imgs.shape, valid_det_masks.shape))
        print('test_imgs: {}, test_det_masks: {}'.format(test_imgs.shape, test_det_masks.shape))
        print()

    valid_step = int(len(valid_imgs)/batch_size)
    _train = np.concatenate([train_imgs, train_det_masks], axis=1)
    _train = torch.Tensor(_train)
    _valid = np.concatenate([valid_imgs, valid_det_masks], axis=1)
    _valid = torch.Tensor(_valid)

    _train = _train.cuda()
    _valid = _valid.cuda()

    train_loader = torch.utils.data.DataLoader(_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(_valid, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, valid_step


def data_unpack(loader):
    images = loader[:, 0:3, :, :]
    masks = loader[:, 3:, :, :]
    masks = masks.long()
    masks = masks.view(masks.size()[0], masks.size()[2], masks.size()[3])
    return images, masks


def train(model, weight=None, gpu=True, batch_size=2, num_epochs=100):
    if weight == None:
        weight = torch.Tensor([1, 1])
    else:
        weightid = (str(weight[1])).split('.')[-1]
        weight = torch.Tensor(weight)


    writer = SummaryWriter()

    train_loader, val_loader, val_steps = data_prepare(batch_size, print_img_shape=True)

    model = model.cuda()
    weight = weight.cuda()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss = nn.NLLLoss(weight=weight)
    best_loss = 99999.0
    best_f1 = 0.0

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs, train_masks = data_unpack(datapack)

            optimizer.zero_grad()
            train_out = model(train_imgs)
            t_loss = NLLLoss(train_out, train_masks)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                writer.add_scalar('train_loss', train_loss / 10, (105 * epoch + i + 1))
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs, val_masks = data_unpack(datapack)

            # optimizer.zero_grad()
            val_out = model(val_imgs)
            v_loss = NLLLoss(val_out, val_masks)
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                '''
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_name = './ckpt/net_4_w' + weightid + '_di.pkl'
                    torch.save(model.state_dict(), save_name)
                '''
                end = time.time()
                time_spent = end - start
                writer.add_scalar('val_loss', val_loss, epoch)
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss = 0.0
                p, r, f = MyMetrics(model)
                if f > best_f1:
                    best_f1 = f
                    save_name = './ckpt/net_di_m2_n4_w' + weightid + '.pkl'
                    torch.save(model.state_dict(), save_name)
                writer.add_scalar('precision', p, epoch)
                writer.add_scalar('recall', r, epoch)
                writer.add_scalar('f1_score', f, epoch)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                print('******************************************************************************')

    # writer.export_scalars_to_json('./loss.json')
    writer.close()


def train_2(model, weight=None, data_dir='', preprocess=True, gpu=True, batch_size=2, num_epochs=100):
    if weight == None:
        weight = torch.Tensor([1, 1])
    else:
        weightid = (str(weight[1])).split('.')[-1]
        weight = torch.Tensor(weight)


    writer = SummaryWriter()

    data = dataset.CRC_softmask(data_dir, target_size=256)
    x_train, y_train = data.load_train(preprocess=preprocess)
    train_count = len(x_train)

    x_val, y_val = data.load_val(preprocess=preprocess)
    val_count = len(x_val)
    val_steps = int(val_count / batch_size)
    print('training imgs:', train_count)
    print('val imgs:', val_count)

    trainset = np.concatenate([x_train, y_train], axis=1)
    trainset = torch.Tensor(trainset)

    valset = np.concatenate([x_val, y_val], axis=1)
    valset = torch.Tensor(valset)

    model, trainset, valset, weight = gpu_cuda(model, trainset, valset, weight)
    if gpu:
        model = model.cuda()
        trainset = trainset.cuda()
        valset = valset.cuda()
        weight = weight.cuda()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    NLLLoss = nn.NLLLoss(weight=weight)
    # NLLLoss = MyLoss(weight=weight)
    best_loss = 99999.0
    best_f1 = 0.0

    for epoch in range(num_epochs):

        start = time.time()
        train_loss = 0.0
        val_loss = 0.0
        for i, datapack in enumerate(train_loader, 0):
            train_imgs = datapack[:, 0:3, :, :]
            train_masks = datapack[:, 3:, :, :]

            train_masks = train_masks.long()

            train_masks = train_masks.view(
                train_masks.size()[0],
                train_masks.size()[2],
                train_masks.size()[3]
            )

            optimizer.zero_grad()
            train_out = model(train_imgs)
            t_loss = NLLLoss(train_out, train_masks)
            t_loss.backward()
            optimizer.step()
            train_loss += t_loss.item()

            if i % 10 == 9:
                print('epoch: %3d, step: %3d loss: %.5f' % (epoch + 1, i + 1, train_loss / 10))
                writer.add_scalar('train_loss', train_loss / 10, (105 * epoch + i + 1))
                train_loss = 0.0

        for i, datapack in enumerate(val_loader, 0):
            val_imgs = datapack[:, 0:3, :, :]
            val_masks = datapack[:, 3:, :, :]

            val_masks = val_masks.long()
            val_masks = val_masks.view(
                val_masks.size()[0],
                val_masks.size()[2],
                val_masks.size()[3]
            )

            # optimizer.zero_grad()
            val_out = model(val_imgs)
            v_loss = NLLLoss(val_out, val_masks)
            val_loss += v_loss.item()

            if i % val_steps == val_steps - 1:
                val_loss = val_loss / val_steps
                '''
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_name = './ckpt/net_4_w' + weightid + '_di.pkl'
                    torch.save(model.state_dict(), save_name)
                '''
                end = time.time()
                time_spent = end - start
                writer.add_scalar('val_loss', val_loss, epoch)
                print('epoch: %3d, time: %.5f val_loss: %.5f' % (epoch + 1, time_spent, val_loss))
                val_loss = 0.0
                p, r, f = MyMetrics(model)
                if f > best_f1:
                    best_f1 = f
                    save_name = './ckpt/net_di_m2_n4_w' + weightid + '.pkl'
                    torch.save(model.state_dict(), save_name)
                writer.add_scalar('precision', p, epoch)
                writer.add_scalar('recall', r, epoch)
                writer.add_scalar('f1_score', f, epoch)
                print('p:', p)
                print('r:', r)
                print('f:', f)
                print('******************************************************************************')

    # writer.export_scalars_to_json('./loss.json')
    writer.close()

if __name__ == '__main__':
    fcn36_basic = FCN_36_Module()
    print(fcn36_basic)
