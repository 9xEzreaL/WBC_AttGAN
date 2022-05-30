import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import argparse
import datetime
import json
from os.path import join
import torch.utils.data as data
from util.metrix import Accuracy
from torchsummary import summary
import torchvision.utils as vutils
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
from torchsampler import ImbalancedDatasetSampler
from cnn_finetune import make_model

"""
Train a classfier model and save it so that you can use this classfier to test your generated data.
"""

attrs_default = [
    'band', 'blast', 'meta',
    'myelo', 'promyelo', 'seg'
]

attrs_dict = {'2':'blast',
              '5':'promyelo',
              '4':'myelo',
              '3':'meta',
              '1':'band',
              '6':'seg'}

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ', 'wbc', 'WBC_all'],
                        default='WBC_all')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/WBC_all')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/WBC_all.txt')

    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32) # training batch size
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--b_distribution', dest='b_distribution', default='none',
                        choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=12,
                        help='# of sample images')  # valid batch size
    parser.add_argument('--net', dest='net', default='vgg',
                        choices=['vgg', 'inception', 'densenet', 'resnet', 'resnext', 'nasnet'])
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=777)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--resize', dest='resize', default=0, type=int, help='image resize')


    return parser.parse_args(args)


class inception_model(nn.Module):
    def __init__(self):
        super(inception_model, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.aux_logits = False
        num_fc = self.model.fc.in_features
        self.model.fc = nn.Linear(num_fc, 6)
        # num_aux = self.model.AuxLogits.fc.in_features
        # self.model.AuxLogits.fc = nn.Linear(num_aux, 6)

    def forward(self, input):
        # input = torch.rand((2,3,299,299))
        output = self.model(input)
        return output

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.model = models.resnet34(pretrained=True)
        in_features = self.model.fc.in_features  # + [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.model.fc = nn.Linear(in_features, 6)
        # self.f_size = int(self.f_size / 2)

    def forward(self, input):
        output = self.model(input)
        return output

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.model = models.vgg13_bn(pretrained=True)
        self.model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 6))

    def forward(self, input):
        output = self.model(input)
        return output

class densenet(nn.Module):
    def __init__(self):
        super(densenet, self).__init__()
        self.model = models.densenet121(pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=in_features, out_features=6, bias=True)

    def forward(self, input):
        output = self.model(input)
        return output

class resnext(nn.Module):
    def __init__(self):
        super(resnext, self).__init__()
        self.model = make_model(model_name='resnext101_64x4d', num_classes=6, pretrained=True)

    def forward(self, input):
        output = self.model(input)
        return output

class nasnet(nn.Module):
    def __init__(self):
        super(nasnet, self).__init__()
        self.model = make_model(model_name='nasnetamobile', num_classes=6, pretrained=True)

    def forward(self, input):
        output = self.model(input)
        return output

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class Classifier():
    def __init__(self, args, net):
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False

        if net == 'inception':
            self.model = inception_model()
        if net == 'resnet':
            self.model = resnet()
        if net == 'vgg': # seems fail
            self.model = VGG()
        if net == 'densenet':
            self.model = densenet()
        if net == 'resnext':
            self.model = resnext()
        if net == 'nasnet':
            self.model = nasnet()

        self.model.train()
        if self.gpu: self.model.cuda()
        summary(self.model, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.optim_model = optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)

    def set_lr(self, lr):
        for g in self.optim_model.param_groups:
            g['lr'] = lr

    def train_model(self, img_a, label): #(self, img, label) [0., 0., 0., 0., 1., 0.]
        for p in self.model.parameters():
            p.requires_grad = True

        # label = torch.argmax(label, 1)
        fake_label = self.model(img_a)
        # fake_label = tuple_fake_label[1]
        label = label.type(torch.float)
        # fake_label = torch.argmax(fake_label, 1).type(torch.float)
        # dc_loss = F.binary_cross_entropy_with_logits(fake_label, label)
        dc_loss = F.l1_loss(fake_label, label)
        d_loss = dc_loss

        self.optim_model.zero_grad()
        d_loss.backward()
        self.optim_model.step()

        errD = {
            'd_loss': d_loss.item()
        }
        return errD

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
            'optim_model': self.optim_model.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'model' in states:
            self.model.load_state_dict(states['model'])
        if 'optim_model' in states:
            self.optim_model.load_state_dict(states['optim_model'])



if __name__=='__main__':
    args = parse()
    print(args)

    args.lr_base = args.lr
    args.n_attrs = len(args.attrs)
    args.betas = (args.beta1, args.beta2)

    os.makedirs(join('output', args.experiment_name), exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
    os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
    writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

    with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if args.data == 'WBC_all':
        from data import WBC_all
        train_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'train', args.attrs, args.resize)
        valid_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs, args.resize)
        test_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'test', args.attrs, args.resize)

    train_dataloader = data.DataLoader(
        train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=True
    )
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
        shuffle=False, drop_last=False
    )
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.num_workers,
        shuffle=False, drop_last=False
    )

    print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

    classifier = Classifier(args, net=args.net)
    progressbar = Progressbar()

    it = 0
    it_per_epoch = len(train_dataset) // args.batch_size
    for epoch in range(args.epochs):
        # train with base lr in the first 100 epochs
        # and half the lr in the last 100 epochs
        lr = args.lr_base / (10 ** (epoch // 100))
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        for img_a, att_a, _ in progressbar(train_dataloader):
            img_a = img_a.cuda() if args.gpu else img_a
            att_a = att_a.cuda() if args.gpu else att_a
            att_a = att_a.type(torch.float)

            errD = classifier.train_model(img_a, att_a)
            add_scalar_dict(writer, errD, it+1, 'D')
            progressbar.say(epoch=epoch, iter=it + 1, d_loss=errD['d_loss'])
            if (it + 1) % args.save_interval == 0:
                # To save storage space, I only checkpoint the weights of G.
                # If you'd like to keep weights of G, D, optim_G, optim_D,
                # please use save() instead of saveG().
                classifier.save(os.path.join(
                    'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
                ))
            it += 1

        att_t_list =[]
        recon_attr_list = []
        att_t_class = []
        for img_t, att_t, ID in progressbar(test_dataloader):
            classifier.eval()
            with torch.no_grad():
                img_t = img_t.cuda() if args.gpu else img_t  # [1,3,256,256]
                att_t = att_t.cuda() if args.gpu else att_t
                att_t = att_t.type(torch.float)

                fake_label = classifier.model(img_t)
                # label = torch.Tensor(np.where(att_t.cpu() == 1.)[1])
                # fake_label = torch.argmax(fake_label, 1)
                # labels.append(label.detach().cpu())
                # fake_labels.append(fake_label.detach().cpu())

                att_t_list.append(att_t.detach().cpu())
                recon_attr_list.append(fake_label.detach().cpu())
                att_t_class.append(att_t)
        get_metrix = Accuracy()
        recon_accuracy = get_metrix(att_t_list, recon_attr_list, att_t_class)
        recon_accuracy = np.array(
            [recon_accuracy[int(i) - 1][int(j) - 1] for i in attrs_dict for j in attrs_dict]).reshape(
            (len(attrs_dict), len(attrs_dict)))
        print(f'Epoch {epoch} recon accuracy : \n{recon_accuracy}')

# CUDA_VISIBLE_DEVICES=3 python InceptionV3.py --experiment_name classifier/inception --gpu --resize 299 --img_size 299
