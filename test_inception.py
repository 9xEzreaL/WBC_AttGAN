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
from cnn_finetune import make_model


"""
Use pretrained classifier model to test generated data and get testing accuracy.
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
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/try')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/try.txt')

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
                        choices=['vgg', 'inception_model', 'densenet', 'resnet'])
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
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

if __name__=='__main__':
    args = parse()
    print(args)

    args.lr_base = args.lr
    args.n_attrs = len(args.attrs)
    writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

    if args.data == 'WBC_all':
        from data import WBC_all
        test_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'test_cls', args.attrs, args.resize)

    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.num_workers,
        shuffle=False, drop_last=False
    )

    progressbar = Progressbar()
    if args.net == 'inception':
        model = inception_model()
    if args.net == 'resnet':
        model = resnet()
    if args.net == 'vgg':  # seems fail
        model = VGG()
    if args.net == 'densenet':
        model = densenet()
    if args.net == 'resnext':
        model = resnext()
    if args.net == ' nasnet':
        model = nasnet()

    model.eval()
    model.load_state_dict(torch.load('output/classifier/resnet/checkpoint/weights.184.pth')['model'])
    if args.gpu: model.cuda()


    att_t_list = []
    recon_attr_list = []
    att_t_class = []
    for img_t, att_t, _ in progressbar(test_dataloader):
        img_t = img_t.cuda() if args.gpu else img_t
        att_t = att_t.cuda() if args.gpu else att_t
        att_t = att_t.type(torch.float)

        output = model(img_t)
        att_t_list.append(att_t.detach().cpu())
        recon_attr_list.append(output.detach().cpu())
        att_t_class.append(att_t.detach().cpu())
    get_metrix = Accuracy()
    recon_accuracy = get_metrix(att_t_list, recon_attr_list, att_t_class)
    recon_accuracy = np.array(
        [recon_accuracy[int(i) - 1][int(j) - 1] for i in attrs_dict for j in attrs_dict]).reshape(
        (len(attrs_dict), len(attrs_dict)))
    print(f'accuracy : \n{recon_accuracy}')


# output/classifier/inception/checkpoint/weights.114.pth
# output/classifier/resnet/checkpoint/weights.184.pth
# output/clasifier/denset/checkpoint/weights.191.pth

# default_MP_3_correct_adv_l1_EP200
# default_MP_3_correct_adv_l1_EP180(not used)
# generate_selected   from EP200