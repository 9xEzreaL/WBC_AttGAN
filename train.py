# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Main entry point for training AttGAN network."""

import argparse
import datetime
import json
import os
from os.path import join
import numpy as np
import torch.utils.data as data
from util.metrix import Accuracy

import torch
import torchvision.utils as vutils
from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
from torchsampler import ImbalancedDatasetSampler



# attrs_default = [
#     'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
#     'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
# ]
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
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ', 'wbc', 'WBC_all'], default='WBC_all')
    parser.add_argument('--data_path', dest='data_path', type=str, default='data/WBC_all')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='data/WBC_all.txt')
    # not every dataset used
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    
    parser.add_argument('--img_size', dest='img_size', type=int, default=256)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32) # training batch size
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=12, help='# of sample images') # valid batch size
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=1000)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--aug', default=['deg','flp'], type=str, help='tio augmentation: aff, ela, gam, noi, mot, bia, spa')

    
    return parser.parse_args(args)

args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))
if args.data == 'wbc':
    from data import WBC
    train_dataset = WBC(args.data_path, args.attr_path, args.img_size, 'train', args.attrs, args.aug)
    valid_dataset = WBC(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs, args.aug)
if args.data == 'WBC_all':
    from data import WBC_all
    train_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'train', args.attrs, args.aug)
    valid_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs, args.aug)
    test_dataset = WBC_all(args.data_path, args.attr_path, args.img_size, 'test', args.attrs, args.aug)
if args.data == 'CelebA':
    from data import CelebA
    train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
if args.data == 'CelebA-HQ':
    from data import CelebA_HQ
    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)

train_dataloader = data.DataLoader(
    train_dataset, sampler=ImbalancedDatasetSampler(train_dataset), batch_size=args.batch_size,
    num_workers=args.num_workers,drop_last=True
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

attgan = AttGAN(args)
progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

fixed_img_a, fixed_att_a, _ = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float) # type attr belongs to
sample_att_b_list = [fixed_att_a]
sample_att_t_list =[]
for i in range(args.n_attrs): # 6
    # tmp = torch.zeros((args.n_samples, args.n_attrs), dtype=torch.float32).cuda() if args.gpu else torch.zeros((args.n_samples, args.n_attrs), dtype=torch.float32)
    # tmp[:, i] = 1
    tmp = torch.cat([sample_att_b_list[0][i*2].unsqueeze(0)]*12,0)
    tmp_t = torch.cat([sample_att_b_list[0][i*2].unsqueeze(0)],0)
    sample_att_b_list.append(tmp) # len will be 1+6
    sample_att_t_list.append(tmp_t)

it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    attgan.set_lr(lr)
    writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_a, att_a, _ in progressbar(train_dataloader):
        attgan.train()
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        att_a_ = (att_a * 2 - 1) * args.thres_int
        if args.b_distribution == 'none':
            att_b_ = (att_b * 2 - 1) * args.thres_int
        if args.b_distribution == 'uniform':
            att_b_ = (att_b * 2 - 1) + \
                     (att_b * 2 - 1)*torch.rand_like(att_b) * \
                     (2 * args.thres_int)
        if args.b_distribution == 'truncated_normal':
            att_b_ = (att_b * 2 - 1) * \
                     (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * \
                     (2 * args.thres_int)
        
        if (it+1) % (args.n_d+1) != 0:
            errD = attgan.trainD(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errD, it+1, 'D')
        else:
            errG = attgan.trainG(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
        
        if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            attgan.saveG(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))


        if (it+1) % args.sample_interval == 0:
            attgan.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                for i, att_b in enumerate(sample_att_b_list):
                    att_b_ = (att_b * 2 - 1) * args.thres_int
                    # seems no need~~~
                    # if i > 0:
                    #     att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
                    samples.append(attgan.G(fixed_img_a, att_b_))
                samples = torch.cat(samples, dim=3)
                writer.add_image('sample', vutils.make_grid(samples, nrow=1, normalize=True, range=(-1., 1.)), it+1)
                vutils.save_image(samples, os.path.join(
                        'output', args.experiment_name, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=True, range=(-1., 1.))
        it += 1

    attr_list = []
    gen_attr_list = []
    att_t_list = []
    recon_attr_list = []
    for img_t, att_t, ID in progressbar(test_dataloader):
        attgan.eval()
        with torch.no_grad():
            img_t = img_t.cuda() if args.gpu else img_t  # [1,3,256,256]
            if (epoch + 1) % 10 == 0:
                for type in attrs_dict:
                    os.makedirs(
                        join('output', args.experiment_name, 'sample_testing', 'Epoch_' + str(epoch + 1),
                             attrs_dict[type]),
                        exist_ok=True)
            for att_t_changed in sample_att_t_list:
                att_t_changed = att_t_changed.cuda() if args.gpu else att_t_changed  # [1,6]
                att_t_changed = att_t_changed.type(torch.float)  # [[0,0,0,0,0,1]] class we want change
                att_t_changed_ = (att_t_changed * 2 - 1) * args.thres_int  # [[-0.5,-0.5,-0.5,-0.5,-0.5,0.5]] class type for generator use

                generated_img_t = attgan.G(img_t, att_t_changed_)
                _, generated_img_t_soft_classes = attgan.D(generated_img_t)

                # deal save image
                if (epoch + 1) % 10 == 0:
                    att_t_changed_name = attrs_dict[str(int(np.where(att_t_changed[0].cpu() == 1)[0]) + 1)]
                    att_t_name = attrs_dict[str(int(np.where(att_t[0].cpu() == 1)[0]) + 1)]
                    vutils.save_image(generated_img_t, os.path.join(
                        'output', args.experiment_name, 'sample_testing', 'Epoch_' + str(epoch + 1), att_t_changed_name,
                        f'{ID[0]}'
                    ), nrow=1, normalize=True, range=(-1., 1.))
                attr_list.append(att_t_changed.detach().cpu())
                gen_attr_list.append(generated_img_t_soft_classes.detach().cpu())

            # deal recon
            att_t = att_t.cuda() if args.gpu else att_t
            att_t = att_t.type(torch.float)
            att_t_ = (att_t * 2 - 1) * args.thres_int
            recon_img_t = attgan.G(img_t, att_t_)
            _, recon_img_t_soft_classes = attgan.D(recon_img_t)
            att_t_list.append(att_t.detach().cpu())
            recon_attr_list.append(recon_img_t_soft_classes.detach().cpu())
    get_metrix = Accuracy()
    accuracy = get_metrix(attr_list, gen_attr_list)
    accuracy = [accuracy[int(i) - 1] for i in attrs_dict]
    recon_accuracy = get_metrix(att_t_list, recon_attr_list)
    recon_accuracy = [recon_accuracy[int(i) - 1] for i in attrs_dict]
    print(f'Epoch {epoch} recon accuracy : {recon_accuracy}')
    print(f'Epoch {epoch} generated accuracy : {accuracy}')
# python train.py