"""
Image inpainting using multi-resolution deep image priors.
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.resnet import ResNet
from models.unet import UNet, MultiresUNet
from preprocess import get_pyramid, preprocess_img

from PIL import Image

import os
import argparse
import numpy as np

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help='Path to input image', required=True)
    parser.add_argument(
        '-m', '--mask', help='Mask for inpainting', required=True)
    parser.add_argument(
        '-a', '--arch', choices=['unet', 'resnet'], default='unet')
    parser.add_argument(
        '-o', '--outdir', help='Output directory', default='./')
    parser.add_argument('-lr', '--lr', help='Learning rate',
                        type=float, default=1e-4)
    parser.add_argument('-t', '--type', help='Pyramid type',
                        choices=['gauss', 'laplacian'], default='gauss')
    parser.add_argument('-n', '--nlevels',
                        help='No. of pyramid levels', type=int, default=3)
    parser.add_argument('-e', '--epochs', help='No. of epochs', default=10000, type=int)
    parser.add_argument('-s', '--save', type=int, help='Epochs after which to save images', default=10)

    return parser


class ReconstructPyramid(nn.Module):
    """
    Reconstructing original image from the pyramid.
    """

    def __init__(self, type, nlevels):
        super().__init__()
        self.type = type
        self.nlevels = nlevels

    def reconstruct_gauss_pyr(self, pyr):
        x = pyr[0]
        #x = F.interpolate(x, size=None, scale_factor=2, mode='bilinear')
        return x

    def reconstruct_laplace_pyr(self, pyr):
        x = pyr[-1]
        sizes = [pyr[i].size() for i in range(len(pyr))]
        #print(sizes)
        for i in range(self.nlevels-2, -1, -1):
            tmp = F.interpolate(x, size=None,
                                    scale_factor=2, mode='bilinear')
            #print(tmp.shape)
            x = tmp + pyr[i]
        return x

    def forward(self, pyr):
        if self.type == 'gauss':
            return self.reconstruct_gauss_pyr(pyr)
        else:
            return self.reconstruct_laplace_pyr(pyr)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Load image and mask
    img = np.array(Image.open(args.input).resize((256, 256)), dtype=np.float32)
    mask = np.array(Image.open(args.mask).convert('L').resize((256, 256)), dtype=np.float32)

    if img is None or mask is None:
        raise Exception(
            'Image {} or mask {} not found'.format(args.img, args.mask))

    img, mask = preprocess_img(img, mask)
    img_in = torch.FloatTensor(img.transpose(2,0,1)).unsqueeze(0)
    rnd_mat = torch.rand_like(img_in).to(device)
    rnd_mat = rnd_mat * 0.1
    if args.arch == 'resnet':
        raise Exception('Resnet not implemented currently.')

    elif args.arch == 'unet':
        net = MultiresUNet(num_input_channels=3, num_output_channels=3, nlevels=args.nlevels,
                           upsample_mode='nearest', 
                           need_sigmoid=True, need_bias=True, pad='reflection' )

    recon_loss = nn.MSELoss()
    level_losses = [nn.MSELoss() for i in range(args.nlevels)]
    recon_op = ReconstructPyramid(args.type, args.nlevels)

    x = get_pyramid(img, args.type, args.nlevels)
    x = [i.to(device) for i in x]
    mask_t = torch.FloatTensor(np.dstack([mask, mask, mask]).transpose(2,0,1)).unsqueeze(0).to(device)
    masks_pyr = []
    tmp_mask = mask_t.clone()
    for i in range(args.nlevels):
        masks_pyr.append(tmp_mask)
        mask_pyr = F.interpolate(tmp_mask, scale_factor=0.5)
        tmp_mask = mask_pyr.clone()
    net = net.to(device)
    recon_op = recon_op.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        net.train()
        gen_pyr = net(rnd_mat)
        recon_img = recon_op(gen_pyr)

        recon_loss_val = recon_loss(
            recon_img*mask_t, img_in*mask_t)
        level_loss_val = torch.tensor(0.0)
        for i in range(1,args.nlevels):
           #print(x[i].size(), gen_pyr[i].size())
           level_loss_val += level_losses[i](gen_pyr[i]* masks_pyr[i], x[i]* masks_pyr[i])
           #level_loss_val += level_losses[i](gen_pyr[i], x[i])
        #level_loss_vals = level_losses[i](gen_pyr[i]) for i in range(args.nlevels)]
        opt.zero_grad()
        total_loss = recon_loss_val + level_loss_val
        total_loss.backward()
        opt.step()

        print('Total loss:{:.05f}, recon loss:{:.05f}, levelwise loss: {:.05f}'.format(
            total_loss.item(), recon_loss_val.item(), level_loss_val.item()))
        if epoch % args.save == 0:
            net.eval()
            with torch.no_grad():
                out_pyr = net(rnd_mat)
                #print(out_pyr[0].shape)
                output_img = out_pyr[0][0,...].cpu().detach().numpy().transpose(1,2,0)
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.tick_params(top=False,left=False,right=False,bottom=False, labelleft=False, labelbottom=False)
                ax.imshow(output_img)
                plt.savefig(os.path.join(args.outdir, '{}.png'.format(epoch)), bbox_inches='tight'), 
                plt.close()

if __name__ == '__main__':
    main()