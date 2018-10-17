import torch
from torch import nn
from util.resblock import BasicBlock
from torch.autograd import Variable
from scalar_quantizer import quantize
import math


# class _netE(nn.Module):
#     def __init__(self, nc, nz, ngf, kernel=2, padding=1, img_width=64, img_height=64, quant_levels=None, do_comp=False, ncenc=8, nresenc=0, detenc=False, ngpu=1):
#         super(_netE, self).__init__()
#         self.ngpu = ngpu
#         self.detenc = detenc
#
#         main_list = [
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ngf, kernel, 2, padding, bias=False),
#             nn.ReLU(True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ngf, ngf * 2, kernel, 2, padding, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ngf * 2, ngf * 4, kernel, 2, padding, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ngf * 4, ngf * 8, kernel, 2, padding, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True)
#         ]
#
#         # state size. (ndf*8) x 4 x 4
#         if do_comp:
#             main_list += [
#                 nn.Conv2d(ngf * 8, ncenc, 3, 1, 1, bias=True),
#                 quantize(quant_levels),
#                 nn.ConvTranspose2d(ncenc, ngf * 8, 3, 1, 1, bias=True)
#             ]
#
#         if nresenc > 0:
#             main_list += [BasicBlock(ngf * 8, ngf * 8) for _ in range(nresenc)]
#
#         self.main = nn.Sequential(*main_list)
#
#         # compute mean
#         self.lin_mu = nn.Conv2d(ngf * 8, nz, (img_height//16, img_width//16), 1, 0, bias=True)
#         # compute sigma diagonals
#         if not detenc:
#             self.lin_logsig = nn.Conv2d(ngf * 8, nz, (img_height//16, img_width//16), 1, 0, bias=True)
#
#
#     def forward(self, input):
#         use_cuda = isinstance(input.data, torch.cuda.FloatTensor)
#         if use_cuda and self.ngpu > 1:
#             enc = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             enc = self.main(input)
#
#         output = self.lin_mu(enc)
#
#         if not self.detenc:
#             logsig = self.lin_logsig(enc)
#
#             if use_cuda:
#                 noise = torch.FloatTensor(*list(logsig.size())).normal_().cuda()
#             else:
#                 noise = torch.FloatTensor(*list(logsig.size())).normal_()
#
#             output = output + torch.sqrt(1e-8 + torch.exp(logsig)) * Variable(noise)
#
#         return output


class _netE(nn.Module):
    def __init__(self, nc, nz, ngf, kernel=2, padding=1, img_width=64, img_height=64, quant_levels=None, do_comp=False, ncenc=8, nresenc=0, detenc=False, noisedelta=0.5, bnz=False, ngpu=1, rmvquant = False):
        super(_netE, self).__init__()
        self.ngpu = ngpu
        self.detenc = detenc or not do_comp
        self.noisedelta = noisedelta
        self.nfmodelz = math.ceil(nz / ((img_height//16) * (img_width//16))) + ncenc #ngf * 8
        self.ncenc = ncenc
        ngf = 128 # !!!!!!!!!!!!!!
        model_down_list = [
            # input is (nc) x 32 x 32          -->  128
            nn.Conv2d(nc, ngf, kernel, 2, padding, bias=False),
            nn.ReLU(True),
            # state size. (ndf) x 16  x 16     -->    256
            nn.Conv2d(ngf, ngf * 2, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ndf*2) x 8 x 8        -->    512
            nn.Conv2d(ngf * 2, ngf * 4, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ndf*4) x 4 x 4       -->   1024
            nn.Conv2d(ngf * 4, ngf * 8, kernel, 2, padding, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
 
        ]

        if do_comp:
            model_down_list += [
                nn.Conv2d(ngf * 8, ncenc, 3, 1, 1, bias=True)
            ]
            if not rmvquant:
                model_down_list +=[
                    quantize(quant_levels)
                ]

        self.model_down = nn.Sequential(*model_down_list)

        # state size. (ndf*8) x 4 x 4
        if do_comp:
            model_z_list = [
                nn.ConvTranspose2d(ncenc, ngf * 8, 3, 1, 1, bias=True) if detenc else nn.ConvTranspose2d(self.nfmodelz, ngf * 8, 3, 1, 1, bias=True) #nn.ConvTranspose2d(ngf * 8, ngf * 8, 3, 1, 1, bias=True)
            ]
        else:
            model_z_list = []

        if nresenc > 0:
            model_z_list += [BasicBlock(ngf * 8, ngf * 8) for _ in range(nresenc)]

        model_z_list += [nn.Conv2d(ngf * 8, nz, (img_height//16, img_width//16), 1, 0, bias=False)]

        if bnz:
            model_z_list += [nn.BatchNorm2d(nz)] #, affine=False

        # compute mean
        self.model_z = nn.Sequential(*model_z_list)


    def forward(self, input):
        use_cuda = isinstance(input.data, torch.cuda.FloatTensor)
        if use_cuda and self.ngpu > 1:
            out_down = nn.parallel.data_parallel(self.model_down, input, range(self.ngpu))
        else:
            out_down = self.model_down(input)

        if not self.detenc:
            # import pdb; pdb.set_trace()
            # out_down_pad_size = list(out_down.size())
            # out_down_pad_size[1] = self.nfmodelz - self.ncenc
            # out_down_pad = torch.zeros(out_down_pad_size)
            # if use_cuda:
            #     out_down_pad = out_down_pad.cuda()
            # out_down = torch.cat([out_down, Variable(out_down_pad)], 1)
            #
            # if use_cuda:
            #     noise = torch.FloatTensor(*list(out_down.size())).uniform_(-self.noisedelta, self.noisedelta).cuda()
            # else:
            #     noise = torch.FloatTensor(*list(out_down.size())).uniform_(-self.noisedelta, self.noisedelta)
            #
            # out_down = out_down + Variable(noise)
            # out_down[:, self.ncenc:, :, :] = out_down[:, self.ncenc:, :, :]/self.noisedelta

            out_down_pad_size = list(out_down.size())
            out_down_pad_size[1] = self.nfmodelz - self.ncenc
            out_down_pad = torch.zeros(out_down_pad_size)
            out_down_pad.uniform_(-self.noisedelta, self.noisedelta)
            # out_down_pad.normal_(0, math.sqrt(self.noisedelta))
            if use_cuda:
                out_down_pad = out_down_pad.cuda()
            out_down = torch.cat([out_down, Variable(out_down_pad)], 1)

        if use_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model_z, out_down, range(self.ngpu))
        else:
            output = self.model_z(out_down)

        return output


    def print_encoding(self, input):
        use_cuda = isinstance(input.data, torch.cuda.FloatTensor)
        if use_cuda and self.ngpu > 1:
            out_down = nn.parallel.data_parallel(self.model_down, input, range(self.ngpu))
        else:
            out_down = self.model_down(input)

        if not self.detenc:
            # import pdb; pdb.set_trace()
            # out_down_pad_size = list(out_down.size())
            # out_down_pad_size[1] = self.nfmodelz - self.ncenc
            # out_down_pad = torch.zeros(out_down_pad_size)
            # if use_cuda:
            #     out_down_pad = out_down_pad.cuda()
            # out_down = torch.cat([out_down, Variable(out_down_pad)], 1)
            #
            # if use_cuda:
            #     noise = torch.FloatTensor(*list(out_down.size())).uniform_(-self.noisedelta, self.noisedelta).cuda()
            # else:
            #     noise = torch.FloatTensor(*list(out_down.size())).uniform_(-self.noisedelta, self.noisedelta)
            #
            # out_down = out_down + Variable(noise)
            # out_down[:, self.ncenc:, :, :] = out_down[:, self.ncenc:, :, :]/self.noisedelta

            out_down_pad_size = list(out_down.size())
            out_down_pad_size[1] = self.nfmodelz - self.ncenc
            out_down_pad = torch.zeros(out_down_pad_size)
            out_down_pad.uniform_(-self.noisedelta, self.noisedelta)
            # out_down_pad.normal_(0, math.sqrt(self.noisedelta))
            if use_cuda:
                out_down_pad = out_down_pad.cuda()
            #out_down = torch.cat([out_down, Variable(out_down_pad)], 1)

        return out_down


class _netG(nn.Module):
    def __init__(self, nc, nz, ngf, kernel=2, padding=1, output_padding=0, img_width=64, img_height=64, nresdec=0, ngpu=1):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        ngf = 128 # !!!!!!!!!!!!!!
        # input is Z, going into a convolution
        main_list = [nn.ConvTranspose2d(nz, ngf * 8, (8,8), 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True)]

        if nresdec > 0:
            main_list += [BasicBlock(ngf * 8, ngf * 8) for _ in range(nresdec)]

        main_list += [
           
            # state size. (ngf*8) x 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel, 2, padding, output_padding, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel, 2, padding, output_padding, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32x32
            nn.Conv2d( 2*ngf,      nc, 3, 1, 1, bias=True),
            #nn.ReLU(True)
            # state size. 32x32x1
        ]

        self.main = nn.Sequential(*main_list)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output



class _netDz(nn.Module):
    def __init__(self, nz, ndf=512, ndl=5, ngpu=0, avbtrick=False, sigmasq=1):
        super(_netDz, self).__init__()
        self.ngpu = ngpu
        self.avbtrick = avbtrick
        self.sigmasqz = sigmasq
        self.nz = nz

        layers = [[nn.Linear(ndf, ndf), nn.ReLU(True)] for _ in range(ndl-2)]

        layers = [nn.Linear(nz, ndf), nn.ReLU(True)] \
                    + sum(layers, []) \
                    + [nn.Linear(ndf, 1)]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        if self.avbtrick:
            output = output - torch.norm(input, p=2, dim=1, keepdim=True)**2 / 2 / self.sigmasqz \
                        - 0.5 * math.log(2 * math.pi) \
                        - 0.5 * self.nz * math.log(self.sigmasqz)

        return output.view(-1, 1).squeeze(1)


class CustomLayerNorm(nn.Module):
    def __init__(self, norm_shape):
        super(CustomLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((1, norm_shape[0], 1, 1)))
        self.beta = nn.Parameter(torch.zeros((1, norm_shape[0], 1, 1)))
        self.layernorm = nn.LayerNorm(norm_shape, elementwise_affine=False)

    def forward(self, x):
        x = self.layernorm(x)

        return self.gamma * x + self.beta


class _netDim(nn.Module):
    def __init__(self, nc=3, ndf=64, kernel=2, padding=1, img_width=64, img_height=64, ngpu=1):
        super(_netDim, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(    nc,      ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            #             nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf, img_height//2, img_width//2]),
            # CustomLayerNorm([ndf, img_height//2, img_width//2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 2, img_height//4, img_width//4]),
            # CustomLayerNorm([ndf * 2, img_height//4, img_width//4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 4, img_height//8, img_width//8]),
            # CustomLayerNorm([ndf * 4, img_height//8, img_width//8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel, 2, padding, bias=False),
            nn.LayerNorm([ndf * 8, img_height//16, img_width//16]),
            # CustomLayerNorm([ndf * 8, img_height//16, img_width//16]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, (img_height//16, img_width//16), 1, 0, bias=False),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class _netDimSN(nn.Module):
    def __init__(self, nc=3, ndf=64, kernel=2, padding=1, img_width=64, img_height=64, ngpu=1):
        super(_netDimSN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            #             nn.ReLU(True),
            nn.Conv2d(ndf, ndf, kernel, 2, padding, bias=True),
            # nn.LayerNorm([ndf, img_height//2, img_width//2]),
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=True),
            nn.LayerNorm([ndf, img_height//2, img_width//2]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel, 2, padding, bias=True),
            # nn.LayerNorm([ndf * 2, img_height//4, img_width//4]),
            nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1, bias=True),
            nn.LayerNorm([ndf * 2, img_height//4, img_width//4]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel, 2, padding, bias=True),
            # nn.LayerNorm([ndf * 4, img_height//8, img_width//8]),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1, bias=True),
            nn.LayerNorm([ndf * 4, img_height//8, img_width//8]),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, kernel, 2, padding, bias=False),
            # nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=True),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, (img_height//16, img_width//16), 1, 0, bias=False),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LayerNorm([ndf * 8, img_height//8, img_width//8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, (img_height//8, img_width//8), 1, 0, bias=True)
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
