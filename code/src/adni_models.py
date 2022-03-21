import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import Swish
import pdb

def get_activation_layer(name):
    return {
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'softplus': nn.Softplus,
        'leaky_relu': nn.LeakyReLU,
        'swish': Swish,
        'tanh': nn.Tanh
    }[name]


class ResNet(nn.Module):
    def __init__(self, in_depth, n_blocks, interm_depths, bottleneck=True, n_out_linear=None, dropout=0.):
        super(ResNet, self).__init__()
        self.name = 'Resnet'
        assert(len(n_blocks) == len(interm_depths))
        self.init_conv = nn.Conv3d(in_depth, interm_depths[0], kernel_size=7, stride=2, padding=3, bias=True)
        self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.stages = nn.ModuleList([self._build_stage(n_blocks[i], interm_depths[max(0, i - 1)],
                                                       out_depth=interm_depths[i], stride=1 if i == 0 else 2,
                                                       bottleneck=bottleneck) for i in range(len(n_blocks))])

        self.pool_linear = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        if n_out_linear is not None:
            self.output_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], n_out_linear, dropout=dropout)
        else:
            self.output_head = None
        
        for m in self.modules():
            if type(m) in (nn.Conv3d, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            if type(m) is nn.BatchNorm3d:
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def encode(self, x):
        _, _, latent, _ = self.forward(x, None)
        return latent
    
    def forward(self, x, g):
        x = self.init_conv(x)
        x = self.init_pool(x)
        for s in self.stages:
            x = s(x)
        x = self.pool_linear(x)
        latent = self.flatten(x)
        if self.output_head is not None:
            x = self.output_head(latent)

        return None, x, latent, None

    def _build_stage(self, n_blocks, in_depth, hidden_depth=None, out_depth=None, stride=2, dilation=1,
                     batchnorm=True, activation='swish', zero_output=True, bottleneck=True):
        if out_depth is None:
            out_depth = in_depth * stride
        blocks = [ResNetBlock(in_depth if i == 0 else out_depth, hidden_depth, out_depth, 
                                stride=stride if i == 0 else 1,
                                dilation=dilation, 
                                batchnorm=batchnorm, 
                                activation=activation, 
                                zero_output=zero_output, 
                                bottleneck=bottleneck) for i in range(n_blocks)]
        return nn.Sequential(*blocks)

    def _build_linear_head(self, in_depth, hidden_depths, out_depth, activation='swish',
                           batchnorm=True, dropout=0.):
        layers = []
        depths = [in_depth, *hidden_depths, out_depth]

        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        layers.append(nn.Flatten())
        for i in range(len(depths) - 1):
            if dropout > 0.:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(depths[i], depths[i + 1], bias=not batchnorm))
            if i != len(depths) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(depths[i + 1], eps=1e-8))
                layers.append(get_activation_layer(activation)())

        return nn.Sequential(*layers)



class TauResNet(nn.Module):
    def __init__(self, in_depth, n_blocks, interm_depths, bottleneck=True, n_out_linear=None, dropout=0., const=0.01):
        super(TauResNet, self).__init__()
        print('Using TauResNet!!!')
        
        self.name = 'TauResNet'
        self.const = const
        assert(len(n_blocks) == len(interm_depths))
        self.init_conv = nn.Conv3d(in_depth, interm_depths[0], kernel_size=7, stride=2, padding=3, bias=True)
        self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.stages = nn.ModuleList([self._build_stage(n_blocks[i], interm_depths[max(0, i - 1)],
                                                       out_depth=interm_depths[i], stride=1 if i == 0 else 2,
                                                       bottleneck=bottleneck) for i in range(len(n_blocks))])

        self.pool_linear = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        #Tau head Layers
        self.tau_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], interm_depths[-1], dropout=dropout)
        self.tanh = get_activation_layer('tanh')()
        
        # y-predictor
        if n_out_linear is not None:
            self.output_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], n_out_linear, dropout=dropout)
        else:
            self.output_head = None

            
        
        for m in self.modules():
            if type(m) in (nn.Conv3d, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            if type(m) is nn.BatchNorm3d:
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def encode(self, x):
        x = self.init_conv(x)
        x = self.init_pool(x)
        for s in self.stages:
            x = s(x)
        x = self.pool_linear(x)
        x = self.flatten(x)
        return x
        
    def encodeTauNet(self, x, gval):
        x = self.encode(x)
        li_latent = self.project_sphere(x)
        gi_latent = self.TauNet(li_latent, gval)
        return gi_latent

    def encodeProject(self, x):
        x = self.encode(x)
        li_latent = self.project_sphere(x)
        return li_latent

    def TauNet(self, li_latent, gval):
        fct = self.tau_head(li_latent)
        gi_latent = gval + self.const*self.tanh(fct)
        return gi_latent

    def project_sphere(self, mu):
        mu_centered = (mu - torch.mean(mu, dim=1, keepdim=True))
        mu_norm = torch.norm(mu_centered, dim=1, keepdim=True)
        return mu_centered / mu_norm
    
    def forward(self, x, gval):
        x = self.encode(x)
        
        li_latent = self.project_sphere(x)
        gi_latent = self.TauNet(li_latent, gval)
        
        if self.output_head is not None:
            pred = self.output_head(li_latent)

        return None, pred, gi_latent, None

    def _build_stage(self, n_blocks, in_depth, hidden_depth=None, out_depth=None, stride=2, dilation=1,
                     batchnorm=True, activation='swish', zero_output=True, bottleneck=True):
        if out_depth is None:
            out_depth = in_depth * stride
        blocks = [ResNetBlock(in_depth if i == 0 else out_depth, hidden_depth, out_depth, 
                                stride=stride if i == 0 else 1,
                                dilation=dilation, 
                                batchnorm=batchnorm, 
                                activation=activation, 
                                zero_output=zero_output, 
                                bottleneck=bottleneck) for i in range(n_blocks)]
        return nn.Sequential(*blocks)

    def _build_linear_head(self, in_depth, hidden_depths, out_depth, activation='swish',
                           batchnorm=True, dropout=0.):
        layers = []
        depths = [in_depth, *hidden_depths, out_depth]

        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        layers.append(nn.Flatten())
        for i in range(len(depths) - 1):
            if dropout > 0.:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(depths[i], depths[i + 1], bias=not batchnorm))
            if i != len(depths) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(depths[i + 1], eps=1e-8))
                layers.append(get_activation_layer(activation)())

        return nn.Sequential(*layers)




class BNetPred(nn.Module):
    def __init__(self, device, interm_depths, n_out_linear, dropout=0.):
        super(BNetPred, self).__init__()

        print('Using BNetPred !!!')
        
        self.name = 'BNetPred'
        self.output_dim = n_out_linear
        self.latent_dim = interm_depths[-1]
        self.device = device
        
        # Dim up
        group_dim = int((self.latent_dim*(self.latent_dim-1))/2)
        self.fcdimup = nn.Linear(self.latent_dim, group_dim)

        # BNet layers
        self.bnet_head = MLP(self.latent_dim, [self.latent_dim*2], self.latent_dim, dropout=dropout)

        # y-predictor
        self.output_head = MLP(self.latent_dim, [self.latent_dim*2], self.output_dim, dropout=dropout)

    def cayley_map(self, vec):
        N = vec.shape[0]
        A = torch.zeros((N, self.latent_dim, self.latent_dim)).to(self.device)
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=-1)
        A[:, tril_indices[0], tril_indices[1]] = vec
        A = A.transpose(-1, -2) - A

        I = torch.eye(self.latent_dim).to(self.device)
        I = I.reshape((1, self.latent_dim, self.latent_dim))
        I = I.repeat(N, 1, 1)

        Q = I + A
        B = (I - A)
        norms = (torch.norm(B, p = 1, dim=(1,2)) * torch.norm(B, p = float('inf'), dim=(1,2)))
        Z = B.transpose(-1, -2)/ norms.view(N, 1, 1)
        for _ in range(10):
            Z = 0.25*Z@(13*I - B@Z@(15 * I - B@Z@(7*I - B @Z)))
        return Z@Q

    def encode(self, li_latent, gi_latent):
        gi_latent = self.fcdimup(gi_latent) # 64*64 -> 64*2016
        gi_rot = self.cayley_map(gi_latent) # 64*2016 -> 64*64*64

        # 64*64*64 X 64*64*1 = 64*64
        mis = (gi_rot.transpose(-1, -2) @ li_latent.unsqueeze(-1)).squeeze(-1)
        
        fct = self.bnet_head(mis)
        
        latent = (gi_rot @ fct.unsqueeze(-1)).squeeze(-1)
        return latent

    def forward(self, li_latent, gi_latent):
        latent = self.encode(li_latent, gi_latent)
        pred = self.output_head(latent)

        return None, pred, latent
    
    
    
class MLP(nn.Module):
    def __init__(self, in_depth, hidden_depths, out_depth, activation='swish', batchnorm=True, dropout=0.):
        super(MLP, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.depths = [in_depth, *hidden_depths, out_depth]

        self.linear_layers = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.act = nn.ModuleList([])

        for i in range(len(self.depths) - 1):
            self.linear_layers.append(nn.Linear(self.depths[i], self.depths[i + 1], bias=not batchnorm))
            if i != len(self.depths) - 2:
                if batchnorm:
                    self.norm.append(nn.BatchNorm1d(self.depths[i + 1], eps=1e-8))
                self.act.append(get_activation_layer(activation)())

    def forward(self, x):
        for i in range(len(self.depths) - 1):
            if self.dropout > 0.:
                x = F.dropout(x, self.dropout, self.training)
            x = self.linear_layers[i](x)
            if i != len(self.depths) - 2:
                if self.batchnorm:
                    x = self.norm[i](x)
                x = self.act[i](x)
            #if i == len(self.depths) - 3:
            #    latent = x
        #return x, latent
        return x
        
    def l1_norm(self):
        return sum([torch.norm(l.weight, 1) for l in self.linear_layers])
    
    def l2_norm(self):
        return sum([torch.norm(l.weight, 2) for l in self.linear_layers])


class ResNetBlock(nn.Module):
    def __init__(self, in_depth, hidden_depth=None, out_depth=None, stride=1, dilation=1,
                 batchnorm=True, activation='swish', zero_output=True, bottleneck=True):
        super(ResNetBlock, self).__init__()
        if out_depth is None:
            out_depth = in_depth * stride
        if stride > 1:
            self.shortcut_layer = nn.Conv3d(in_depth, out_depth, kernel_size=3, stride=stride,
                                            padding=1, dilation=dilation, bias=True)
        else:
            self.shortcut_layer = None

        layers = []
        if bottleneck:
            if hidden_depth is None:
                hidden_depth = in_depth // 4
            k_sizes = [3, 1, 3]
            depths = [in_depth, hidden_depth, hidden_depth, out_depth]
            paddings = [1, 0, 1]
            strides = [1, 1, stride]
            dilations = [dilation, 1, dilation]
        else:
            if hidden_depth is None:
                hidden_depth = in_depth
            k_sizes = [3, 3]
            depths = [in_depth, hidden_depth, out_depth]
            paddings = [1, 1]
            strides = [1, stride]
            dilations = [dilation, dilation]
        
        for i in range(len(k_sizes)):
            if batchnorm:
                layers.append(nn.BatchNorm3d(depths[i], eps=1e-8))
            layers.append(get_activation_layer(activation)())
            layers.append(nn.Conv3d(depths[i], depths[i + 1], k_sizes[i], padding=paddings[i],
                                    stride=strides[i], dilation=dilations[i], bias=False))
        
        self.layers = nn.Sequential(*layers)
        # for i, l in enumerate(self.layers):
        #     self.add_module("layer_{:d}".format(i), l)
        
    def forward(self, x):
        Fx = self.layers(x)
        if self.shortcut_layer is not None:
            x = self.shortcut_layer(x)
        return x + Fx



