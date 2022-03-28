import argparse
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import normalize
#from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns


from src import dataloader as mydatasets, model as models
from src import adni_models
from src.average_meter import AverageMeter

def get_data(args, device, model_taunet, model_bnet, dataloader):
    X = []
    Y = []
    C = []
    G = []
    count = 0
    for images, labels, control, gattrs in dataloader:
        count += 1
        # Measure computed for a subset for Adult dataset
        if args.dataset_name == 'Adult' and count > 70:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        gattrs = gattrs.to(device)
        if args.equiv_type == 'ell2':
            li_latent = model_taunet.encodeProject(images)
            gi_latent = model_taunet.encodeTauNet(images, gattrs.unsqueeze(-1))
            latent = model_bnet.encode(li_latent, gi_latent)
        else:
            _, _, latent, _ = model_taunet(images, gattrs.unsqueeze(-1))
        X.append(latent)
        Y.append(labels)
        C.append(control)
        G.append(gattrs)

    X = torch.cat(X)
    Y = torch.cat(Y)
    C = torch.cat(C)
    G = torch.cat(G)
    return X, Y, C, G


def compute_mmd_loss(args, device, mu, labels, logf):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float().to(device)
    unq_labels = torch.unique(labels)
    for i in unq_labels:
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    lap_kernel = lap_kernel * s_prod
    loss = -lap_kernel[c_diff == True].sum()
    loss += (len(unq_labels)-1)*lap_kernel[c_diff == False].sum()

    message = 'mmd_measure_loss {}'.format(100*loss) # Scaling loss by 100
    print(message)
    logf.write(message + '\n')

    
def run_mmd_measure(args, device, testset, equivar_model_name, invar_model_name, model_path, output_path, logf):
    dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    dummy_x, _, _, _ = testset.__getitem__(0)
    input_dim = dummy_x.size(0)
    
    # LOADING THE MODELS FOR DIFFERENT CASES
    if args.equiv_type == 'ell2' and  args.dataset_name == 'German':
        # Loading the equivar model
        model_taunet = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.01).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
        
        # Loading the invar model            
        model_bnet = models.BNetEncDec(device=device, latent_dim=args.latent_dim, output_dim=input_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, invar_model_name + '.pth')
        model_bnet.load_state_dict(torch.load(path))
        model_bnet.name = invar_model_name
        model_bnet.eval()

        with torch.no_grad():
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
            
    elif args.equiv_type == 'ell2' and args.dataset_name == 'Adult':
        # Loading the equivar model            
        model_taunet = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.1).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()

        # Loading the invar model            
        model_bnet = models.BNetEncDec(device=device, latent_dim=args.latent_dim, output_dim=input_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, invar_model_name + '.pth')
        model_bnet.load_state_dict(torch.load(path))
        model_bnet.name = invar_model_name
        model_bnet.eval()

        with torch.no_grad():        
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
        
    elif args.equiv_type == 'ell2' and  (args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP'):
        # Loading the equivar model            
        model_taunet = adni_models.TauResNet(in_depth=1, n_blocks=args.blocks, interm_depths=args.channels, bottleneck=args.use_bottleneck_layers, n_out_linear=2, dropout=0.5, const=0.01)
        model_taunet = model_taunet.to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()

        # Loading the invar model            
        model_bnet = adni_models.BNetPred(device=device, interm_depths=args.channels, n_out_linear=2, dropout=0.5)
        model_bnet = model_bnet.to(device)
        path = os.path.join(model_path, invar_model_name + '.pth')
        model_bnet.load_state_dict(torch.load(path))
        model_bnet.name = invar_model_name
        model_bnet.eval()

        with torch.no_grad():
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
        
    elif args.equiv_type != 'ell2' and args.dataset_name == 'German':
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
        
        model_bnet = None

        with torch.no_grad():        
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
            
    elif args.equiv_type != 'ell2' and args.dataset_name == 'Adult':
        model_taunet = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()
        
        model_bnet = None

        with torch.no_grad():        
            X, Y, C, G = get_data(args, device, model_taunet, model_bnet, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
            
    elif args.equiv_type != 'ell2' and (args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP'):
        model_taunet = adni_models.ResNet(in_depth=1, n_blocks=args.blocks, interm_depths=args.channels, bottleneck=args.use_bottleneck_layers, n_out_linear=2, dropout=0.5)
        model_taunet = model_taunet.to(device)
        path = os.path.join(model_path, equivar_model_name + '.pth')
        model_taunet.load_state_dict(torch.load(path))
        model_taunet.name = equivar_model_name
        model_taunet.eval()

        model_bnet = None

        with torch.no_grad():
            X, Y, C, G = get_data(args, device, model_taunet, None, dataloader)
        compute_mmd_loss(args, device, X, C, logf)    
        
    else:
        raise NotImplementedError
