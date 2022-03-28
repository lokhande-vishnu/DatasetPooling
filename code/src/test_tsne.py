import argparse
import os, sys, time, shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import normalize

from src import dataloader as mydatasets, model as models
from src import adni_models
from src.average_meter import AverageMeter

def get_data(args, device, model, dataloader, flag_latent):
    model.eval()
    X = []
    Y = []
    C = []
    G = []
    for images, labels, control, gattrs in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        gattrs = gattrs.to(device)
        if args.equiv_type == 'ell2':
            if flag_latent:
                latent = model.encodeProject(images)
            else:
                latent = model.encodeTauNet(images, gattrs.unsqueeze(-1))
        else:
            latent = model.encode(images)
        X.append(latent)
        Y.append(labels)
        C.append(control)
        G.append(gattrs)

    X = torch.cat(X).cpu().detach().numpy()
    Y = torch.cat(Y).cpu().detach().numpy()
    C = torch.cat(C).cpu().detach().numpy()
    G = torch.cat(G).cpu().detach().numpy()

    return X, Y, C, G


def get_samples(X, Y, C, G):
    idx = np.random.permutation(X.shape[0])
    #nsamples = min(10000, X.shape[0])
    nsamples = min(200, X.shape[0])    
    selected = idx[0:nsamples]
    return X[selected, :], Y[selected], C[selected], G[selected]

def compute_equivar_measure_loss(X, Y, C, G, flag_latent, logf):
    Xmin = np.min(X, 1, keepdims=True)
    Xmax = np.max(X, 1, keepdims=True)
    X = (X - Xmin) / (Xmax - Xmin)
    
    X_diff = np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0)
    G_diff = np.expand_dims(G, axis=1) - np.expand_dims(G, axis=0)
    G_diff = np.expand_dims(G_diff, axis=-1)
    measure = ((G_diff + X_diff)**2).sum(-1).mean()

    
    if flag_latent:
        message = 'Latent: Equivar test_measure_loss {}'.format(measure)
        print(message)
        logf.write(message + '\n')
    else:
        message = 'Group: Equivar test_measure_loss {}'.format(measure)
        print(message)
        logf.write(message + '\n')
    
def compute_equivar_measure_gap(X, Y, C, G, flag_latent, logf):
    Xmin = np.min(X, 1, keepdims=True)
    Xmax = np.max(X, 1, keepdims=True)
    X = (X - Xmin) / (Xmax - Xmin)
    
    X_diff = ((np.expand_dims(X, axis=1) - np.expand_dims(X, axis=0))**2).sum(-1)
    G_diff = np.absolute(np.expand_dims(G, axis=1) - np.expand_dims(G, axis=0))
    measure = (G_diff * X_diff).mean()

    if flag_latent:
        message = 'Latent: Equivar test_measure_gap {}'.format(measure)
        print(message)
        logf.write(message + '\n')
    else:
        message = 'Group: Equivar test_measure_gap {}'.format(measure)
        print(message)
        logf.write(message + '\n')
        

def run_tsne_plots(args, device, testset, model_name, output_path, logf):
    dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    dummy_x, _, _, _ = testset.__getitem__(0)
    input_dim = dummy_x.size(0)
    
    name = model_name
    path = os.path.join(output_path, 'snapshots', name + '.pth')

    if args.dataset_name == 'German':
        if args.equiv_type == 'ell2':
            model = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.01).to(device)        
        else:
            model = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
    elif args.dataset_name == 'Adult':
        if args.equiv_type == 'ell2':
            model = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.1).to(device)        
        else:
            model = models.BaselineEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0).to(device)
    elif args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP':
        if args.equiv_type == 'ell2':
            model = adni_models.TauResNet(in_depth=1, n_blocks=args.blocks, interm_depths=args.channels, bottleneck=args.use_bottleneck_layers, n_out_linear=2, dropout=0.5, const=0.01)
            model = model.to(device)
            #model = torch.nn.DataParallel(model).to(device)
        else:
            model = adni_models.ResNet(in_depth=1, n_blocks=args.blocks, interm_depths=args.channels, bottleneck=args.use_bottleneck_layers, n_out_linear=2, dropout=0.5)
            model = model.to(device)
            #model = torch.nn.DataParallel(model).to(device)
    else:
        raise NotImplementedError

    # Loading the model
    model.load_state_dict(torch.load(path))
    model.name = name

    if args.dataset_name == 'German':
        flag_latent = False
        X, Y, C, G = get_data(args, device, model, dataloader, flag_latent)
        Gdz = (G*10).astype(int) # Discretizing G
        compute_equivar_measure_gap(X, Y, C, Gdz, flag_latent, logf)
    elif args.dataset_name == 'Adult':
        flag_latent = False
        X, Y, C, G = get_data(args, device, model, dataloader, flag_latent)
        Gdz = (G*10).astype(int) # Discretizing G
        np.random.seed(args.seed)
        Xs, Ys, Cs, Gs = get_samples(X, Y, C, Gdz)
        compute_equivar_measure_gap(Xs, Ys, Cs, Gs, flag_latent, logf)
    elif args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP':
        flag_latent = False
        X, Y, C, G = get_data(args, device, model, dataloader, flag_latent)
        Gdz = (G*10).astype(int) # Discretizing G
        np.random.seed(args.seed)        
        Xs, Ys, Cs, Gs = get_samples(X, Y, C, Gdz)
        compute_equivar_measure_gap(Xs, Ys, Cs, Gs, flag_latent, logf)
    else:
        raise NotImplementedError
