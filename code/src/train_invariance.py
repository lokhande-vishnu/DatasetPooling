import argparse
from src import dataloader as mydatasets, model as models
from src import adni_models
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt

from src.average_meter import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
sys.path.append('../..')

def mmd_lap_loss(args, mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float()
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
    
    return loss

def invar_epoch(args, device, epoch, model_taunet, model_bnet, opt, dataloader, writer, tag='train'):
    loss_logger = AverageMeter()
    recons_loss_logger = AverageMeter()
    pred_loss_logger = AverageMeter()
    comp_loss_logger = AverageMeter()
    mu_logger = AverageMeter()
    sigma_logger = AverageMeter()
    prior_loss_logger = AverageMeter()
    train = tag == 'train'

    model_taunet.eval()
    if train:
        model_bnet.train()
    else:
        model_bnet.eval()

    total_steps = len(dataloader.dataset)//args.batch_size
    y_correct = 0
    y_total = 0
    y_true_pos = 0
    y_pos = 0
    start_time = time.time()

    for idx, (x, y, c, g) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        g = g.to(device)

        with torch.no_grad():
            li_latent = model_taunet.encodeProject(x)
            gi_latent = model_taunet.encodeTauNet(x, g.unsqueeze(-1))

        recons, pred_logits, latent = model_bnet(li_latent, gi_latent)

        if recons is not None:
            recons_loss = F.mse_loss(recons, x)
        else:
            recons_loss = torch.tensor(0).to(device)
                
        if pred_logits is not None:
            pred_loss = F.cross_entropy(pred_logits, y)
        else:
            pred_loss = torch.tensor(0).to(device)

        comp_loss = mmd_lap_loss(args, latent, c)

        loss = args.comp_lambda * recons_loss + pred_loss + args.alpha * (args.comp_lambda + args.beta) * comp_loss

        if args.add_prior:
            loss += args.beta * prior_loss
            prior_loss_logger.update(prior_loss.item())
                
        mu_logger.update(latent.norm(dim=-1).mean())

        # Log the losses
        loss_logger.update(loss.item())
        recons_loss_logger.update(recons_loss.item())
        if pred_logits is not None:
            pred_loss_logger.update(pred_loss.item())
        comp_loss_logger.update(comp_loss.item())

        pred = torch.argmax(pred_logits, 1)
        y_correct += torch.sum(pred == y)
        y_total += x.size(0)
        y_pos += torch.sum(y)
        y_true_pos += torch.sum(y[pred == 1])

        if idx % args.log_step == 0:
            #log_loss(epoch, args.num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

    model_name = 'invar_'
    accuracy = y_correct * 100.0 / y_total
    precision = y_true_pos * 100.0 / y_pos
    comp_loss_avg = comp_loss_logger.avg
    recons_loss_avg = recons_loss_logger.avg
    print(tag, 'accuracy:', accuracy.item(), 'recons_loss:', recons_loss_avg, 'comp_loss:', comp_loss_avg)
    
    writer.add_scalar(model_name + 'acc/' + tag, accuracy, epoch)
    writer.add_scalar(model_name + 'recons_loss/' + tag, recons_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'pred_loss/' + tag, pred_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'comp_loss/' + tag, comp_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'mu/' + tag, mu_logger.avg, epoch)
    writer.add_scalar(model_name + 'loss/' + tag, loss_logger.avg, epoch)
    return accuracy, recons_loss_avg, comp_loss_avg

def train_invar(args, device, model_path, logf, model_taunet, model_bnet, opt, trainloader, valloader, testloader, writer):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    best_val_acc = 0    
    for epoch in range(1, args.num_epochs + 1):
        invar_epoch(args, device, epoch, model_taunet, model_bnet, opt, trainloader, writer, tag='train')

        with torch.no_grad():            
            val_acc, val_recons, val_comp = invar_epoch(args, device, epoch, model_taunet, model_bnet, opt, valloader, writer, tag='val')
            if testloader is not None:
                test_acc, test_recons, test_comp = invar_epoch(args, device, epoch, model_taunet, model_bnet, opt, testloader, writer, tag='test')
            else:
                test_acc, test_recons, test_comp = None, None, None
            print(' ')
        
        if val_acc > best_val_acc:
            name = 'Invar_best_val_acc'
            model_bnet.name = name
            path = os.path.join(model_path, name + '.pth')
            torch.save(model_bnet.state_dict(), path)
            best_val_acc = val_acc
            message = 'INVAR Best val_acc{} val_recons{} val_comp{}\n INVAR test_acc{} test_recons{} test_comp{}\n Saving model{}\n'.format(
                best_val_acc, val_recons, val_comp, test_acc, test_recons, test_comp, path)
            print(message)
            logf.write(message + '\n')
        if epoch % args.save_step == 0:
            name = 'Invar_ckpt_' + str(epoch)
            path = os.path.join(model_path, name + '.pth')
            model_bnet.name = name
            torch.save(model_bnet.state_dict(), path)
        lr_scheduler.step()
        if args.alpha < args.alpha_max:
            args.alpha *= args.alpha_gamma
    name = 'Invar'
    model_bnet.name = name
    path = os.path.join(model_path, name + '.pth')
    torch.save(model_bnet.state_dict(), path)


def run_invariance(args, device, model_path, logf, trainset, valset, testset, writer, equivar_model_name):

    # Loading the dataset
    if args.dataset_name == 'German':
        drop_last = True
    else:
        drop_last = False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              drop_last=drop_last)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    if testset is not None:
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    else:
        testloader = None
    dummy_x, _, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)

    # Loading the equivar model
    if args.dataset_name == 'German':
        model_taunet = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.01).to(device)
    elif args.dataset_name == 'Adult':
        model_taunet = models.TauNetEncDec(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=0, const=0.1).to(device)
    elif args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP':
        model_taunet = adni_models.TauResNet(in_depth=1, n_blocks=args.blocks, interm_depths=args.channels, bottleneck=args.use_bottleneck_layers, n_out_linear=2, dropout=0.5, const=0.01)
        model_taunet = model_taunet.to(device)
        #model_taunet = torch.nn.DataParallel(model_taunet).to(device)
    else:
        raise NotImplementedError
        
    path = os.path.join(model_path, equivar_model_name + '.pth')
    model_taunet.load_state_dict(torch.load(path))
    model_taunet.name = equivar_model_name
    model_taunet.eval()

    # Loading the invar model
    torch.manual_seed(args.seed)
    if args.dataset_name == 'German':
        model_bnet = models.BNetEncDec(device=device, latent_dim=args.latent_dim, output_dim=input_dim, feature_dim=0).to(device)
    elif args.dataset_name == 'Adult':
        model_bnet = models.BNetEncDec(device=device, latent_dim=args.latent_dim, output_dim=input_dim, feature_dim=0).to(device)
    elif args.dataset_name == 'ADNI' or args.dataset_name == 'ADCP':
        model_bnet = adni_models.BNetPred(device=device, interm_depths=args.channels, n_out_linear=2, dropout=0.5)
        model_bnet = model_bnet.to(device)
        #model_bnet = torch.nn.DataParallel(model_bnet).to(device)
    else:
        raise NotImplementedError

    opt = optim.Adam(model_bnet.parameters(), lr=args.lr)
    train_invar(args, device, model_path, logf, model_taunet, model_bnet, opt, trainloader, valloader, testloader, writer)
