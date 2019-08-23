import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import models
import loader

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
            
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='CUB_200_2011',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-o', '--ortho', dest='ortho', action='store_true',
                    help='activate orthogonality reg')
parser.add_argument('--ortho-lambda', '--lambda', default=1e-4, type=float,
                    metavar='W', help='lambda ortho (default: 1e-4)')       
parser.add_argument('-n', '--net-type', default='default', type=str, choices=['googlenet','default'],
                    help='network type (googlenet,default)')                                 
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: imprint_checkpoint)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num-sample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')
parser.add_argument('--aug', action='store_true', help='whether use data augmentation during training')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    conf_name = args.data
    conf_name += '_ortho' if args.ortho else '' 
    conf_name += ('_pre_' + args.net_type) if args.net_type != 'default' else ''
    
    args.checkpoint = os.path.join(args.checkpoint, conf_name, 'imprint_checkpoint')
    print(args.data)
    print(args.checkpoint)

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.Net(extractor_type=args.net_type).cuda()


    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True


    ## check orthogonality
    #import numpy as np
    #W = model.classifier.fc.weight.data
    #print(W.size())
    
    #d_list = []
    #for i in range(W.size(0)):
    #    for j in range(i,W.size(0)):
        
    #        if i==j:
    #            continue
             
    #        r1 = W[i]
    #        r2 = W[j]
                
            #r1 = torch.nn.functional.normalize(r1,p=2,dim=0)
            #r2 = torch.nn.functional.normalize(r2,p=2,dim=0)
    #        d = torch.dot(r1,r2)
    #        d_list.append(d.item())
            
    #d_list = np.array(d_list)
    #np.save('ortho_dotprod_hist.npy', d_list)
    # 
    #import matplotlib.pyplot as plt
    #plt.hist(d_list, bins=200, range=(-0.25,0.25), normed=True)  # arguments are passed to np.histogram
    #plt.title('Dot product histogram $\sigma$ = {:02f}'.format(np.std(d_list)))
    #plt.show() 

    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    novel_trasforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]) if not args.aug else transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    novel_dataset = loader.ImageLoader(
        args.data,
        novel_trasforms,
        train=True, num_classes=200, 
        num_train_sample=args.num_sample, 
        novel_only=True, aug=args.aug)

    novel_loader = torch.utils.data.DataLoader(
        novel_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_classes=200, novel_only=args.test_novel_only),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    imprint(novel_loader, model)
    test_acc = validate(val_loader, model)

    save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': test_acc,
        }, checkpoint=args.checkpoint)


from sklearn.metrics import confusion_matrix
def compute_conf_matrix(scores, labels, interest_indexes):
    pred_labels = torch.max(scores, 1)[1].cpu().numpy()
    real_labels = labels.cpu().numpy()
    
    m = confusion_matrix(real_labels, pred_labels)
    return m
    
def imprint(novel_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    bar = Bar('Imprinting', max=len(novel_loader))
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(novel_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()

            # compute output
            output = model.extract(input)

            if batch_idx == 0:
                output_stack = output
                target_stack = target
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, target), 0)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(novel_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td
                        )
            bar.next()
        bar.finish()
    
    new_weight = torch.zeros(100, 256)
    for i in range(100):
        tmp = output_stack[target_stack == (i + 100)].mean(0) if not args.random else torch.randn(256)
        new_weight[i] = tmp / tmp.norm(p=2)
    weight = torch.cat((model.classifier.fc.weight.data, new_weight.cuda()))
    model.classifier.fc = nn.Linear(256, 200, bias=False)
    model.classifier.fc.weight.data = weight
    
def validate(val_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_novel = AverageMeter()
    top5_novel = AverageMeter()
    
    top1_base = AverageMeter()
    top5_base = AverageMeter()
        
    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        
        outputs = []
        targets = []
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)


            # store all
            outputs.append(output)
            targets.append(target)

            # measure accuracy
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            
            #print(target)
            #print('target size: ',target.size())            
            #print('target novel size: ',target[target>=100].size())
            #print('target base size: ',target[target<100].size())            
         
            
            prec1_novel, prec5_novel = accuracy(output[target>=100], target[target>=100], topk=(1, 5))
            prec1_base, prec5_base   = accuracy(output[target<100], target[target<100], topk=(1, 5))
            
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            top1_novel.update(prec1_novel.item(), input[target>=100].size(0))
            top5_novel.update(prec5_novel.item(), input[target>=100].size(0))
            
            top1_base.update(prec1_base.item(), input[target<100].size(0))
            top5_base.update(prec5_base.item(), input[target<100].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

             # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} | top5: {top5: .4f} | top1 novel: {top1_novel: .4f} | top1 base: {top1_base: .4f} | top1 upper bound: {top1_avg: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        top1_novel=top1_novel.avg,
                        top1_base=top1_base.avg,
                        top1_avg=top1_novel.avg*.5+top1_base.avg*.5
                        )
            bar.next()
        bar.finish()
        
        # conf matrix
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)        
        
        conf_m = compute_conf_matrix(outputs, targets, [])
        df_cm = pd.DataFrame(conf_m, index = [i for i in range(200)],
              columns = [i for i in range(200)])
        plt.figure(figsize = (200,200))
        #sn.heatmap(df_cm, annot=True)
        sn.heatmap(df_cm, annot=False)
        plt.show()         
        
    return top1.avg


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

if __name__ == '__main__':
    main()
