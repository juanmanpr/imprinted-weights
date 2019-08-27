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

import pdb

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from utils import ortho_reg
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='CUB_200_2011',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--ortho', dest='ortho', action='store_true',
                    help='activate orthogonality reg')
parser.add_argument('--ortho-lambda', '--lambda', default=1e-4, type=float,
                    metavar='W', help='lambda ortho (default: 1e-4)')       
parser.add_argument('-n', '--net-type', default='default', type=str, choices=['googlenet','default'],
                    help='network type (googlenet,default)')                    
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: weight_gen)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--random', action='store_true', help='whether use random novel weights')
parser.add_argument('--num-sample', default=1, type=int,
                    metavar='N', help='number of novel sample (default: 1)')
parser.add_argument('--test-novel-only', action='store_true', help='whether only test on novel classes')


best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    conf_name = args.data
    conf_name += '_ortho' if args.ortho else '' 
    conf_name += ('_' + args.net_type) if args.net_type != 'default' else ''
    
    args.checkpoint = os.path.join(args.checkpoint, conf_name, 'dynamic_checkpoint')

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    model = models.DynamicNet(extractor_type=args.net_type).cuda()
    weight_generator = models.WeightGenerator().cuda()

    print('==> Reading from model checkpoint..')
    assert os.path.isfile(args.model), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model checkpoint '{}' (epoch {})"
            .format(args.model, checkpoint['epoch']))
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = loader.EpisodicLoader(
        args.data,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )

    novel_dataset = loader.ImageLoader(
        args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        train=True, num_classes=200, 
        num_train_sample=args.num_sample, 
        novel_only=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        loader.ImageLoader(args.data, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]), num_classes=200, novel_only=args.test_novel_only),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
 
    novel_loader = torch.utils.data.DataLoader(
        novel_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True) 


    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(weight_generator.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.94)

    novel_weights_validation = get_novel_weights(novel_loader, model, weight_generator)

    title = 'Episodic training of weight generator'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        validate(val_loader, model, novel_weights_validation, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        # train for one epoch
        train_loss, train_acc = train(train_loader, model, weight_generator, criterion, optimizer, epoch)

        # evaluate on validation set
        test_loss, test_acc = validate(val_loader, model, novel_weights_validation, criterion)

        # append logger file
        logger.append([lr, train_loss, test_loss, train_acc, test_acc])

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_prec1
        best_prec1 = max(test_acc, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_prec1)

def get_novel_weights(novel_loader, model, weight_generator):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    weight_generator.eval()
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
        
    gen_weight = weight_generator(new_weight.cuda())
    return gen_weight

def train(train_loader, model, weight_generator, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Training  ', max=len(train_loader))
    for batch_idx, (base_samples, base_labels, fake_novel_samples, fake_novel_query, fake_novel_labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
                
        base_samples            = torch.cat(base_samples).cuda()
        base_labels             = torch.cat(base_labels).cuda()
        fake_novel_samples      = torch.cat(fake_novel_samples).cuda()
        fake_novel_labels       = torch.cat(fake_novel_labels).cuda()
        fake_novel_query        = torch.cat(fake_novel_query).cuda()
            
        fake_train = model.extract(fake_novel_samples)
        unique_novel_labels = torch.unique(fake_novel_labels)
        
        new_weight = torch.zeros(unique_novel_labels.shape[0], 256)
        for i, f_l in enumerate(unique_novel_labels):
            tmp = fake_train[fake_novel_labels == f_l].mean(0)
            new_weight[i] = tmp / tmp.norm(p=2)
        new_weight = new_weight.cuda()
        
        gen_weight = weight_generator(new_weight)
        
        # compute output of the sampled 10-way classification problem
        input = torch.cat((base_samples, fake_novel_query))
        unique_base_labels = torch.unique(base_labels)
        output = model(input, base_class_indexes = unique_base_labels, 
                        novel_class_classifiers = gen_weight,
                        detach_feature=True)
        
        lst_lab = np.repeat(list(range(10)), 5)
        target = torch.LongTensor(lst_lab).cuda()
        
        loss = criterion(output, target)


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        model.weight_norm()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)
    
def validate(val_loader, model, generated_weights, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    bar = Bar('Testing   ', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input, base_class_indexes = None,
                             novel_class_classifiers = generated_weights) 
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
            
if __name__ == '__main__':
    main()