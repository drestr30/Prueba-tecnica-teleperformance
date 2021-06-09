# encoding: utf-8
"""
The emotion recognition Torch models implementation.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from  torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from test import test, plot_confusion_matrix
from tqdm import tqdm
import shutil

SAVE_PATH = './saved_models'


def fit(model, epochs, train_loader, valid_loader, config):

    if 'l2' not in config:
        config['l2'] = None

    if 'lr' not in config:
        config['lr'] = 0.001

    if config['optim'] is 'SGD':
        optimizer = optim.SGD(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'], momentum=config['momentum'])
    else:
        optimizer = optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['l2'])
    optimizer.zero_grad()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                     patience=round(config['patience']/4), verbose=True)

    if config['loss'] is 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss().cuda()
    else:
        loss_func = nn.BCELoss(reduce='sum')

    train_loss = []
    train_acc = []
    val_loss = []
    val_accuracies = []
    no_best_count = 0
    best_acc1 = 0

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train(epoch, model, train_loader, optimizer, loss_func)
        # evaluate on validation set
        v_loss, v_acc = validate(valid_loader, model, loss_func)
        validation_loss = v_loss.avg
        validation_acc = v_acc.avg
        # remember best acc and save checkpoint
        is_best = validation_acc > best_acc1
        best_acc1 = max(validation_acc, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config['modelName'],
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            # 'optimizer': optimizer.state_dict(),
        }, is_best, f'{config["modelName"]}_best', f'{config["modelName"]}_checkpoint')
        scheduler.step(t_loss.avg)
        if is_best:
            no_best_count = 0
        else: no_best_count = no_best_count + 1

        if no_best_count > config['patience']:
            print('Early stop finish training after {} epochs'.format(epoch))
            break

        train_loss.append(t_loss.avg)
        train_acc.append(t_acc.avg)
        val_loss.append(v_loss.avg)
        val_accuracies.append(v_acc.avg)

    plot_training_stats(train_loss, val_loss,train_acc, val_accuracies,
                        save='{}/{}_plot'.format(SAVE_PATH, config['modelName']), config=config)

    return best_acc1, epoch

def train(epoch, model, train_loader, optimizer, loss_func):
    losses = AverageMeter()
    top_acc = AverageMeter()

    model.train()
    pbar = tqdm(train_loader)
    # i=0
    for batch in pbar:
        # if i >15:
        #     break
        # i += 1
        data, target = batch
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # data, target = Variable(data), Variable(target)

        output = model(data)
        loss = loss_func(output, target)
        acc = accuracy(output, target)
        # measure accuracy and record loss

        losses.update(loss.item(), data.size(0))
        top_acc.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar_text = 'Epoch: {0} Loss {loss.val:.4f} ({loss.avg:.4f}) Acc1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, loss=losses, top1=top_acc)
        pbar.set_description(bar_text)

    return losses, top_acc

def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top_acc = AverageMeter()
    # switch to evaluate mode
    model.eval()
    pbar = tqdm(val_loader)

    with torch.no_grad():
        # i = 0
        for (input, target) in pbar:
            # if i > 15:
            #     break
            # i += 1
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top_acc.update(acc1, input.size(0))

            bar_text = 'Validation: Loss {loss.val:.4f} ({loss.avg:.4f}) Acc1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses, top1=top_acc)
            pbar.set_description(bar_text)

    return losses, top_acc

def save_checkpoint(state, is_best, bestname='model_best', filename='checkpoint'):
    torch.save(state, '{}/{}.pth.tar'.format(SAVE_PATH, filename))
    if is_best:
        shutil.copyfile('{}/{}.pth.tar'.format(SAVE_PATH, filename), '{}/{}.pth.tar'.format(SAVE_PATH, bestname))

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.clone().detach().cpu()
    target = target.clone().detach().cpu()
    output = np.argmax(output, axis=-1)
    # output = (output > 0.5).float()
    # target = (target > 0.5).float()
    acc = accuracy_score(target, output)
    return acc

def plot_training_stats(t_loss, v_loss, t_acc, v_acc, save=False, config=None):
    plt.figure()
    plt.plot(t_loss)
    plt.plot(t_acc)
    plt.plot(v_loss)
    plt.plot(v_acc)
    plt.title('{} model statistics'.format(config['modelName']))
    plt.ylabel('loss - acc')
    plt.xlabel('epoch')
    plt.legend(['T loss', 'T acc', 'V loss', 'V_acc'], loc='upper left')

    if save is not None:
        plt.savefig(f'{save}.png')
    else:
        plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

