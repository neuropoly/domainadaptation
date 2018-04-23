import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from medicaltorch.models import Unet, NoPoolASPP
from medicaltorch.datasets import SCGMChallenge2D
from medicaltorch.datasets import mt_collate
from medicaltorch.losses import dice_loss

import medicaltorch.metrics as metrics
import medicaltorch.transforms as mt_transform

import torchvision as tv

from tqdm import *
from tensorboardX import SummaryWriter

def decay_poly_lr(current_epoch, num_epochs, initial_lr):
    initial_lrate = initial_lr
    factor = 1.0 - (current_epoch / num_epochs)
    lrate = initial_lrate * np.power(factor, 0.9)
    return lrate


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr * cosine_rampdown(current_epoch, num_epochs)


def decay_constant_lr(current_epoch, num_epochs, initial_lr):
    return initial_lr


def get_current_consistency_weight(weight, epoch, rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return weight * sigmoid_rampup(epoch, rampup)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def adjust_learning_rate(optimizer, epoch, step_in_epoch,
                         total_steps_in_epoch, initial_lr, rampup_begin):
    lr = initial_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = linear_rampup(epoch, 15) * (initial_lr - rampup_begin) + rampup_begin

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ctx, ema=False):
    drop_rate = ctx["drop_rate"]
    bn_momentum = ctx["bn_momentum"]
    model = Unet(drop_rate=drop_rate, bn_momentum=bn_momentum)
    model = nn.DataParallel(model)
    model.cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def create_loader(ctx, dataset, source=True):
    batch_size = ctx["source_batch_size"] if source else ctx["target_batch_size"]
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=ctx["num_workers"],
                                       collate_fn=mt_collate,
                                       pin_memory=True)


def cmd_train(ctx):
    source_train = SCGMChallenge2D(ctx["rootdir_gmchallenge"], # 1 - 3 = train
                                   site_ids=range(1, 4), subj_ids=range(1, 3), rater_ids=[4])
    target_train = SCGMChallenge2D(ctx["rootdir_gmchallenge"], # 1 - 3 = train
                                   site_ids=[4], subj_ids=range(1,3), rater_ids=[4])

    source_val = SCGMChallenge2D(ctx["rootdir_gmchallenge"],
                                 site_ids=range(1,4), subj_ids=range(8, 11), rater_ids=[4])
    target_val = SCGMChallenge2D(ctx["rootdir_gmchallenge"],
                                 site_ids=[4], subj_ids=range(8, 11), rater_ids=[4])

    source_train_mean, source_train_std = source_train.compute_mean_std()
    target_train_mean, target_train_std = target_train.compute_mean_std()
    # if sys.version_info >= (3, 0):
    #     source_train_mean, source_train_std = [source_train_mean]*ctx["source_batch_size"], [source_train_std]*ctx["source_batch_size"]
    #     target_train_mean, target_train_std = [target_train_mean]*ctx["target_batch_size"], [target_train_std]*ctx["target_batch_size"]

    #TODO apply different dropout, noise and translation
    source_transform = tv.transforms.Compose([
        # mt_transform.ToPIL(),
        mt_transform.CenterCrop2D((200, 200)),
        mt_transform.ToTensor(),
        mt_transform.Normalize(source_train_mean, source_train_std),
    ])

    target_transform = tv.transforms.Compose([
        mt_transform.CenterCrop2D((200, 200)),
        mt_transform.ToTensor(),
        mt_transform.Normalize(target_train_mean, target_train_std),
    ])
    #
    # #TODO add setter to transform
    source_train.transform = source_val.transform = source_transform
    target_train.transform = target_val.transform = target_transform

    supervised_only = ctx["supervised_only"]
    use_consistency = ctx["use_consistency"]
    num_epochs = ctx["num_epochs"]
    num_workers = ctx["num_workers"]
    experiment_name = ctx["experiment_name"]
    cons_weight = ctx["cons_weight"]
    initial_lr = ctx["initial_lr"]
    consistency_rampup = ctx["consistency_rampup"]

    if "constant" in ctx["decay_lr"]:
        decay_lr_fn = decay_constant_lr

    if "poly" in ctx["decay_lr"]:
        decay_lr_fn = decay_poly_lr

    if "cosine" in ctx["decay_lr"]:
        decay_lr_fn = cosine_lr

    source_train_loader = create_loader(ctx, source_train)
    target_train_loader = create_loader(ctx, target_train, source=False)
    source_val_loader = create_loader(ctx, source_val)
    target_val_loader = create_loader(ctx, target_val, source=False)

    model = create_model(ctx)
    model_ema = create_model(ctx, ema=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=ctx["weight_decay"])

    writer = SummaryWriter(log_dir="log_{}".format(experiment_name))

    for epoch in tqdm(range(1, num_epochs+1), desc="Epochs"):
        start_time = time.time()

        # Rampup -----
        initial_lr_rampup = ctx["initial_lr_rampup"]

        if initial_lr_rampup > 0:
            if epoch <= initial_lr_rampup:
                lr = initial_lr * sigmoid_rampup(epoch, initial_lr_rampup)
            else:
                lr = decay_lr_fn(epoch-initial_lr_rampup,
                                 num_epochs-initial_lr_rampup,
                                 initial_lr)
        else:
            lr = decay_lr_fn(epoch, num_epochs, initial_lr)

        writer.add_scalar('learning_rate', lr, epoch)

        for param_group in optimizer.param_groups:
            tqdm.write("Learning Rate: {:.6f}".format(lr))
            param_group['lr'] = lr

        consistency_weight = get_current_consistency_weight(cons_weight, epoch, consistency_rampup)
        writer.add_scalar('consistency_weight', consistency_weight, epoch)

        # Train mode
        model.train()
        model_ema.train()

        task_loss = 0
        target_train_iter = iter(target_train_loader)
        j = 0
        for i, source_input in enumerate(source_train_loader): #TODO presumes first_set > second_set
            try:
                target_input = target_train_iter.next()
            except:
                target_train_iter = iter(target_train_loader)
                target_input = target_train_iter.next()

            s_image, s_gt = source_input["input"], source_input["gt"]
            t_image = target_input["input"]

            s_var_image = torch.autograd.Variable(s_image).cuda()
            s_var_gt = torch.autograd.Variable(s_gt).cuda(async=True)

            t_var_image = torch.autograd.Variable(t_image).cuda()

            student_source_out = model(s_var_image)
            student_target_out = model(t_var_image)
            teacher_target_out = model_ema(t_var_image)
            #TODO fixed in torch 0.4
            teacher_target_out_nograd = torch.autograd.Variable(torch.from_numpy(teacher_target_out.data.cpu().numpy()),
                                                                requires_grad=False).cuda()

            task_loss = dice_loss(student_source_out, s_var_gt)
            consistency_loss = consistency_weight * F.binary_cross_entropy(student_target_out,
                                                                           teacher_target_out_nograd)

            loss = task_loss + consistency_loss
            optimizer.zero_grad()
            loss.backward()
            if ctx["clip_norm"]:
                norm = nn.utils.clip_grad_norm(model.parameters(), ctx["clip_norm_value"])

            optimizer.step()



def run_main():
    if len(sys.argv) <= 1:
        print("\ndomainadapt [config filename].json\n")
        return

    try:
        with open(sys.argv[1], "r") as fhandle:
            ctx = json.load(fhandle)
    except FileNotFoundError:
        print("\nFile {} not found !\n".format(sys.argv[1]))
        return

    command = ctx["command"]

    if command == 'train':
        cmd_train(ctx)


if __name__ == '__main__':
    run_main()
