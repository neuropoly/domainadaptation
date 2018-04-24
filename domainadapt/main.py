import sys
import json
import time
import numpy as np
from collections import defaultdict

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
import torchvision.utils as vutils

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


def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def create_model(ctx, ema=False):
    drop_rate = ctx["drop_rate"]
    bn_momentum = ctx["bn_momentum"]
    model = Unet(drop_rate=drop_rate, bn_momentum=bn_momentum)
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


def validation(model, model_ema, loader, writer, metric_fns, epoch, prefix):
    val_loss = 0.0
    ema_val_loss = 0.0
    num_samples = 0
    num_steps = 0

    result_dict = defaultdict(float)
    result_ema_dict = defaultdict(float)

    for i, batch in enumerate(loader):
        image, gt = batch["input"], batch["gt"]
        var_image = torch.autograd.Variable(image, volatile=True).cuda()
        var_gt = torch.autograd.Variable(gt, volatile=True).cuda(async=True)

        model_out = model(var_image)
        val_loss = dice_loss(model_out, var_gt)

        model_ema_out = model_ema(var_image)
        model_ema_out_nograd = torch.autograd.Variable(model_ema_out.detach().data, requires_grad=False)
        ema_task_loss = dice_loss(model_ema_out_nograd, var_gt)
        ema_val_loss += ema_task_loss.data[0]

        gt_masks = gt.numpy().astype(np.uint8)
        gt_masks = gt_masks.squeeze(axis=1)

        preds = model_out.data.cpu().numpy()
        preds = threshold_predictions(preds)
        preds = preds.astype(np.uint8)
        preds = preds.squeeze(axis=1)

        for metric_fn in metric_fns:
            for prediction, ground_truth in zip(preds, gt_masks):
                res = metric_fn(prediction, ground_truth)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res

        preds_ema = model_ema_out.data.cpu().numpy()
        preds_ema = threshold_predictions(preds_ema)
        preds_ema = preds_ema.astype(np.uint8)
        preds_ema = preds_ema.squeeze(axis=1)

        for metric_fn in metric_fns:
            for prediction, ground_truth in zip(preds_ema, gt_masks):
                res = metric_fn(prediction, ground_truth)
                dict_key = 'val_ema_{}'.format(metric_fn.__name__)
                result_ema_dict[dict_key] += res

        num_samples += len(preds)
        num_steps += 1

    #TODO check if validation is not inheriting anything from the for loop
    for key, val in result_dict.items():
        result_dict[key] = val / num_samples
    for key, val in result_dict.items():
        result_ema_dict[key] = val / num_samples

    val_loss_avg = val_loss / num_steps

    writer.add_scalars(prefix + '_' + 'metrics', result_dict, epoch)

    ema_val_loss_avg = 0.0
    ema_val_loss_avg = ema_val_loss / num_steps
    tqdm.write("Ema Target Val Loss: {:.6f}".format(ema_val_loss_avg))
    writer.add_scalars(prefix + '_metrics', result_ema_dict, epoch)

    writer.add_scalars(prefix + '_losses', {'val_loss_source': val_loss_avg,
                      'ema_val_loss_source': ema_val_loss_avg},
                      epoch)


def cmd_train(ctx):
    global_step = 0

    num_epochs = ctx["num_epochs"]
    experiment_name = ctx["experiment_name"]
    cons_weight = ctx["cons_weight"]
    initial_lr = ctx["initial_lr"]
    consistency_rampup = ctx["consistency_rampup"]
    weight_decay = ctx["weight_decay"]

    if "constant" in ctx["decay_lr"]:
        decay_lr_fn = decay_constant_lr

    if "poly" in ctx["decay_lr"]:
        decay_lr_fn = decay_poly_lr

    if "cosine" in ctx["decay_lr"]:
        decay_lr_fn = cosine_lr

    source_train = SCGMChallenge2D(ctx["rootdir_gmchallenge"], # 1 - 3 = train
                                   site_ids=range(1, 4), subj_ids=range(1, 3), rater_ids=[4])
    target_train = SCGMChallenge2D(ctx["rootdir_gmchallenge"], # 1 - 3 = train
                                   site_ids=[4], subj_ids=range(1, 3), rater_ids=[4])

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

    source_train_loader = create_loader(ctx, source_train)
    target_train_loader = create_loader(ctx, target_train, source=False)
    source_val_loader = create_loader(ctx, source_val)
    target_val_loader = create_loader(ctx, target_val, source=False)

    model = create_model(ctx)
    model_ema = create_model(ctx, ema=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)

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

        task_loss_total = 0.0
        consistency_loss_total = 0.0
        loss_total = 0.0

        target_train_iter = iter(target_train_loader)
        num_steps = 0
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
            task_loss_total += task_loss.data[0]

            consistency_loss = consistency_weight * F.binary_cross_entropy(student_target_out,
                                                                           teacher_target_out_nograd)
            consistency_loss_total += consistency_loss.data[0]

            loss = task_loss + consistency_loss
            loss_total += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            if ctx["clip_norm"]:
                norm = nn.utils.clip_grad_norm(model.parameters(), ctx["clip_norm_value"])

            optimizer.step()

            num_steps += 1
            global_step += 1

            if epoch <= initial_lr_rampup:
                update_ema_variables(model, model_ema, ctx["ema_alpha"], global_step)
            else:
                update_ema_variables(model, model_ema, ctx["ema_alpha_late"], global_step)

        loss_avg = loss_total / num_steps
        task_loss_avg = task_loss_total / num_steps
        consistency_loss_avg = consistency_loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Consistency Weight: {:.6f}".format(consistency_weight))
        tqdm.write("Composite Loss: {:.6f}".format(loss_avg))
        tqdm.write("Class Loss: {:.6f}".format(task_loss_avg))
        tqdm.write("Consistency Loss: {:.6f}".format(consistency_loss_avg))

        # Write sample images
        if ctx["write_images"] and epoch % ctx["write_images_interval"] == 0:
            try:
                plot_img = vutils.make_grid(student_source_out.data,
                                            normalize=True, scale_each=True)
                writer.add_image('Model Source Prediction', plot_img, epoch)
                plot_img = vutils.make_grid(s_var_image.data,
                                            normalize=True, scale_each=True)
                writer.add_image('Model Source Input', plot_img, epoch)
                plot_img = vutils.make_grid(student_target_out.data,
                                            normalize=True, scale_each=True)
                writer.add_image('Model Target Prediction', plot_img, epoch)
                plot_img = vutils.make_grid(t_var_image.data,
                                            normalize=True, scale_each=True)
                writer.add_image('Model Target Input', plot_img, epoch)


            except:
                tqdm.write("*** Error writing images ***")

        writer.add_scalars('losses', {'composite_loss': loss_avg,
                           'task_loss': task_loss_avg,
                           'consistency_loss': consistency_loss_avg},
                           epoch)

        # Evaluation ####################################################################

        # Evaluation mode
        model.eval()
        model_ema.eval()

        metric_fns = [metrics.dice_score, metrics.jaccard_score, metrics.hausdorff_score,
                      metrics.precision_score, metrics.recall_score,
                      metrics.specificity_score, metrics.intersection_over_union,
                      metrics.accuracy_score]

        validation(model, model_ema,
                   source_val_loader,
                   writer, metric_fns,
                   epoch, 'source_val')
        validation(model, model_ema,
                   target_val_loader,
                   writer, metric_fns,
                   epoch, 'target_val')

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))


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
