import sys
import json
import time
import numpy as np
from collections import defaultdict
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

import medicaltorch.filters as mt_filters
import medicaltorch.losses as mt_losses
import medicaltorch.models as mt_models
import medicaltorch.metrics as mt_metrics
import medicaltorch.datasets as mt_datasets
import medicaltorch.transforms as mt_transforms

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
    model = mt_models.Unet(drop_rate=drop_rate,
                           bn_momentum=bn_momentum)

    if ema:
        for param in model.parameters():
            param.detach_()

    return model.cuda()


def validation(model, model_ema, loader, writer,
               metric_fns, epoch, ctx, prefix):
    val_loss = 0.0
    ema_val_loss = 0.0

    num_samples = 0
    num_steps = 0

    result_dict = defaultdict(float)
    result_ema_dict = defaultdict(float)

    for i, batch in enumerate(loader):
        input_data, gt_data = batch["input"], batch["gt"]

        input_data_gpu = input_data.cuda()
        gt_data_gpu = gt_data.cuda(async=True)

        with torch.no_grad():
            model_out = model(input_data_gpu)
            val_class_loss = mt_losses.dice_loss(model_out, gt_data_gpu)
            val_loss += val_class_loss.item()

            if not ctx["supervised_only"]:
                model_ema_out = model_ema(input_data_gpu)
                ema_val_class_loss = mt_losses.dice_loss(model_ema_out, gt_data_gpu)
                ema_val_loss += ema_val_class_loss.item()

        gt_masks = gt_data_gpu.cpu().numpy().astype(np.uint8)
        gt_masks = gt_masks.squeeze(axis=1)

        preds = model_out.cpu().numpy()
        preds = threshold_predictions(preds)
        preds = preds.astype(np.uint8)
        preds = preds.squeeze(axis=1)

        for metric_fn in metric_fns:
            for prediction, ground_truth in zip(preds, gt_masks):
                res = metric_fn(prediction, ground_truth)
                dict_key = 'val_{}'.format(metric_fn.__name__)
                result_dict[dict_key] += res

        if not ctx["supervised_only"]:
            preds_ema = model_ema_out.cpu().numpy()
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

    val_loss_avg = val_loss / num_steps

    for key, val in result_dict.items():
        result_dict[key] = val / num_samples

    if not ctx["supervised_only"]:
        for key, val in result_ema_dict.items():
            result_ema_dict[key] = val / num_samples

        ema_val_loss_avg = ema_val_loss / num_steps
        writer.add_scalars(prefix + '_ema_metrics', result_ema_dict, epoch)
        writer.add_scalars(prefix + '_losses', {
                           prefix + '_loss': val_loss_avg,
                           prefix + '_ema_loss': ema_val_loss_avg
                       },
                       epoch)
    else:
        writer.add_scalars(prefix + '_losses', {
                           prefix + '_loss': val_loss_avg,
                       },
                       epoch)

    writer.add_scalars(prefix + '_metrics', result_dict, epoch)

    
def linked_batch_augmentation(input_batch, preds_unsup):

    # Teach transformation
    teacher_transform = tv.transforms.Compose([
        mt_transforms.ToPIL(labeled=False),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3, labeled=False),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03),
                                   labeled=False),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(labeled=False),
    ])

    input_batch_size = input_batch.size(0)

    input_batch_cpu = input_batch.cpu().detach()
    input_batch_cpu = input_batch_cpu.numpy()

    preds_unsup_cpu = preds_unsup.cpu().detach()
    preds_unsup_cpu = preds_unsup_cpu.numpy()

    samples_linked_aug = []
    for sample_idx in range(input_batch_size):
        sample_linked_aug = {'input': [input_batch_cpu[sample_idx],
                                       preds_unsup_cpu[sample_idx]]}
        out = teacher_transform(sample_linked_aug)
        samples_linked_aug.append(out)

    samples_linked_aug = mt_datasets.mt_collate(samples_linked_aug)
    return samples_linked_aug


def cmd_train(ctx):
    global_step = 0

    num_workers = ctx["num_workers"]
    num_epochs = ctx["num_epochs"]
    experiment_name = ctx["experiment_name"]
    cons_weight = ctx["cons_weight"]
    initial_lr = ctx["initial_lr"]
    consistency_rampup = ctx["consistency_rampup"]
    weight_decay = ctx["weight_decay"]
    rootdir_gmchallenge_train = ctx["rootdir_gmchallenge_train"]
    rootdir_gmchallenge_test = ctx["rootdir_gmchallenge_test"]
    supervised_only = ctx["supervised_only"]

    if "constant" in ctx["decay_lr"]:
        decay_lr_fn = decay_constant_lr

    if "poly" in ctx["decay_lr"]:
        decay_lr_fn = decay_poly_lr

    if "cosine" in ctx["decay_lr"]:
        decay_lr_fn = cosine_lr

    # Xs, Ys = Source input and source label, train
    # Xt1, Xt2 = Target, domain adaptation, no label, different aug (same sample), train
    # Xv, Yv = Target input and target label, validation

    # Sample Xs and Ys from this
    source_train = mt_datasets.SCGMChallenge2DTrain(rootdir_gmchallenge_train,
                                                    slice_filter_fn=mt_filters.SliceFilter(),
                                                    site_ids=[1, 2], # Test = 1,2,3, train = 1,2
                                                    subj_ids=range(1, 11))

    # Sample Xt1, Xt2 from this
    unlabeled_filter = mt_filters.SliceFilter(filter_empty_mask=False)
    target_adapt_train = mt_datasets.SCGMChallenge2DTest(rootdir_gmchallenge_test,
                                                         slice_filter_fn=unlabeled_filter,
                                                         site_ids=[3], # 3 = train, 4 = test
                                                         subj_ids=range(11, 21))

    # Sample Xv, Yv from this
    target_validation = mt_datasets.SCGMChallenge2DTrain(rootdir_gmchallenge_train,
                                                         slice_filter_fn=mt_filters.SliceFilter(),
                                                         site_ids=[3], # 3 = train, 4 = test
                                                         subj_ids=range(1, 11))

    source_train_mean, source_train_std = source_train.compute_mean_std(True)
    target_adapt_train_mean, target_adapt_train_std = target_adapt_train.compute_mean_std(True)

    # Training source data augmentation
    source_transform = tv.transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.Normalize([source_train_mean], [source_train_std]),
    ])

    # Target adaptation data augmentation
    target_adapt_transform = tv.transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200), labeled=False),
        mt_transforms.ToTensor(),
        mt_transforms.Normalize([target_adapt_train_mean], [target_adapt_train_std]),
    ])

    # Target adaptation data augmentation
    target_val_adapt_transform = tv.transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ToTensor(),
        mt_transforms.Normalize([target_adapt_train_mean], [target_adapt_train_std]),
    ])

    source_train.set_transform(source_transform)

    target_adapt_train.set_transform(target_adapt_transform)
    target_validation.set_transform(target_val_adapt_transform)

    source_train_loader = DataLoader(source_train, batch_size=ctx["source_batch_size"],
                                     shuffle=True, drop_last=True,
                                     num_workers=num_workers,
                                     collate_fn=mt_datasets.mt_collate,
                                     pin_memory=True)

    target_adapt_train_loader = DataLoader(target_adapt_train, batch_size=ctx["target_batch_size"],
                                           shuffle=True, drop_last=True,
                                           num_workers=num_workers,
                                           collate_fn=mt_datasets.mt_collate,
                                           pin_memory=True)

    target_validation_loader = DataLoader(target_validation, batch_size=ctx["target_batch_size"],
                                          shuffle=False, drop_last=False,
                                          num_workers=num_workers,
                                          collate_fn=mt_datasets.mt_collate,
                                          pin_memory=True)

    model = create_model(ctx)

    if not supervised_only:
        model_ema = create_model(ctx, ema=True)
    else:
        model_ema = None

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=initial_lr,
                                 weight_decay=weight_decay)

    writer = SummaryWriter(log_dir="log_{}".format(experiment_name))

    # Training loop
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
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

        if not supervised_only:
            model_ema.train()

        composite_loss_total = 0.0
        class_loss_total = 0.0
        consistency_loss_total = 0.0

        num_steps = 0
        target_adapt_train_iter = iter(target_adapt_train_loader)

        for i, train_batch in enumerate(source_train_loader):
            # Keys: 'input', 'gt', 'input_metadata', 'gt_metadata'

            # Supervised component --------------------------------------------
            train_input, train_gt = train_batch["input"], train_batch["gt"]
            train_input = train_input.cuda()
            train_gt = train_gt.cuda(async=True)
            preds_supervised = model(train_input)
            class_loss = mt_losses.dice_loss(preds_supervised, train_gt)

            if not supervised_only:

                # Unsupervised component ------------------------------------------
                try:
                    target_adapt_batch = target_adapt_train_iter.next()
                except StopIteration:
                    target_adapt_train_iter = iter(target_adapt_train_loader)
                    target_adapt_batch = target_adapt_train_iter.next()

                target_adapt_input = target_adapt_batch["input"]
                target_adapt_input = target_adapt_input.cuda()

                # Teacher forward
                with torch.no_grad():
                    teacher_preds_unsup = model_ema(target_adapt_input)

                linked_aug_batch = \
                    linked_batch_augmentation(target_adapt_input, teacher_preds_unsup)

                adapt_input_batch = linked_aug_batch['input'][0].cuda()
                teacher_preds_unsup_aug = linked_aug_batch['input'][1].cuda()

                # Student forward
                student_preds_unsup = model(adapt_input_batch)
                consistency_loss = consistency_weight * F.mse_loss(student_preds_unsup,
                                                                   teacher_preds_unsup_aug)
            else:
                consistency_loss = torch.FloatTensor([0.]).cuda()

            composite_loss = class_loss + consistency_loss

            optimizer.zero_grad()
            composite_loss.backward()

            optimizer.step()

            composite_loss_total += composite_loss.item()
            consistency_loss_total += consistency_loss.item()
            class_loss_total += class_loss.item()

            num_steps += 1
            global_step += 1

            if not supervised_only:
                if epoch <= initial_lr_rampup:
                    update_ema_variables(model, model_ema, ctx["ema_alpha"], global_step)
                else:
                    update_ema_variables(model, model_ema, ctx["ema_alpha_late"], global_step)

        composite_loss_avg = composite_loss_total / num_steps
        class_loss_avg = class_loss_total / num_steps
        consistency_loss_avg = consistency_loss_total / num_steps

        tqdm.write("Steps p/ Epoch: {}".format(num_steps))
        tqdm.write("Consistency Weight: {:.6f}".format(consistency_weight))
        tqdm.write("Composite Loss: {:.6f}".format(composite_loss_avg))
        tqdm.write("Class Loss: {:.6f}".format(class_loss_avg))
        tqdm.write("Consistency Loss: {:.6f}".format(consistency_loss_avg))

        # Write sample images
        if ctx["write_images"] and epoch % ctx["write_images_interval"] == 0:
            try:
                plot_img = vutils.make_grid(preds_supervised,
                                            normalize=True, scale_each=True)
                writer.add_image('Train Source Prediction', plot_img, epoch)

                plot_img = vutils.make_grid(train_input,
                                            normalize=True, scale_each=True)
                writer.add_image('Train Source Input', plot_img, epoch)

                plot_img = vutils.make_grid(train_gt,
                                            normalize=True, scale_each=True)
                writer.add_image('Train Source Ground Truth', plot_img, epoch)

                # Unsupervised component viz
                if not supervised_only:
                    plot_img = vutils.make_grid(target_adapt_input,
                                                normalize=True, scale_each=True)
                    writer.add_image('Train Target Student Input', plot_img, epoch)

                    plot_img = vutils.make_grid(teacher_preds_unsup,
                                                normalize=True, scale_each=True)
                    writer.add_image('Train Target Student Preds', plot_img, epoch)

                    plot_img = vutils.make_grid(adapt_input_batch,
                                                normalize=True, scale_each=True)
                    writer.add_image('Train Target Teacher Input', plot_img, epoch)

                    plot_img = vutils.make_grid(student_preds_unsup,
                                                normalize=True, scale_each=True)
                    writer.add_image('Train Target Teacher Preds', plot_img, epoch)

                    plot_img = vutils.make_grid(student_preds_unsup,
                                                normalize=True, scale_each=True)
                    writer.add_image('Train Target Student Preds (augmented)', plot_img, epoch)
            except:
                tqdm.write("*** Error writing images ***")

        writer.add_scalars('losses', {'composite_loss': composite_loss_avg,
                                      'class_loss': class_loss_avg,
                                      'consistency_loss': consistency_loss_avg},
                           epoch)

        # Evaluation mode
        model.eval()

        if not supervised_only:
            model_ema.eval()

        metric_fns = [mt_metrics.dice_score, mt_metrics.jaccard_score, mt_metrics.hausdorff_score,
                      mt_metrics.precision_score, mt_metrics.recall_score,
                      mt_metrics.specificity_score, mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        validation(model, model_ema,
                   target_validation_loader,
                   writer, metric_fns,
                   epoch, ctx, 'target_val')

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
