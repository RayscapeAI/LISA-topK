import os
import numpy as np
import json
from tqdm import tqdm, trange

from utils.config import *
from utils.utils import *
from utils.metrics import *
from utils.model import get_densenet121, get_style_densenet121
from utils.dataset import XRayClassificationDataset, get_augmentations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


train_batch_ct = 0

def train_one_epoch(epoch, loader, model, optimizer, criterion, train_writer):
    model.train()
    global train_batch_ct
    train_loss, train_pred_all, train_target_all, domain_all, index_all = [], [], [], [], []

    for train_batch, data in tqdm(enumerate(loader), total=len(loader), desc='Train_' + str(epoch)):
        if config['strategy'] == "StyleDensenet":
            x, x_no_style, target, domain, idx = data
            x_no_style = x_no_style.to(device)
        else:
            x, target, domain, idx = data

        # Select samples and apply mixup strategy
        if config['strategy'] == 'LISA':
            x, target = loader.dataset.LISA(x, target, domain, idx)
        elif config['strategy'] == 'CrossDomain':
            x, target = loader.dataset.CrossDomain(x, target, domain, idx)

        x, target = x.to(device), target.to(device)
        optimizer.zero_grad()

        if config['strategy'] == "StyleDensenet":
            results = model(x, target, x_no_style)
            output = results['logits']
            loss = results['loss']
        else:
            output = model(x)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        train_writer.add_scalar(f'loss_batch', loss.item(), train_batch_ct)
        train_writer.add_scalar(f'lr', get_lr(optimizer), train_batch_ct)
        train_loss.append(loss.item())
        domain_all.append(domain)
        train_batch_ct += 1

        output = get_predictions(output)
        train_pred_all.append(output.detach().cpu().numpy())
        train_target_all.append(target.detach().cpu().numpy())
        index_all.append(idx)

        del loss, x, target, output

    train_pred_all = np.concatenate(train_pred_all)
    train_target_all = np.concatenate(train_target_all)
    domain_all = np.concatenate(domain_all)
    index_all = np.concatenate(index_all)

    return train_pred_all, train_target_all, domain_all, np.mean(train_loss), index_all


@torch.no_grad()
def valid_one_epoch(epoch, valid_loader, model, criterion):
    model.eval()
    valid_loss, valid_pred_all, valid_target_all, domain_all, index_all = [], [], [], [], []

    for valid_batch, (x, target, domain, idx) in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_' + str(epoch)):
        x, target = x.to(device), target.to(device)

        if config['strategy'] == "StyleDensenet":
            results = model(x, target)
            output = results['logits']
            loss = results['loss']
        else:
            output = model(x)
            loss = criterion(output, target)

        output = get_predictions(output)
        valid_pred_all.append(output.cpu().numpy())
        valid_target_all.append(target.cpu().numpy())
        domain_all.append(domain)
        valid_loss.append(loss.item())
        index_all.append(idx)

    valid_pred_all = np.concatenate(valid_pred_all)
    valid_target_all = np.concatenate(valid_target_all)
    domain_all = np.concatenate(domain_all)
    index_all = np.concatenate(index_all)

    return valid_pred_all, valid_target_all, domain_all, np.mean(valid_loss), index_all


def main(results_config, config):
    create_files(results_config.values())
    save_logs(config, results_config['logs_path'])
    train_writer = SummaryWriter(results_config['tensorboard_path'] + '/train')
    valid_writer = SummaryWriter(results_config['tensorboard_path'] + '/valid')
    test_writer  = SummaryWriter(results_config['tensorboard_path'] + '/test')

    if config['strategy'] == "StyleDensenet":
        model = get_style_densenet121(checkpoint_path=config['checkpoint_path'])
    else:
        model = get_densenet121(checkpoint_path=config['checkpoint_path'])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'], weight_decay=config['wc'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-3, patience=config['patience'], factor=0.1)
    criterion = nn.BCEWithLogitsLoss()

    valid_augmentations = get_augmentations(config['image_size'], 'valid')
    train_augmentations = get_augmentations(config['image_size'], 'train') if config['augmentation'] else valid_augmentations

    train_dataset = XRayClassificationDataset(config['train_data_path'], config['metadata_path'], 'train', train_augmentations, config)
    valid_dataset = XRayClassificationDataset(config['valid_data_path'], config['metadata_path'], 'valid', valid_augmentations, config)
    test_dataset =  XRayClassificationDataset(config['valid_data_path'], config['metadata_path'], 'test',  valid_augmentations, config)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    train_batch_ct = 0
    best_score = 0

    if config['only_test'] == False:
        for epoch in trange(config['num_epochs']):
            # Train
            train_pred_all, train_target_all, domain_all, train_loss, train_index_all = train_one_epoch(epoch, train_loader, model, optimizer, criterion, train_writer)
            
            if config['strategy'] is None:
                train_stats = stats_handler(epoch, train_pred_all, train_target_all, domain_all, train_loss, train_writer)
                print(f"Train loss {train_loss} | F1 {train_stats['All']['average/F1']}")
            else:
                train_stats = {'loss': train_loss}
                train_writer.add_scalar('loss', train_loss, epoch)
                print(f"Train loss {train_loss}")

            # Valid
            valid_pred_all, valid_target_all, domain_all, valid_loss, valid_index_all = valid_one_epoch(epoch, valid_loader, model, criterion)
            valid_stats = stats_handler(epoch, valid_pred_all, valid_target_all, domain_all, valid_loss, valid_writer)

            print(f"Valid loss {valid_loss} | F1 {valid_stats['All']['average/F1']}")

            scheduler.step(valid_loss)

            # Save best model
            if valid_stats['All']['average/F1'] > best_score:
                all_stats = {
                    'train': train_stats,
                    'valid': valid_stats
                }

                np.save(results_config['sample_path'] + "/train_pred.npy", train_pred_all)
                np.save(results_config['sample_path'] + "/train_index.npy", train_index_all)
                np.save(results_config['sample_path'] + "/valid_pred.npy", valid_pred_all)
                np.save(results_config['sample_path'] + "/valid_index.npy", valid_index_all)

                save_model(epoch, 'best_f1_model', model, optimizer, results_config['checkpoints_path'], all_stats)
                best_score = valid_stats['All']['average/F1']

    # Test model with best validation mean F1 score
    best_state = torch.load(results_config['checkpoints_path'] + '/best_f1_model.ckpt')
    stats = best_state['stats']
    model.load_state_dict(best_state['state_dict'])

    valid_thresholds = {
        class_name : stats['valid']['All'][class_name + '/F1/threshold']
        for class_name in classes
    }

    test_pred_all, test_target_all, domain_all, loss, test_index_all = valid_one_epoch(0, test_loader, model, criterion)
    test_stats = stats_handler(0, test_pred_all, test_target_all, domain_all, loss, test_writer, thresholds=valid_thresholds)
    stats['test'] = test_stats

    print(f"Test loss {loss} | F1 val th {test_stats['All']['average/F1_val_th']} | F1 {test_stats['All']['average/F1']}")

    # Save statistics and predictions
    with open(results_config['logs_path'] + '/stats.json', 'w') as f:
        json.dump(stats, f)

    np.save(results_config['sample_path'] + "/test_pred.npy", test_pred_all)
    np.save(results_config['sample_path'] + "/test_index.npy", test_index_all)


if __name__ == '__main__':
    exp_root = config['metadata_path'].split('/')[-1].strip('.csv')

    # Define sweep
    for split in ['split_OOD', 'split_ID']:
        for SEED in [1337, 666013, int(1e9+7), 73939133, 13]:
            for strategy in [None, 'LISA']:
                config['split_type'] = split
                config['SEED'] = SEED
                config['strategy'] = strategy

                run_name = '{}_strategy-{}_lr{}-reduce-p5_BCE_aug-True_wc1e-4_{}epochs_seed{}'.format(
                    config['split_type'].replace('split_', ''), config['strategy'], config['lr'], config['num_epochs'], config['SEED']
                )
                box_print(run_name)

                set_seed(config['SEED'])
                train_batch_ct = 0
                print(ROOT)
                results_config = get_results_config(ROOT + 'domain_generalization/results/{}/{}'.format(exp_root, run_name))

                if config['split_type'] == 'split_ID' and config['strategy'] is not None:
                    continue

                main(results_config, config)
