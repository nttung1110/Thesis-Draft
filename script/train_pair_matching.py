import os 
import cv2
import pdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import sys 
sys.path.append("../")

from torch.utils import data
from pathlib import Path

from datasets.hoia_matching import HOIA
from model.bpa_network import PosePairAttenNet

data_root = '/home/nttung/person-in-context/HOI-Det/HOI-A-new'
data_root_2 = '/home/nttung/person-in-context/BANAM/auxilary-data-pair-match-context-pose-attention'
data_root_pose = '/home/nttung/person-in-context/BANAM/pose-data-2019/pose_results_train_gt_2019.json'



def get_args_parser():
    parser = argparse.ArgumentParser('PairNet arguments', add_help=False)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_workers', default=8, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)

    return parser

def train_from_ckpt(path_ckpt, model, optimizer):
    ckpt = torch.load(path_ckpt)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt)
    optimizer.load_state_dict(ckpt['optimizer'])
    print("load ckpt from {}".format(path_ckpt))

    return model, optimizer, start_epoch

def train(continue_to_train, args):
    # set torch config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # create dataloader
    
    train_data = HOIA(data_root, data_root_2, data_root_pose, "train", device)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)

    # testloader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers=8)

    model = PosePairAttenNet(6152, 2048 + 4, 17)
    model.to(device)

    # define loss and criterion
    criterion = nn.BCELoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # init wandb
    wandb.init(project='Thesis', entity='nttung1110')

    wandb.watch(model, log_freq=100)    

    if continue_to_train:
        path_ckpt = "../checkpoint/pair_matching/BPA_10.pth"
        model.to(device)
        model.load_state_dict(torch.load(path_ckpt))
        start_epoch = 12
        # model, optimizer, start_epoch = train_from_ckpt(path_ckpt, model, optimizer)
    else:
        start_epoch = 1
    # train 
    num_epoch = args.epochs
    running_loss = 0.0
    for epoch in range(start_epoch-1, num_epoch):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, kv_dict, obj_query = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            kv = kv_dict["keys_and_values"].float().to(device)
            obj_query = obj_query.float().to(device)
            # construct key and value
            # labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, obj_query, kv, kv)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            # running_loss += loss.item() 

            if (i+1) % 50 == 0:
                # print every 200 mini batches
                print("[epoch_%d, batch_%d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 50))
                wandb.log({'loss_matching_2019_default_param_context_pose_attention_no_one_hot': running_loss/50})
                running_loss = 0.0
        
        if epoch % 1 == 0: # save every 1 epoch
            # Path("./checkpoint_matching_context/").mkdir(parents=True, exist_ok=True)
            save_ckpt = os.path.join('/home/nttung/person-in-context/BANAM/checkpoint/pair_matching', 'BPA_{}.pth'.format(epoch))
            # path_ckpt = "./checkpoint_matching_context_final/pairnet_context_pose_attention_default_param_no_one_hot_{}_epoch.pth".format(epoch)
            torch.save(model.state_dict(), save_ckpt)
    

if __name__ == '__main__':
    continue_to_train = True
    parser = argparse.ArgumentParser('BPA Training script', parents=[get_args_parser()])
    args = parser.parse_args()

    train(continue_to_train, args)


    #GeForce RTX 2080 Ti