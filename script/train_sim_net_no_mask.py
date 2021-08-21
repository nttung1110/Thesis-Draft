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
import sys 

from torch.utils import data
from pathlib import Path
sys.path.append("../")
from datasets.hoia_prediction_no_mask import HOIA
from model.rel_prediction_network_no_mask import SimpleNet

data_root = '/home/nttung/person-in-context/HOI-Det/HOI-A-new'
data_root_2 = '/home/nttung/person-in-context/BPA-Net/auxilary-data-no-mask'


def train_from_ckpt(path_ckpt, model, optimizer):
    ckpt = torch.load(path_ckpt)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt)
    optimizer.load_state_dict(ckpt['optimizer'])
    print("load ckpt from {}".format(path_ckpt))

    return model, optimizer, start_epoch

def train(continue_to_train):
    # set torch config
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


    # create dataloader
    
    train_data = HOIA(data_root, data_root_2, "train")
    # test_data = HOIA(data_root, data_root_2, "test")

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                                shuffle=True, num_workers=8)

    # testloader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers=8)


    model = SimpleNet(6164)
    model.to(device)

    # define loss and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # init wandb
    wandb.init(project='Thesis', entity='nttung1110')

    wandb.watch(model, log_freq=100)    

    if continue_to_train:
        path_ckpt = "./checkpoint/simplenet_first_try_10_epoch.pth"
        model, optimizer, start_epoch = train_from_ckpt(path_ckpt, model, optimizer)
    else:
        start_epoch = 1
    # train 
    num_epoch = 51
    running_loss = 0.0
    for epoch in range(start_epoch-1, num_epoch):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            # labels = labels.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            # running_loss += loss.item() 

            if (i+1) % 50 == 0:
                # print every 50 mini batches
                print("[epoch_%d, batch_%d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 50))
                wandb.log({'loss': running_loss/50})
                running_loss = 0.0
        
        if epoch % 1 == 0: # save every 1 epoch
            # Path("./checkpoint/").mkdir(parents=True, exist_ok=True)
            path_ckpt = os.path.join('/home/nttung/person-in-context/BPA-Net/checkpoint/relation_prediction_no_mask', 'sim_no_mask_net_{}.pth'.format(epoch))
            # path_ckpt = "./checkpoint/simplenet_no_mask_{}_epoch.pth".format(epoch)
            torch.save(model.state_dict(), path_ckpt)
                


if __name__ == '__main__':
    continue_to_train = False
    train(continue_to_train)