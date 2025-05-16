import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# @reference : https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

# TODO Step 0 : DDP 모듈을 import 한다.
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Parse input arguments (Baseline)
parser = argparse.ArgumentParser(description='Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs that meet target before stopping')

# TODO Step 1 : DDP를 위해 input arguments 새롭게 추가한다.
# number of nodes (num_nodes, type = int, default = 1),
# ID for the current node (node_id, type = int, default = 0)
# number of GPUs in each node (num_gpus, type = int, default = 1)

# ? python fashion_mnist.py --node-id 0 --num-gpus 2 --num-nodes 2

# [parament name]       [variable name]
# 1. node-id        >>>     node_id
# 2. num-gpus       >>>     num_gpus
# 3. num-nodes      >>>     num_nodes
parser.add_argument('--node-id', type=int, default=0,
                    help='node id(=host id')
parser.add_argument('--num-gpus', type=int, default=2,
                    help='Number of gpu in node(=host')
parser.add_argument('--num-nodes', type=int, default=1,
                    help='Number of node(host')

args = parser.parse_args()


# TODO Step 2: Compute world size (WORLD_SIZE) using num_gpus and num_nodes
# and specify the IP address/port number for the node associated with
# the main process (global rank = 0):
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9956'

WORLD_SIZE = args.num_nodes * args.num_gpus

# DDP setup for process
def setupDDP(global_rank, world_size, port="1238"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # To properly initialize and synchornize each process,
    # invoke dist.init_process_group with the approrpriate parameters:
    # backend='nccl', world_size=WORLD_SIZE, rank=global_rank
    dist.init_process_group(backend='nccl', world_size=world_size, rank=global_rank)

# DDP cleanup for process
def cleanupDDP():
    dist.destroy_process_group()


# WIDE Model
# Standard convolution block followed by batch normalization for WIDE resnet model
# cbr = convolution + batch normalization + relu
# input size = output size
class cbr_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cbr_block, self).__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=(1,1),
                      padding="same", bias=False, groups=in_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.cbr(x)
        return out

# Basic residual block
# @param {Boolean} scale_input :
# True(1) : 1x1 convolution 을 이용한 input channel 이 output channel 과 같도록 scaling
class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, scale_input):
        super(conv_block, self).__init__()

        # scaling 은 input channels = output channels 되도록 1x1 conv.
        self.scale_input = scale_input
        if scale_input:
            self.scale = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                                   stride=(1,1), padding="same")

        # cbr_block 은 input size = output size 로 크기가 변하지 않음.
        self.layer1 = cbr_block(in_channel, out_channel)
        self.dropout = nn.Dropout2d(p=0.01)
        self.layer2 = cbr_block(out_channel, out_channel)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)

        if self.scale_input:
            residual = self.scale(residual)

        out = out + residual
        return out

# WIDE ResNet
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        # input channels ~ output channels
        nChannels = [1, 16, 160, 320, 640]

        self.input_block = cbr_block(nChannels[0], nChannels[1])

        # Module with alternating components employing input scaling
        self.block1 = conv_block(nChannels[1], nChannels[2], True)
        self.block2 = conv_block(nChannels[2], nChannels[2], False)
        self.pool1 = nn.MaxPool2d(2)

        self.block3 = conv_block(nChannels[2], nChannels[3], True)
        self.block4 = conv_block(nChannels[3], nChannels[3], False)
        self.pool2 = nn.MaxPool2d(2)

        self.block5 = conv_block(nChannels[3], nChannels[4], True)
        self.block6 = conv_block(nChannels[4], nChannels[4], False)

        # Global average pooling
        self.pool = nn.AvgPool2d(7)

        # Feature flattening followed by linear layer
        self.flat = nn.Flatten()

        # nn.Linear(in_features, out_features, bias=True)
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        # input conv block
        out = self.input_block(x)

        # modules to scaling input
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)

        # global average pool
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)

        return out


# training model with optimizer and loss function
def train(model, optimizer, train_loader, loss_fn, device):
    # train mode
    model.train()

    # train loader 에서 images, labels 를 얻음
    for images, labels in train_loader:
        # transfering images and labels to GPU if available
        labels = labels.to(device)
        images = images.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # setting all parameter gradients to zero to avoid gradient accumulation
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update model parameters
        optimizer.step()


# testing (validation)
def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    total_loss = 0

    # eval mode
    model.eval()
    # model parameter 업데이트와 gradient 계산을  멈춤 상태에서 평가
    with torch.no_grad():
        # test loader 에서 images, labels 를 얻음
        for images, labels in test_loader:
            # transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # extracting predicted label
            # 1. torch.max(input, dim, ...) : input 의 dim 을 기준으로 최대값과 그 최대값의 인텍스를 반환하는 함수
            # 2. outputs tensor 크기가 (batch_size, num_classes) 라면 outputs[i] 는 i 번째 샘플에 대한 클래스별 점수
            # 3. torch.max(outputs, 1) 는 outputs 의 두 번째 차원(클래스 차원)을 기준으로 최대값과 그 최대값의 인덱스를 반환
            # 4. torch.max(outputs, 1)[1] 는 해당 최대값의 인덱스 즉, 예측된 클래스를 의미함.
            preds = torch.max(outputs, 1)[1]

            # computing total correct labels and total loss
            total_labels += len(labels)
            correct_labels += torch.sum(preds == labels)
            total_loss += loss

    # computing validation loss and validation accuracy
    v_acc = correct_labels / total_labels
    v_loss = total_loss / total_labels

    return v_acc, v_loss



# TODO Step 3:
# a new 'worker' function that accepts two inputs with no return value:
# (1) the local rank (local_rank) of the process
# (2) the parsed input arguments (args)

# process 별 학습을 수행하는 함수
# The following is the signature for the worker function: worker(local_rank, args)
def worker(local_rank, args):
    # TODO Step 4: Compute the global rank (global_rank) of the spawned process as:
    # =node_id*num_gpus + local_rank.
    global_rank = args.node_id * args.num_gpus + local_rank

    print('worker local_rank: ' + str(local_rank) + ', global rank: ' + str(global_rank) )

    # TODO : DDP setup
    # to properly initialize and synchornize each process,
    # invoke dist.init_process_group with the approrpriate parameters:
    # backend='nccl', world_size=WORLD_SIZE, rank=global_rank
    setupDDP(global_rank, WORLD_SIZE)

    # TODO : Data Downloading
    # TODO Step 5: initialize a download flag (download) that is true
    # only if local_rank == 0. This download flag can be used as an
    # input argument in torchvision.datasets.FashionMNIST.
    # Download the training and validation sets for only local_rank == 0.
    # Call dist.barrier() to have all processes in a given node wait
    # till data download is complete. Following this, for all other
    # processes, torchvision.datasets.FashionMNIST can be called with
    # the download flag as false.
    download = True if local_rank == 0 else False

    if local_rank == 0:
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=download,
                                                      transform= transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=download,
                                                     transform=transforms.Compose([transforms.ToTensor()]))
    # TODO: for synchronizing
    dist.barrier()

    if local_rank != 0:
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=download,
                                                      transform= transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=download,
                                                     transform=transforms.Compose([transforms.ToTensor()]))


    # TODO : Data Loading (with torch.utils.data.distributed.DistributedSampler)
    #  DistributedSampler : 여러 GPU or 머신에 걸쳐 학습 데이터셋을 효율적으로 분배하는데 사용

    # TODO Step 6: generate two samplers (one for the training
    # dataset (train_sampler) and the other for the testing
    # dataset (test_sampler) with  torch.utils.data.distributed.DistributedSampler.
    # Inputs to this function include:
    # (1) the datasets (either train_loader_subset or test_loader_subset)
    # (2) number of replicas (num_replicas), which is the world size (WORLD_SIZE)
    # (3) the global rank (global_rank).
    # Pass the appropriate sampler as a parameter (e.g., sampler = train_sampler)
    # to the training and testing DataLoader

    # training data sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=WORLD_SIZE, rank=global_rank)

    # testing data sampler
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=WORLD_SIZE, rank=global_rank)

    # training data loader
    # 마지막 배치가 배치 크기보다 작은 경우 그 배치를 버리도록 설정
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)

    # testing data loader
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, sampler=test_sampler, drop_last=True)


    # TODO : create model for fashion mnist
    num_classes = 10

    # TODO Step 7: Modify the torch.device call from "cuda:0" to "cuda:<enter local rank here>"
    # to pin the process to its assigned GPU.
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")

    # After the model is moved to the assigned GPU, wrap the model with
    # nn.parallel.DistributedDataParallel, which requires the local rank (local_rank)
    # to be specificed as the 'device_ids' parameter: device_ids=[local_rank]
    # TODO Optional: before moving the model to the GPU, convert standard
    #  batchnorm layers to SyncBatchNorm layers using
    #  torch.nn.SyncBatchNorm.convert_sync_batchnorm.

    model = WideResNet(num_classes).to(device)
    # convert standard batchnorm >>> SyncBatchNorm
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 모델을 여러 GPU에 분산하여 학습할 수 있도록 해줌.
    model = DDP(model, device_ids=[local_rank])

    # TODO : loss function
    loss_fn = nn.CrossEntropyLoss()

    # TODO : optimization (Stochastic gradient descent)
    # 확률적 경사 하강법 (Stochastic Gradient Descent, SGD)
    # 기울기 계산: 확률적 경사 하강법(SGD)은 훈련 데이터셋의 각 샘플을 사용하여 기울기를 계산.
    # 즉, 각 업데이트마다 하나의 샘플 (혹은 작은 배치)을 사용하여 기울기를 계산하고 파라미터를 업데이트.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)

    # TODO : training using batch samples
    total_time = 0

    val_acc = []
    for epoch in range(args.epochs):
        # get the current time
        t0 = time.time()

        # TODO Step 6.5: update the random seed of the DistributedSampler to change
        #  the shuffle ordering for each epoch.
        # It is necessary to do this for the train_sampler, but irrelevant for the test_sampler.
        # The random seed can be altered with the set_epoch method (which accepts the epoch number
        # as an input) of the DistributedSampler.
        train_sampler.set_epoch(epoch)

        # TODO : training
        train(model, optimizer, train_loader, loss_fn, device)

        # TODO Step 8: at the end of every training epoch, synchronize (using dist.barrier())
        #  all processes to compute the slowest epoch time.
        # To compute the number of images processed per second, convert images_per_sec
        # into a tensor on the GPU, and then call torch.distributed.reduce on images_per_sec
        # with global rank 0 as the destination process. The reduce operation computes the
        # sum of images_per_sec across all GPUs and stores the sum in images_per_sec in the
        # master process (global rank 0).
        # Once this computation is done, enable the metrics print statement for only the master process.
        dist.barrier()

        # TODO : compute training time per epoch
        epoch_time = time.time() - t0
        total_time += epoch_time

        # compute the processing time per image
        images_per_sec = torch.tensor(len(train_loader) * args.batch_size / epoch_time).to(device)

        # TODO : testing per epoch
        v_acc, v_loss = test(model, test_loader, loss_fn, device)

        # TODO Step 9: average validation accuracy and loss across all GPUs
        # using torch.distributed.all_reduce. To perform an average operation,
        # provide 'dist.ReduceOp.AVG' as the input for the op parameter in
        # torch.distributed.all_reduce.

        # 모든 프로세스의 accuracy 평균을 계산한다.
        torch.distributed.all_reduce(v_acc, op=dist.ReduceOp.AVG)
        # 모든 프로세스의 loss 평균을 계산한다.
        torch.distributed.all_reduce(v_loss, op=dist.ReduceOp.AVG)

        val_acc.append(v_acc)

        # TODO : result print on global rank == 0
        if global_rank == 0:
            print("Epoch = {:2d}: Cumulative Time = {:5.3f}, Epoch Time = {:5.3f}, Images/sec = {}, "
                  "Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}"
                  .format(epoch+1, total_time, epoch_time, images_per_sec, v_loss, val_acc[-1]))

        # TODO : early stopping
        if len(val_acc) >= args.patience and all(acc >= args.target_accuracy for acc in val_acc[-args.patience:]):
            print('Early stopping after epoch {}'.format(epoch + 1))
            break


    # TODO : DDP cleanup
    cleanupDDP()


    # TODO Step 10: Within __name__ == '__main__', launch each process (total number of
    # processes is equivalent to the number of available GPUs per node) with
    # torch.multiprocessing.spawn(). Input parameters include the worker function,
    # the number of GPUs per node (nprocs), and all the parsed arguments.

if __name__ == '__main__':
    print('node id: ' + str(args.node_id) + ', num of gpus: ' + str(args.num_gpus))
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))
