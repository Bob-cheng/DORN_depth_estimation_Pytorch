import torch
import argparse
from model import DORN, read_model, save_model
from data import get_dataloaders
from loss import OrdinalLoss
from lr_decay import PolynomialLRDecay
from discritization import SID
from progress_tracking import AverageMeter, Result, ImageBuilder
from tensorboardX import SummaryWriter
from datetime import datetime
import os, socket

LOG_IMAGES = 3 # number of images per epoch to log with tensorboard

# Parse arguments
parser = argparse.ArgumentParser(description='DORN depth estimation in PyTorch')
parser.add_argument('--dataset', default='nyu', type=str, help='dataset: kitti or nyu (default: nyu)')
parser.add_argument('--data-path', default='./nyu_official', type=str, help='path to the dataset')
parser.add_argument("--pretrained", action='store_true', help="use a pretrained feature extractor")
parser.add_argument('--epochs', default=200, type=int, help='n of epochs (default: 200)')
parser.add_argument('--bs', default=3, type=int, help='[train] batch size(default: 3)')
parser.add_argument('--bs-test', default=3, type=int, help='[test] batch size (default: 3)')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use (default: 0)')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
train_loader, val_loader = get_dataloaders(args.dataset, args.data_path, args.bs, args.bs_test)
model = DORN(dataset=args.dataset, pretrained=args.pretrained).cuda()
train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}, {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
lr_decay = PolynomialLRDecay(optimizer, args.epochs, args.lr * 1e-2)
criterion = OrdinalLoss()
sid = SID(args.dataset)

# Create logger
log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname() + '_' + args.dataset)
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)
# Log arguments
with open(os.path.join(log_dir, "args.txt"), "a") as f:
    print(args, file=f)

for epoch in range(args.epochs):
    # log learning rate
    for i, param_group in enumerate(optimizer.param_groups):
        logger.add_scalar('Lr/lr_' + str(i), float(param_group['lr']), epoch)
        
    print('Epoch', epoch, 'train in progress...')
    model.train()
    average_meter = AverageMeter()
    image_builder = ImageBuilder()
    for i, sample in enumerate(train_loader):
        input, target = sample[0].cuda(), sample[1].cuda()
        if args.dataset == 'kitti':
            target_dense = sample[2].cuda()
        
        pred_labels, pred_softmax = model(input)
        if args.dataset == 'nyu':
            target_labels = sid.depth2labels(target)  # get ground truth ordinal labels using SID
        elif args.dataset == 'kitti':
            target_labels = sid.depth2labels(target_dense)  # get ground truth ordinal labels using SID
        loss = criterion(pred_softmax, target_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track performance scores
        depth = sid.labels2depth(pred_labels)
        result = Result()
        if args.dataset == 'nyu':
            result.evaluate(depth.detach(), target.detach())
        elif args.dataset == 'kitti':
            result.evaluate(depth.detach(), target.detach())
        average_meter.update(result, input.size(0))
        if i <= LOG_IMAGES:
            if args.dataset == 'nyu':
                image_builder.add_row(input[0,:,:,:], target[0,:,:], depth[0,:,:])
            elif args.dataset == 'kitti':
                image_builder.add_row(input[0,:,:,:], target_dense[0,:,:], depth[0,:,:])
        
    # log performance scores with tensorboard
    average_meter.log(logger, epoch, 'Train')
    if LOG_IMAGES:
        logger.add_image('Train/Image', image_builder.get_image(), epoch)

    lr_decay.step()
        
    print('Epoch', epoch, 'test in progress...')
    model.eval()
    average_meter = AverageMeter()
    image_builder = ImageBuilder()
    for i, sample in enumerate(val_loader):
        input, target = sample[0].cuda(), sample[1].cuda()
        if args.dataset == 'kitti':
            target_dense = sample[2].cuda()
        
        with torch.no_grad():
            pred_labels, _ = model(input)

        # track performance scores
        pred = sid.labels2depth(pred_labels)
        result = Result()
        if args.dataset == 'nyu':
            result.evaluate(pred.detach(), target.detach())
        elif args.dataset == 'kitti':
            result.evaluate(pred.detach(), target.detach())
        average_meter.update(result, input.size(0))
        if i <= LOG_IMAGES:
            if args.dataset == 'nyu':
                image_builder.add_row(input[0,:,:,:], target[0,:,:], pred[0,:,:])
            elif args.dataset == 'kitti':
                image_builder.add_row(input[0,:,:,:], target_dense[0,:,:], pred[0,:,:])
    
    # log performance scores with tensorboard
    average_meter.log(logger, epoch, 'Test')
    if LOG_IMAGES:
        logger.add_image('Test/Image', image_builder.get_image(), epoch)
    
    print()
    
logger.close()

model_path = save_model(model, args.dataset, args.pretrained)
# loaed_model = read_model(model_path, args.dataset, args.pretrained)