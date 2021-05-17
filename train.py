import torch
import argparse
from model import DORN, read_checkpoint, read_model, save_model
from data import get_dataloaders
from loss import OrdinalLoss
from lr_decay import PolynomialLRDecay
from discritization import SID
from progress_tracking import AverageMeter, Result, ImageBuilder
from tensorboardX import SummaryWriter
from datetime import datetime
import os, socket
from test import test_performace

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
parser.add_argument('--log-interval', default=400, type=int, help='data logging interval (default: 400)')
parser.add_argument('--check-point', default='', type=str, help='the checkpoint file path, start from checkpoint (default: "")')
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
train_loader, val_loader = get_dataloaders(args.dataset, args.data_path, args.bs, args.bs_test)
model = DORN(dataset=args.dataset, pretrained=args.pretrained)

# parallel
print('GPU count: {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    # if GPU number > 1, then use multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

model.to(device)

if isinstance(model, torch.nn.DataParallel):
    train_params = [{'params': model.module.get_1x_lr_params(), 'lr': args.lr}, {'params': model.module.get_10x_lr_params(), 'lr': args.lr * 10}]
else:
    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}, {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

# load checkpoint
if args.check_point != '' and os.path.exists(args.check_point):
    checkpoint = read_checkpoint(args.check_point)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('checkpoint loaded: ', args.check_point)

lr_decay = PolynomialLRDecay(optimizer, args.epochs * 20000 // args.bs, args.lr * 1e-2)
criterion = OrdinalLoss()
sid = SID(args.dataset)

# Create logger
log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname() + '_' + args.dataset)
os.makedirs(log_dir)
logger = SummaryWriter(log_dir)
# Log arguments
with open(os.path.join(log_dir, "args.txt"), "a") as f:
    print(args, file=f)

global_steps = 0
LOG_INTERVAL = args.log_interval
average_meter = AverageMeter()
image_builder = ImageBuilder()

for epoch in range(args.epochs):        
    print('Epoch', epoch, 'train in progress...')
    model.train()
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
        lr_decay.step()

        # track performance scores
        depth = sid.labels2depth(pred_labels)
        result = Result()
        if args.dataset == 'nyu':
            result.evaluate(depth.detach(), target.detach())
        elif args.dataset == 'kitti':
            result.evaluate(depth.detach(), target.detach())
        average_meter.update(result, input.size(0))
        if global_steps % LOG_INTERVAL <= LOG_IMAGES:
            if args.dataset == 'nyu':
                image_builder.add_row(input[0,:,:,:], target[0,:,:], depth[0,:,:])
            elif args.dataset == 'kitti':
                image_builder.add_row(input[0,:,:,:], target_dense[0,:,:], depth[0,:,:])

        # log with tensorboard
        if global_steps % LOG_INTERVAL == LOG_IMAGES:
            # log learning rate
            for idx, param_group in enumerate(optimizer.param_groups):
                logger.add_scalar('Lr/lr_' + str(idx), float(param_group['lr']), global_steps)
            # log performance scores with tensorboard
            average_meter.log(logger, global_steps, 'Train')
            if LOG_IMAGES:
                logger.add_image('Train/Image', image_builder.get_image(), global_steps)
            
            # test performance
            model.eval()
            test_performace(model, val_loader, logger, args.dataset, LOG_IMAGES, global_steps)

            # reset
            model.train()
            average_meter = AverageMeter()
            image_builder = ImageBuilder()
            print(f'global steps: {global_steps}')
        global_steps += 1
    
    # lr_decay.step()    
    print('Epoch', epoch, 'test in progress...')
    model.eval()
    test_performace(model, val_loader, logger, args.dataset, LOG_IMAGES, global_steps)
   
    # save model after each epoch
    model_path = save_model(model, optimizer, args.dataset, args.pretrained)
    print()
    
logger.close()

# model_path = save_model(model, args.dataset, args.pretrained)
# loaed_model = read_model(model_path, args.dataset, args.pretrained)