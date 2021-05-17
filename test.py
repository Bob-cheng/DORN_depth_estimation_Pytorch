from progress_tracking import AverageMeter, Result, ImageBuilder
import torch
from discritization import SID


def test_performace(model, val_loader, logger, dataset, LOG_IMAGES, global_steps):
    sid = SID(dataset)
    model.eval()
    average_meter_eval = AverageMeter()
    image_builder_eval = ImageBuilder()
    for i, sample in enumerate(val_loader):
        input, target = sample[0].cuda(), sample[1].cuda()
        if dataset == 'kitti':
            target_dense = sample[2].cuda()
        
        with torch.no_grad():
            pred_labels, _ = model(input)

        # track performance scores
        pred = sid.labels2depth(pred_labels)
        result = Result()
        if dataset == 'nyu':
            result.evaluate(pred.detach(), target.detach())
        elif dataset == 'kitti':
            result.evaluate(pred.detach(), target.detach())
        average_meter_eval.update(result, input.size(0))
        if i <= LOG_IMAGES:
            if dataset == 'nyu':
                image_builder_eval.add_row(input[0,:,:,:], target[0,:,:], pred[0,:,:])
            elif dataset == 'kitti':
                image_builder_eval.add_row(input[0,:,:,:], target_dense[0,:,:], pred[0,:,:])
    
    # log performance scores with tensorboard
    average_meter_eval.log(logger, global_steps, 'Test')
    logger.add_image('Test/Image', image_builder_eval.get_image(), global_steps)