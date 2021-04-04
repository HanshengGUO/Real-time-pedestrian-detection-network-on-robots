from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset_ms import VOCDatasetMS
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.config import DefaultConfig

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=24, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
transform = Transforms()
config = DefaultConfig
train_dataset = VOCDatasetMS(root_dir='/data/Datasets/voc/VOCdevkit/VOC2012', scale_mode='multi', split='trainval',use_difficult=False,is_train=True,augment=transform, mean=config.mean,std=config.std)

model = FCOSDetector(mode="training").cuda()
model = torch.nn.DataParallel(model)
# model.load_state_dict(torch.load('/mnt/cephfs_new_wj/vc/zhangzhenghao/FCOS.Pytorch/output1/model_6.pth'))

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs
#WARMPUP_STEPS_RATIO = 0.12
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))
print("total_images : {}".format(len(train_dataset)))
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = 501

GLOBAL_STEPS = 1
LR_INIT = 1e-3
LR_END = 1e-5
###################################################################################################
# model pruning
###################################################################################################
pruning_lambda = 5e-1
bn_layers = [
    'module.fcos_body.backbone.block1.1',
    'module.fcos_body.backbone.block1.5',
    'module.fcos_body.backbone.block1.9',
    'module.fcos_body.backbone.block1.12',
    'module.fcos_body.backbone.block1.15',
    'module.fcos_body.backbone.block1.19',
    'module.fcos_body.backbone.block1.22',
    'module.fcos_body.backbone.block1.25',
    'module.fcos_body.backbone.block2.2',
    'module.fcos_body.backbone.block2.5',
    'module.fcos_body.backbone.block2.8',
    'module.fcos_body.backbone.block2.11',
    'module.fcos_body.backbone.block2.14',
    'module.fcos_body.backbone.block3.2',
    'module.fcos_body.backbone.block3.5',
    'module.fcos_body.backbone.block3.8',
    'module.fcos_body.backbone.block3.11',
    'module.fcos_body.backbone.block3.14'
]

###################################################################################################
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

model.train()

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        #lr = lr_func()
        if GLOBAL_STEPS < WARMPUP_STEPS:
           lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
           lr = LR_INIT * 0.1
           for param in optimizer.param_groups:
               param['lr'] = lr
        if GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
           lr = LR_INIT * 0.01
           for param in optimizer.param_groups:
              param['lr'] = lr
        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]

        ###################################################################################################
        # model pruning
        ###################################################################################################
        gamma_loss = []
        for k,v in model.named_parameters(): 
            if 'weight' in k: 
                if '.'.join(k.split('.')[:-1]) in bn_layers:
                    gamma_loss.append(pruning_lambda * torch.abs(v).mean())
        gamma_loss = torch.stack(gamma_loss)

        (loss.mean()+gamma_loss.mean()).backward()
        ###################################################################################################
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)
        if GLOBAL_STEPS%50 == 0:
            print(
                "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                 losses[2].mean(), cost_time, lr, loss.mean()))
        GLOBAL_STEPS += 1

    torch.save(model.state_dict(),
               "./darknet19_ms_2x_lasso_v1_lambda5e_1_from_scratch/model_{}.pth".format(epoch + 1))














