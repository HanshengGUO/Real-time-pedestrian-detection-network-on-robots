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
import os
from model.config import DefaultConfig

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12, help="number of epochs")
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
output_dir = 'finetune_pruned'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

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

GLOBAL_STEPS = 1
LR_INIT = 1e-3
LR_END = 1e-5
lr = LR_INIT
optimizer = torch.optim.SGD(model.parameters(),lr =LR_INIT,momentum=0.9,weight_decay=0.0001)

model.train()

for epoch in range(EPOCHS):
    for epoch_step, data in enumerate(train_loader):

        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        if GLOBAL_STEPS == int(TOTAL_STEPS*0.6):
            lr = LR_INIT * 0.1
            for param in optimizer.param_groups:
                param['lr'] = lr
        if GLOBAL_STEPS == int(TOTAL_STEPS*0.8):
            lr = LR_INIT * 0.01
            for param in optimizer.param_groups:
                param['lr'] = lr
        start_time = time.time()
        
        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
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
               os.path.join(output_dir, "model_slim_{}.pth".format(epoch + 1)))














