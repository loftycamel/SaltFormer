from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics as metrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from segmentation_models_pytorch.losses import FocalLoss, TverskyLoss
from losses.lovasz_losses import lovasz_loss_ignore_empty, lovasz_hinge
from losses import AutomaticWeightedLoss
from nets.saltdnn import SaltDDN
from nets.saltformer import SaltFormer
from utils import create_lr_scheduler,labels_one_hot
from utils.metric import KIoU

def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2

def symmetric_lovasz_ignore_empty(outputs, targets, truth_image):
    return (lovasz_loss_ignore_empty(outputs, targets, truth_image) +
            lovasz_loss_ignore_empty(-outputs, 1 - targets, truth_image)) / 2

def deep_supervised_criterion(logit, logit_pixel, logit_image, truth_pixel, truth_image, is_average=True):
    loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image, reduce=is_average)
    loss_pixel = 0
    for l in logit_pixel:
        loss_pixel += symmetric_lovasz_ignore_empty(l.squeeze(1), truth_pixel, truth_image)
    #salt_target = labels_one_hot(logit, truth_pixel)
    #loss = symmetric_lovasz(logit.squeeze(1), salt_target)
    return 0.05 * loss_image + 0.1 * loss_pixel #+ 1 * loss#loss_image,loss_pixel,loss #

class MultiTaskModule(pl.LightningModule):
    def __init__(self,
                 model_name: str = 'saltformer',
                 model_hparams: Optional[dict] = None,
                 optimizer_name: str = 'Adam',
                 optimizer_hparams: Optional[dict] = None,
                 loss_hparams: Optional[dict] = None,
                 lr_scheduler_name: str = 'step',
                 lr_scheduler_hparams: Optional[dict] = None,
                 aux_params: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = SaltFormer(**model_hparams)
        #
        self.awloss = AutomaticWeightedLoss(4)
        self.train_acc = metrics.Accuracy(task='binary')
        # MeanAveragePrecision(iou_type='segm',iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) # KIoU()
        self.train_iou = KIoU()
        self.valid_acc = metrics.Accuracy(task='binary')
        self.valid_iou = KIoU()
        self.test_acc = metrics.Accuracy(task='binary')
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros(
            (32, 3, 128, 128), dtype=torch.float32)
        
        self.loss_masks = nn.BCEWithLogitsLoss()
        self.loss_edge = FocalLoss('binary', alpha=self.hparams.loss_hparams['alpha'], gamma=self.hparams.loss_hparams['gamma'])
        self.loss_sdm = nn.HuberLoss(delta=self.hparams.loss_hparams['delta'])


    # def _create_model(self, model_hparams):
    #     if model_name in self.model_dict:
    #         return self.model_dict[model_name](**model_hparams)
    #     else:
    #         assert False, f'Unknown model name "{model_name}". Available models are: {str(self.model_dict.keys())}'
    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):#, optimizer_idx):
        # print(batch.shape)
        x = batch['input']
        salt_truth = batch['mask']
        edge_truth = batch['edge']
        sdm_truth = batch['sdm']
#         y = F.one_hot(y.long().squeeze_(1), 2).permute(0, 3, 1, 2)
        # print(y.shape)
        salt_pred, edge_pred, sdm_pred, logit_pixel, logit_image = self(x)

        truth_image = (salt_truth.sum(dim=(1, 2)) > 0).float()
        loss0 = deep_supervised_criterion(salt_pred, logit_pixel, logit_image, salt_truth, truth_image)
        salt_target = labels_one_hot(salt_pred, salt_truth)
        loss1 = self.hparams.loss_hparams['epsilon']*self.loss_masks(salt_pred, salt_target.float())+self.hparams.loss_hparams['mu']*symmetric_lovasz(salt_pred.squeeze(1), salt_target)

        edge_target = labels_one_hot(edge_pred, edge_truth)
        # print(edge_target.shape)
        # print(edge_truth.shape)
        loss2 = self.loss_edge(edge_pred, edge_target)
        loss3 = self.loss_sdm(sdm_pred.squeeze(1), sdm_truth.float())
        optimizer_idx=0
        if(optimizer_idx == 0):
            #loss =0.01*loss0+ 0.2*loss1 + loss2 + 0.8*loss3 #0.6*loss1 + 0.3*loss2+0.1*loss3#,loss2,loss3#
            loss = self.awloss(0.05*loss0,0.1*loss1,loss2,loss3)
            self.log('train_total_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

            #salt_pred_numpy = salt_pred.cpu().detach().numpy()

            salt_pred_d = F.softmax(salt_pred.detach(), dim=1).argmax(dim=1).cpu()
            salt_truth_d= salt_truth.long().cpu()
            self.train_acc(salt_pred_d.view(-1),salt_truth_d.view(-1))
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

            self.train_iou(salt_pred_d, salt_truth_d)
            self.log('train_mAP', self.train_iou, on_step=False, on_epoch=True)
        elif(optimizer_idx == 1):
            loss = loss2
            self.log('train_edge_loss', loss, on_step=True, on_epoch=True)
        elif(optimizer_idx == 2):
            loss = loss3
            self.log('train_sdm_loss', loss, on_step=True, on_epoch=True)
        # elif(optimizer_idx==3):
        #     loss = loss3

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['input']
        salt_truth = batch['mask']
        edge_truth = batch['edge']
        sdm_truth = batch['sdm']

#         y = F.one_hot(y.long().squeeze_(1), 2).permute(0, 3, 1, 2)
        # print(y.shape)
        salt_pred, edge_pred, sdm_pred, logit_pixel, logit_image = self(x)

        salt_pred_d = F.softmax(salt_pred.detach(), dim=1).argmax(dim=1).cpu()
        salt_truth_d= salt_truth.long().cpu()

        self.valid_acc(salt_pred_d.view(-1),salt_truth_d.view(-1))
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)
        # By default logs it per epoch (weighted average over batches)
        # self.log('val_acc', salt_pred_acc.mean(), sync_dist=True)
        self.valid_iou(salt_pred_d, salt_truth_d)
        self.log('val_mAP', self.valid_iou, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x = batch['input']
        salt_truth = batch['mask']
        edge_truth = batch['edge']
        sdm_truth = batch['sdm']
#         y = F.one_hot(y.long().squeeze_(1), 2).permute(0, 3, 1, 2)
        # print(y.shape)
        salt_pred, edge_pred, sdm_pred, logit_pixel, logit_image = self(x)

        self.test_acc(F.softmax(salt_pred, dim=1).argmax(
            dim=1).view(-1), salt_truth.long().view(-1))
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == 'Adam':
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            # optimizer = optim.AdamW([{'params': self.model.encoder.parameters()},
            #                          {'params': self.model.decoder.parameters()},
            #                          {'params': self.model.segmentation_head.parameters()}], **self.hparams.optimizer_hparams)
            optimizer = optim.AdamW(self.model.parameters(), **self.hparams.optimizer_hparams)
            # optimizer2 = optim.AdamW(
            #     self.model.classification_head.parameters(), **self.hparams.optimizer_hparams)
            # optimizer3 = optim.AdamW(
            #     self.model.regression_head.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **
                                  self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        #
        scheduler = create_lr_scheduler(
            optimizer, self.hparams.lr_scheduler_name, **self.hparams.lr_scheduler_hparams)
        lr_scheduler = scheduler
        if self.hparams.lr_scheduler_name == 'plateau':
            lr_scheduler = {"scheduler": scheduler, "monitor": "train_total_loss",}
        # scheduler = optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[20, 40, 80], gamma=0.1)
        #
        # scheduler2 = optim.lr_scheduler.MultiStepLR(
        #     optimizer2, milestones=[20, 40, 80], gamma=0.1)
        # scheduler3 = optim.lr_scheduler.MultiStepLR(
        #     optimizer3, milestones=[20, 40, 80], gamma=0.1)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler} #[optimizer, optimizer2, optimizer3], [scheduler, scheduler2, scheduler3]
