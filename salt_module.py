from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics as metrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import segmentation_models_pytorch as smp
from torchvision.models.segmentation import fcn_resnet101, FCN_ResNet101_Weights, fcn_resnet50, FCN_ResNet50_Weights
from nets import SaltFormer
from nets.denseaspp import DenseASPP
from model.cmt import CMT_Ti
# from losses.label_smoothing import LabelSmoothingCELoss
from utils import create_lr_scheduler, labels_one_hot
from utils.metric import KIoU

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from nets.segformer import SegFormer,SegFormerHead

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['out']


class SaltModule(pl.LightningModule):
    def __init__(self,
                 model_name: str = 'saltformer',
                 model_hparams: Optional[dict] = None,
                 optimizer_name: str = 'Adam',
                 optimizer_hparams: Optional[dict] = None,
                 lr_scheduler_name: str = 'step',
                 lr_scheduler_hparams: Optional[dict] = None,
                 aux_params: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model_name = model_name
        self.model = self._create_model(model_name, model_hparams)

        #init model
        
        #
        self.train_acc = metrics.Accuracy(task='binary')
        self.train_iou = KIoU()
        self.valid_iou = KIoU()
        self.valid_acc = metrics.Accuracy(task='binary')
        self.test_acc = metrics.Accuracy(task='binary')
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros(
            (32, 3, 128, 128), dtype=torch.float32)

    def _create_model(self, model_name, model_hparams):
        num_cls=model_hparams['classes']
        if model_name.lower() == 'saltformer':
            return SaltFormer(**model_hparams)
        elif model_name.lower() == 'fcn':
            weights = FCN_ResNet101_Weights.DEFAULT
            return ModelWrapper(fcn_resnet101(num_classes=num_cls))
        elif model_name.lower() == 'unet':
            return smp.Unet(**model_hparams)
        elif model_name.lower() == 'deeplabv3plus':
            return smp.DeepLabV3Plus(**model_hparams)
        elif model_name.lower() == 'denseaspp':
            return DenseASPP(nclass=num_cls)
        elif model_name.lower() == 'fpn':
            # config_vit = CONFIGS_ViT_seg['ViT-L_16']
            # config_vit.n_classes=num_cls
            # config_vit.n_skip=0
            # config_vit.patches.grid = (8,8)
            # model = ViT_seg(config_vit, img_size=128, num_classes=config_vit.n_classes)
            model = smp.FPN(**model_hparams)
            # return SaltFormer('resnet34')
            return model
        elif model_name.lower() == 'segformer':
            backbone = smp.encoders.get_encoder('mit_b5')
            decode_head = SegFormerHead(
                in_channels=(64, 128, 320, 512),
                dropout_p=0.1,
                num_classes=2,
                align_corners=False,
                embed_dim=768,)
            return SegFormer(backbone,decode_head)
        else:
            assert False, f'Unknown model name "{model_name}".'

    def forward(self, imgs):
        output = self.model(imgs)
        return output

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(
                self.model.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **
                                  self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        scheduler = create_lr_scheduler(
            optimizer, self.hparams.lr_scheduler_name, **self.hparams.lr_scheduler_hparams)
        lr_scheduler = scheduler
        if self.hparams.lr_scheduler_name == 'plateau':
            lr_scheduler = {"scheduler": scheduler, "monitor": "train_loss",}
        return { 
                "optimizer":optimizer,
                "lr_scheduler": lr_scheduler
            }

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs = batch['input']
        labels = batch['mask']

        preds = self.model(imgs)
        pred_masks = preds
        if isinstance(preds, tuple):
            pred_masks = preds[0]
        
        # target= self.labels_one_hot(preds, labels)
        loss = self.calc_criterion(pred_masks, labels)

        #acc = (preds.argmax(dim=-1) == labels).float().mean()
        salt_pred_d = F.softmax(pred_masks.detach(), dim=1).argmax(dim=1).cpu()
        labels_d = labels.long().cpu()        
        self.train_acc(salt_pred_d.view(-1), labels_d.view(-1))
        self.train_iou(salt_pred_d, labels_d)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_mAP', self.train_iou, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        #log
        if batch_idx==10:
            tensorboard_logger = self.logger.experiment
            tensorboard_logger.add_image('origin_images',imgs[10],self.global_step)
            tensorboard_logger.add_image('origin_masks', labels[10],self.global_step,dataformats='HW')
            tensorboard_logger.add_image("pred_images", salt_pred_d[10],self.global_step,dataformats='HW')

        return loss  # Return tensor to call ".backward" on

    @staticmethod
    def calc_criterion(preds,targets):
        loss_fn=nn.BCEWithLogitsLoss()#
        #loss_fn = LabelSmoothingCrossEntropy()
        loss_fn1=smp.losses.LovaszLoss('multiclass') 
        loss_fn2=nn.BCEWithLogitsLoss()
        targets_onehot = labels_one_hot(preds, targets)
        #loss_fn1(preds,targets)#
        return  loss_fn(preds,targets_onehot.float())# + 0.6*loss_fn1(preds,targets) #+ 0.7*loss_fn2(logit_img.view(-1),truth_img)#

    def validation_step(self, batch, batch_idx):
        imgs = batch['input']
        labels = batch['mask']
        preds = self.model(imgs)
        # acc = (labels == preds).float().mean()
        if isinstance(preds, tuple):
            preds = preds[0]
        salt_pred_d = F.softmax(preds.detach(), dim=1).argmax(dim=1).cpu()
        labels_d = labels.long().cpu()
        self.valid_acc(salt_pred_d.view(-1), labels_d.view(-1))
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        # By default logs it per epoch (weighted average over batches)
        self.valid_iou(salt_pred_d, labels_d)
        self.log('val_mAP', self.valid_iou, on_step=False, on_epoch=True)
        # self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs = batch['input']
        labels = batch['mask']
        preds = self.model(imgs)  # .argmax(dim=-1)
        # acc = (labels == preds).float().mean()
        # # By default logs it per epoch (weighted average over batches), and returns it afterwards
        # self.log("test_acc", acc)
        if isinstance(preds, tuple):
            preds = preds[0]
        # num_classes = preds.shape[1]

        self.test_acc(F.softmax(preds, dim=1).argmax(
            dim=1).view(-1), labels.long().view(-1))
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
    