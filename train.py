import os
import argparse
import yaml
import torch
import pytorch_lightning as pl

from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import *

from datasets import *
from transforms import *
from utils import choose_device
from mt_module import MultiTaskModule
from salt_module import SaltModule
from kfold import SaltKFoldDataModule,KFoldLoop
"""

"""
def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg=parser.add_argument
    # arg('--model', default='saltddn',choices=['fcn','unet','deeplabv3+','densenet','saltddn'])
    # arg('--backbone', default='resnet101',choices=['resnet101','resnet50','senet50','mobilenet'])
    arg('--multi',dest='multi_task',action='store_true')
    arg('--no-multi',dest='multi_task',action='store_false')
    arg('--kfold',default=1,type=int)
    root_dir = Path(os.path.dirname(__file__))
    default_log_dir = root_dir/'runs' #+ '_' + socket.gethostname()
    default_data_dir = root_dir/'datasets'
    arg("-l",'--log-dir', type=Path, default=default_log_dir)
    arg('--data-dir', type=Path, default=default_data_dir)
    arg("-d",'--device',default='cuda',choices=['cuda','cpu'])
    arg("-q", '--stratify',default=None,choices=['coverage','shape'])
    arg("-b",'--batch-size', type=int, default=32)
    arg("-s",'--data-splits', type=tuple, default=(0.8,0.2))
    arg("-m","--max-epochs", type=int, default=350)
    arg("-c", "--config-file", type=Path, default='config\\default.yaml',help="Path to the config.", required=False)
    arg("-k","--checkpoint-file", type=Path, default=None)
    return parser.parse_args()

def main():
    args=get_args()
    
    if args.config_file is not None:
        with open(args.config_file) as f:
            hparams = yaml.load(f, Loader=yaml.SafeLoader)
    
    current_time = datetime.now().strftime('%Y-%m-%dT%H')
    log_dir = args.log_dir / args.config_file.stem / current_time
    device = choose_device(args.device)
    use_gpu = device.type == 'cuda'

    # model,result = train(args.model, 
    #     model_hparams={'encoder_name':args.backbone},
    #     optimizer_name=args.optimizer,
    #     optimizer_hparams={'lr': 1e-3, 'weight_decay': 1e-4})
    model_name = hparams['model_name']
    checkpoint_path= log_dir/'checkpoints'
    pl.seed_everything(42)  # To be reproducable
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    #Logger
    logger = TensorBoardLogger(log_dir)
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=checkpoint_path/model_name,  # Where to save models
        accelerator='cuda' if use_gpu else 'cpu',
        # devices=3,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode='max', monitor='val_mAP',
                filename='model-epoch{epoch:02d}-map{val_mAP:.4f}',
                auto_insert_metric_name=False
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor('epoch'),
            #EarlyStopping('val_mAP',min_delta=0.0001,patience=6)
        ],
        #strategy='ddp'
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if args.kfold ==1:
        dm = SaltDataModule(args.data_dir,splits=[.8,.2],stratify=args.stratify,batch_size=args.batch_size)
    else:
        dm = SaltKFoldDataModule(args.data_dir,stratify=args.stratify,batch_size=args.batch_size)
        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = KFoldLoop(args.kfold, export_path=checkpoint_path)
        trainer.fit_loop.connect(internal_fit_loop)
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = checkpoint_path / (model_name + '.ckpt')
    if args.checkpoint_file is not None:
        pretrained_filename = args.checkpoint_file
    
    params_space={
    "learning_rate": np.arange(0.000001,0.001,0.00005),
    "alpha":np.arange(0.90,0.99,0.01),
    "gamma":np.arange(1.5,3,0.1),
    "delta":np.arange(0.1,1,0.1),
    "mu":np.arange(0.1,1,0.1),
    }
    # for lr in params_space['learning_rate']:
        # for alpha in params_space['alpha']:
            # for gamma in params_space['gamma']:
                # for delta in params_space['delta']:
                    # for mu in params_space['mu']:
                        # hparams['optimizer_hparams']['lr'] = lr
                        # hparams['loss_hparams']['alpha'] = alpha
                        # hparams['loss_hparams']['gamma'] = gamma
                        # hparams['loss_hparams']['delta'] = delta
                        # hparams['loss_hparams']['mu'] = mu

    if pretrained_filename.is_file():
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        # Automatically loads the model with the saved hyperparameters
        if args.multi_task:
            model = MultiTaskModule.load_from_checkpoint(pretrained_filename)
        else:
            model = SaltModule.load_from_checkpoint(pretrained_filename)
    else:
        if args.multi_task:
            model = MultiTaskModule(**hparams)
        else:
            model = SaltModule(**hparams)
    trainer.fit(model, datamodule=dm)
    # print(trainer.checkpoint_callback.best_model_path)
    # model = MultiTaskModule.load_from_checkpoint(
    #     trainer.checkpoint_callback.best_model_path
    # )  # Load best checkpoint after training
    
    # Test best model on validation and test set
    test_result = trainer.test(model, datamodule=dm, verbose=False)
    print(test_result,hparams)
#    result = {'test': 0, 'val': test_result[0]['test_acc_epoch']}

if __name__ == '__main__':
    main()