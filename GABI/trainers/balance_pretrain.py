
import os
import torch
from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd
import numpy as np

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveReconstructImagingAndTabularDataset import ContrastiveReconstructImagingAndTabularDataset

from datasets.ContrastiveImageDataset import ContrastiveImageDataset
from datasets.ContrastiveTabularDataset import ContrastiveTabularDataset
from datasets.MaskTabularDataset import MaskTabularDataset

from utils.utils import grab_arg_from_checkpoint, create_logdir
from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from models.Tips.LFM2Loss_fisher_visualization import TIP3Loss


def load_datasets(hparams):
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
    hparams.transform = transform.__repr__()
    if hparams.strategy == 'tip':  
      train_dataset = ContrastiveReconstructImagingAndTabularDataset(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_train_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
        hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
      val_dataset = ContrastiveReconstructImagingAndTabularDataset(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_val_tabular, hparams.corruption_rate,  hparams.replace_random_rate, hparams.replace_special_rate, 
        hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
    else:
  
      train_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
      val_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'imaging':
    # for SSL image models
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.crop_scale_lower)
    hparams.transform = transform.__repr__()
    train_dataset = ContrastiveImageDataset(
      data=hparams.data_train_imaging, labels=hparams.labels_train, 
      transform=transform, delete_segmentation=hparams.delete_segmentation, 
      augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading,
      target=hparams.target, augmentation_speedup=hparams.augmentation_speedup)
    val_dataset = ContrastiveImageDataset(
      data=hparams.data_val_imaging, labels=hparams.labels_val, 
      transform=transform, delete_segmentation=hparams.delete_segmentation, 
      augmentation_rate=hparams.augmentation_rate, img_size=hparams.img_size, live_loading=hparams.live_loading,
      target=hparams.target, augmentation_speedup=hparams.augmentation_speedup)
  elif hparams.datatype == 'tabular':
    # for SSL tabular models
    if hparams.algorithm_name == 'SCARF':
      train_dataset = ContrastiveTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
      val_dataset = ContrastiveTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    elif hparams.algorithm_name == 'VIME':
      train_dataset = MaskTabularDataset(hparams.data_train_tabular, hparams.labels_train, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
      val_dataset = MaskTabularDataset(hparams.data_val_tabular, hparams.labels_val, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot)
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset

def pretrain(hparams, wandb_logger):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
    import torch
    import pandas as pd
    import os
    import glob
    from torch.utils.data import DataLoader

    # ---------- Prepare ----------
    pl.seed_everything(hparams.seed)
    train_dataset, val_dataset = load_datasets(hparams)

    train_loader = DataLoader(
        train_dataset, num_workers=hparams.num_workers,
        batch_size=hparams.batch_size, pin_memory=True,
        shuffle=True, persistent_workers=True)

    val_loader = DataLoader(
        val_dataset, num_workers=hparams.num_workers,
        batch_size=hparams.batch_size, pin_memory=True,
        shuffle=False, persistent_workers=True)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
    model = TIP3Loss(hparams)

    # ---------- Callbacks ----------
    callbacks = [
        ModelCheckpoint(
            filename='best_model_{epoch:02d}_val_balanced_accuracy_{supervised.val.balanced_accuracy:.4f}',
            dirpath=logdir,
            monitor='supervised.val.balanced_accuracy',
            mode='max',
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='supervised.val.balanced_accuracy',
            mode='max',
            patience=20,
            min_delta=1e-3,
            verbose=True
        )
    ]

    trainer = pl.Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=200,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        enable_progress_bar=hparams.enable_progress_bar,
        detect_anomaly=True,
        gradient_clip_algorithm="norm"
    )

    # ---------- Train ----------
    if hparams.resume_training:
        trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

    # ---------- Test ----------
    print("开始测试 ...")


    if hparams.eval_datatype == 'imaging':
        test_dataset = ImageDataset(
            hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging,
            hparams.delete_segmentation, 0,
            hparams.img_size,
            target=hparams.target, train=False, live_loading=hparams.live_loading,
            task=hparams.task, dataset_name=hparams.dataset_name,
            augmentation_speedup=hparams.augmentation_speedup
        )
    elif hparams.eval_datatype == 'tabular':
        test_dataset = TabularDataset(
            hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular,
            0, 0, train=False,
            eval_one_hot=hparams.eval_one_hot,
            field_lengths_tabular=hparams.field_lengths_tabular,
            data_base=hparams.data_base, strategy=hparams.strategy,
            missing_tabular=hparams.missing_tabular,
            missing_strategy=hparams.missing_strategy,
            missing_rate=hparams.missing_rate, target=hparams.target
        )
    elif hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']:
        test_dataset = ImagingAndTabularDataset(
            hparams.data_test_eval_imaging, hparams.delete_segmentation, 0,
            hparams.data_test_eval_tabular, hparams.field_lengths_tabular,
            hparams.eval_one_hot, hparams.labels_test_eval_imaging,
            hparams.img_size, hparams.live_loading,
            train=False, target=hparams.target, corruption_rate=0.0,
            data_base=hparams.data_base,
            missing_tabular=hparams.missing_tabular,
            missing_strategy=hparams.missing_strategy,
            missing_rate=hparams.missing_rate,
            augmentation_speedup=hparams.augmentation_speedup,
            algorithm_name=hparams.algorithm_name
        )
    else:
        raise Exception('Unsupported eval_datatype for testing')

    drop = ((len(test_dataset) % hparams.batch_size) == 1)
    test_loader = DataLoader(
        test_dataset, num_workers=hparams.num_workers,
        batch_size=hparams.batch_size, pin_memory=True,
        shuffle=False, drop_last=drop, persistent_workers=True
    )

    print(f"Number of testing batches: {len(test_loader)}")

    # ---------- Test both best & last ----------
    best_candidates = glob.glob(os.path.join(logdir, 'best_model_*.ckpt'))
    best_ckpt = sorted(best_candidates, key=os.path.getmtime)[-1] if best_candidates else None
    last_ckpt = os.path.join(logdir, 'last.ckpt')

    ckpts_to_test = {"best": best_ckpt, "last": last_ckpt}
    all_results = {}

    for name, ckpt in ckpts_to_test.items():
        if ckpt and os.path.exists(ckpt):
            print(f"\n开始测试 [{name}] 模型: {ckpt}")
            results = trainer.test(model, test_loader, ckpt_path=ckpt)
            if results:
                test_metrics = results[0]
                all_results[name] = test_metrics
                pd.DataFrame([test_metrics]).to_csv(
                    os.path.join(logdir, f'test_results_{name}.csv'), index=False)
                print(f"{name} 测试结果:", test_metrics)
            else:
                print(f"{name} 未产生测试结果。")
        else:
            print(f"未找到 {name} 模型。")
    if not all_results:
        print("未找到任何模型进行测试。")

    return all_results




