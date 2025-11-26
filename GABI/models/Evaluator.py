from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer
from models.TabularModelTransformer import TabularModelTransformer


class Evaluator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.eval_datatype == 'imaging':
            self.model = ImagingModel(self.hparams)
        elif self.hparams.eval_datatype == 'multimodal':
            assert self.hparams.strategy == 'tip'
            if self.hparams.finetune_ensemble:
                self.model = TIPBackboneEnsemble(self.hparams)
            else:
                self.model = TIPBackbone(self.hparams)
        elif self.hparams.eval_datatype == 'tabular':
            #self.model = TabularModelTransformer(self.hparams)
            self.model = TabularModel(self.hparams)
        elif self.hparams.eval_datatype == 'imaging_and_tabular':
            if self.hparams.algorithm_name == 'DAFT':
                self.model = DAFT(self.hparams)
            elif self.hparams.algorithm_name in {'CONCAT', 'MAX'}:
                if self.hparams.strategy == 'tip':
                    self.model = MultimodalModelTransformer(self.hparams)
                else:
                    self.model = MultimodalModel(self.hparams)
            elif self.hparams.algorithm_name == 'MUL':
                self.model = MultimodalModelMUL(self.hparams)

        num_classes = self.hparams.num_classes
        task = 'binary' if num_classes == 2 else 'multiclass'

        # Training metrics
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_train = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_train = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.precision_train = torchmetrics.Precision(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')

        # Validation metrics
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_val = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.precision_val = torchmetrics.Precision(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')
        self.confmat_val = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

        # Test metrics
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_test = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')
        self.prc_auc_test = torchmetrics.AveragePrecision(task=task, num_classes=num_classes, average='macro')
        self.confmat_test = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_val_score = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        if len(y_hat.shape) == 1:
            y_hat = torch.unsqueeze(y_hat, 0)
        return y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)
        self.f1_train(y_hat, y)
        self.precision_train(y_hat, y)
        self.balanced_acc_train(y_hat, y)

        self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
        return loss

    def training_epoch_end(self, _) -> None:
        self.log('eval.train.f1', self.f1_train.compute(), on_epoch=True)
        self.log('eval.train.precision', self.precision_train.compute(), on_epoch=True)
        self.log('eval.train.balanced_acc', self.balanced_acc_train.compute(), on_epoch=True)
        self.log('eval.train.acc', self.acc_train.compute(), on_epoch=True)
        self.log('eval.train.auc', self.auc_train.compute(), on_epoch=True)

        self.f1_train.reset()
        self.precision_train.reset()
        self.balanced_acc_train.reset()
        self.acc_train.reset()
        self.auc_train.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        x, y = batch
        # --- 添加以下调试打印 ---
        #print(f"DEBUG: 当前批次的标签 (y): {y}")
        #print(f"DEBUG: 标签最小值: {y.min().item()}, 最大值: {y.max().item()}")
        #print(f"DEBUG: 标签的唯一值: {torch.unique(y)}")
        #print(f"DEBUG: 标签数据类型: {y.dtype}")
        # 确保你的 Evaluator 模型中存在并正确设置了 'self.num_classes'
        #print(f"DEBUG: 期望的类别数量 (n_classes): {self.hparams.num_classes}")
    # --- 调试打印结束 ---
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]
        #print('[DEBUG]')
        #print(y_hat)
        #print(self.hparams.num_classes)

        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)
        self.f1_val(y_hat, y)
        self.precision_val(y_hat, y)
        self.balanced_acc_val(y_hat, y)
        self.confmat_val(y_hat, y)

        self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    def validation_epoch_end(self, _) -> None:
        if self.trainer.sanity_checking:
            return

        self.log('eval.val.acc', self.acc_val.compute(), on_epoch=True)
        self.log('eval.val.auc', self.auc_val.compute(), on_epoch=True)
        self.log('eval.val.f1', self.f1_val.compute(), on_epoch=True)
        self.log('eval.val.precision', self.precision_val.compute(), on_epoch=True)
        # Calculate balanced accuracy for logging
        current_balanced_acc_val = self.balanced_acc_val.compute()
        self.log('eval.val.balanced_acc', current_balanced_acc_val, on_epoch=True)

        confmat = self.confmat_val.compute()
        self.log('eval.val.confmat', confmat, on_epoch=True)

        if self.hparams.num_classes >= 3:
            self.log('eval.val.TP_CN', confmat[0][0])
            self.log('eval.val.TP_MCI', confmat[1][1])
            self.log('eval.val.TP_AD', confmat[2][2])

        # --- 修改这里以正确记录 best_balanced_acc ---
        # 如果你总是希望以 balanced_acc_val 作为最佳指标，无论 hparams.target 是什么
        self.best_val_score = max(self.best_val_score, current_balanced_acc_val)
        # 如果你希望根据 hparams.target 决定，但只在特定情况下使用 balanced_acc
        # if self.hparams.target == 'balanced_accuracy': # 假设有一个这样的target设置
        #     self.best_val_score = max(self.best_val_score, current_balanced_acc_val)
        # elif self.hparams.target == 'dvm':
        #     self.best_val_score = max(self.best_val_score, self.acc_val.compute())
        # else: # 默认或其他情况使用AUC
        #     self.best_val_score = max(self.best_val_score, self.auc_val.compute())


        self.acc_val.reset()
        self.auc_val.reset()
        self.f1_val.reset()
        self.precision_val.reset()
        self.balanced_acc_val.reset()
        self.confmat_val.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        x, y = batch
        y_hat = self.forward(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)
        self.f1_test(y_hat, y)
        self.balanced_acc_test(y_hat, y)
        self.prc_auc_test(y_hat, y)
        self.confmat_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        self.log('test.acc', self.acc_test.compute(), on_epoch=True)
        self.log('test.auc', self.auc_test.compute(), on_epoch=True)
        self.log('test.f1', self.f1_test.compute(), on_epoch=True)
        self.log('test.balanced_acc', self.balanced_acc_test.compute(), on_epoch=True)
        self.log('test.prc_auc', self.prc_auc_test.compute(), on_epoch=True)

        confmat = self.confmat_test.compute()
        self.log('test.confmat', confmat, on_epoch=True)

        if self.hparams.num_classes >= 3:
            self.log('test.TP_CN', confmat[0][0])
            self.log('test.TP_MCI', confmat[1][1])
            self.log('test.TP_AD', confmat[2][2])

        self.acc_test.reset()
        self.auc_test.reset()
        self.f1_test.reset()
        self.balanced_acc_test.reset()
        self.prc_auc_test.reset()
        self.confmat_test.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr_eval,
            weight_decay=self.hparams.weight_decay_eval
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(10 / self.hparams.check_val_every_n_epoch),
            min_lr=self.hparams.lr * 0.0001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'eval.val.loss',
                "strict": False
            }
        }