import copy
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from typing import Tuple, Any # 导入 Any

# 导入您的模型文件
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstram_imbalance import TIPBackbone # 导入修改后的TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer


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
                self.model = TIPBackbone(self.hparams) # 这里会实例化 TIPBackbone
        elif self.hparams.eval_datatype == 'tabular':
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

        # Metrics 初始化
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_train = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_train = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.precision_train = torchmetrics.Precision(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')

        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_val = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.precision_val = torchmetrics.Precision(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')
        self.confmat_val = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=num_classes, average='macro')
        self.f1_test = torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro')
        self.balanced_acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro')
        self.prc_auc_test = torchmetrics.AveragePrecision(task=task, num_classes=num_classes, average='macro')
        self.confmat_test = torchmetrics.ConfusionMatrix(task=task, num_classes=num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.best_val_score = 0

        # ====== 新增：模态不平衡处理相关变量 ======
        # 从 hparams 中获取 missing_tabular 属性，确保它在 hparams 中定义
        self.missing_tabular = getattr(hparams, 'missing_tabular', False) # 假设默认值为 False

        if self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']: # 只在多模态场景下启用
            self.imaging_trace_list = []
            self.tabular_trace_list = []
            self.softmax = nn.Softmax(dim=1)
            self.tanh = nn.Tanh()
            # 用于FIM累积（每个epoch清零）
            self._fim_imaging_epoch_sum = None # 会在第一个batch时初始化
            self._fim_tabular_epoch_sum = None # 会在第一个batch时初始化
        # ==========================================


    def forward(self, x_tuple: Tuple[torch.Tensor, torch.Tensor, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_tuple 应该包含 (image_data, tabular_data, missing_mask_or_None)
        # TIPBackbone 接收 x=(x_i, x_t, missing_mask)
        # TIPBackbone 返回 (out_fused, out_imaging, out_tabular)
        if self.hparams.eval_datatype in ['imaging', 'tabular']:
            # For single modality, forward expects only one tensor
            # Need to decide how single-modality models' forward is structured
            # Assuming it takes just x_i or x_t directly
            # For simplicity, if it's single-modal, return its output three times
            single_mod_output = self.model(x_tuple[0] if self.hparams.eval_datatype == 'imaging' else x_tuple[1])
            if len(single_mod_output.shape) == 1:
                single_mod_output = torch.unsqueeze(single_mod_output, 0)
            return single_mod_output, single_mod_output, single_mod_output
        elif self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']:
            # TIPBackbone 应该返回融合、图像和表格的预测
            # 将 None 传递给 missing_mask，如果它不存在
            # 确保 x_tuple 始终有三个元素
            return self.model(x_tuple)
        else:
            raise ValueError(f"Unsupported eval_datatype: {self.hparams.eval_datatype}")

    def _common_step(self, batch: Tuple, stage: str):
        """
        一个通用的处理训练、验证和测试步骤的函数。
        根据 self.missing_tabular 的值解包 batch。
        """
        if self.missing_tabular:
            # 当 self.missing_tabular 为 True 时，dataloader 返回 (im, tab, missing_mask, path), label
            # batch 实际上是 [(im, tab, missing_mask, path), label]
            (x_imaging, x_tabular, missing_mask, path_dummy), y = batch
        else:
            # 当 self.missing_tabular 为 False 时，dataloader 返回 (im, tab, path), label
            # batch 实际上是 [(im, tab, path), label]
            (x_imaging, x_tabular, path_dummy), y = batch
            # 如果 missing_tabular 为 False，向 forward 传入 None 作为缺失掩码
            missing_mask = None # 显式设置为 None

        # 获取所有输出：融合预测、图像模态独立预测、表格模态独立预测
        # forward 方法接收 (x_imaging, x_tabular, missing_mask_or_None)
        y_fused, y_imaging, y_tabular = self.forward((x_imaging, x_tabular, missing_mask))

        # 计算损失
        loss_fused = self.criterion(y_fused, y)
        loss_imaging = self.criterion(y_imaging, y)
        loss_tabular = self.criterion(y_tabular, y)

        # 指标计算 (使用融合输出)
        y_hat_log = torch.softmax(y_fused.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat_log = y_hat_log[:, 1]

        if stage == 'train':
            self.acc_train(y_hat_log, y)
            self.auc_train(y_hat_log, y)
            self.f1_train(y_hat_log, y)
            self.precision_train(y_hat_log, y)
            self.balanced_acc_train(y_hat_log, y)
            self.log(f'eval.{stage}.loss', loss_fused, on_epoch=True, on_step=False) # 记录融合损失
        elif stage == 'val':
            self.acc_val(y_hat_log, y)
            self.auc_val(y_hat_log, y)
            self.f1_val(y_hat_log, y)
            self.precision_val(y_hat_log, y)
            self.balanced_acc_val(y_hat_log, y)
            self.confmat_val(y_hat_log.argmax(dim=-1) if self.hparams.num_classes > 2 else y_hat_log > 0.5, y)
            self.log(f'eval.{stage}.loss', loss_fused, on_epoch=True, on_step=False)
        elif stage == 'test':
            self.acc_test(y_hat_log, y)
            self.auc_test(y_hat_log, y)
            self.f1_test(y_hat_log, y)
            self.balanced_acc_test(y_hat_log, y)
            self.prc_auc_test(y_hat_log, y)
            self.confmat_test(y_hat_log.argmax(dim=-1) if self.hparams.num_classes > 2 else y_hat_log > 0.5, y)
            self.log(f'eval.{stage}.loss', loss_fused, on_epoch=True, on_step=False)
        
        return loss_fused, loss_imaging, loss_tabular, y_imaging, y_tabular, y

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        # 单模态情况下不应用模态不平衡逻辑
        if self.hparams.eval_datatype in ['imaging', 'tabular']:
            # 单模态 batch 通常是 (data, label)
            x, y = batch
            # 根据 hparams.eval_datatype 选择正确的输入张量
            if self.hparams.eval_datatype == 'imaging':
                input_tensor = x
            else: # 'tabular'
                input_tensor = x
            
            # 传入 dummy tuple，因为 forward 期望 Tuple[Tensor, Tensor, Any]
            y_fused, _, _ = self.forward((input_tensor, torch.empty_like(input_tensor), None)) # 使用 empty_like 创建形状匹配的张量，或传递 None
            loss = self.criterion(y_fused, y)
            self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
            
            # Log metrics for single modality (using fused output which is actually single mod output)
            y_hat_log = torch.softmax(y_fused.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat_log = y_hat_log[:, 1]
            self.acc_train(y_hat_log, y)
            self.auc_train(y_hat_log, y)
            self.f1_train(y_hat_log, y)
            self.precision_train(y_hat_log, y)
            self.balanced_acc_train(y_hat_log, y)
            return loss

        elif self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']:
            loss_fused, loss_imaging, loss_tabular, y_imaging, y_tabular, y = self._common_step(batch, 'train')

            # 3. 计算 k 值 (基于 imaging_trace_list)
            k = 1.0
            if self.current_epoch > 0: # 只有在第一个epoch之后才计算k
                if len(self.imaging_trace_list) >= 10:
                    tr1 = sum(self.imaging_trace_list[-10:]) / 10
                    tr2 = sum(self.imaging_trace_list[-11:-1]) / 10
                elif len(self.imaging_trace_list) >= 2: # 如果不足10个，至少用最近2个
                    tr1 = self.imaging_trace_list[-1]
                    tr2 = self.imaging_trace_list[-2]
                else: # 否则，k为1
                    tr1 = 1.0 # 避免除以0
                    tr2 = 1.0
                if tr1 != 0:
                    k = (tr1 - tr2) / tr1
                else:
                    k = 1.0 # 避免除以0

            # 4. 计算并调整梯度 (模态不平衡核心逻辑)
            opt = self.optimizers()
            opt.zero_grad() # 清零所有梯度

            # 备份全局模型参数，用于梯度调整
            global_imaging_params = {name: p.clone().detach() for name, p in self.model.named_parameters() if 'encoder_imaging' in name or 'head_imaging' in name}
            global_tabular_params = {name: p.clone().detach() for name, p in self.model.named_parameters() if 'encoder_tabular' in name or 'head_tabular' in name}

            # 损失的反向传播 (retain_graph=True 以便后续计算其他损失的梯度)
            self.manual_backward(loss_fused, retain_graph=True)
            self.manual_backward(loss_imaging, retain_graph=True)
            self.manual_backward(loss_tabular) # 最后一个损失不需要 retain_graph=True

            # 计算分数
            score_imaging = torch.sum(self.softmax(y_imaging)[torch.arange(y_imaging.size(0)), y])
            score_tabular = torch.sum(self.softmax(y_tabular)[torch.arange(y_tabular.size(0)), y])

            beta = 0.0
            beta2 = 0.0

            if (score_imaging > score_tabular) and k > 0.04:
                gap = self.tanh(score_imaging - score_tabular)
                beta = 0.95 * torch.exp(gap)
                beta2 = 0.0
            elif (score_tabular > score_imaging) and k > 0.04:
                gap = self.tanh(score_tabular - score_imaging)
                beta2 = 0.1 * torch.exp(gap)
                beta = 0.0
            else:
                beta = 0.0
                beta2 = 0.0

            # 应用梯度调整
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if 'encoder_imaging' in name or 'head_imaging' in name:
                        if name in global_imaging_params:
                            param.grad += beta * (param.data - global_imaging_params[name].data)
                    elif 'encoder_tabular' in name or 'head_tabular' in name:
                        if name in global_tabular_params:
                            param.grad += beta2 * (param.data - global_tabular_params[name].data)
            
            # 5. 计算 FIM (Fisher Information Matrix)
            if self._fim_imaging_epoch_sum is None: # 第一次运行，初始化FIM累加器
                self._fim_imaging_epoch_sum = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters() if 'encoder_imaging' in name or 'head_imaging' in name}
                self._fim_tabular_epoch_sum = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters() if 'encoder_tabular' in name or 'head_tabular' in name}

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if 'encoder_imaging' in name or 'head_imaging' in name:
                        self._fim_imaging_epoch_sum[name] += (param.grad.data * param.grad.data) / len(batch) # 累加并按 batch_size 平均
                    elif 'encoder_tabular' in name or 'head_tabular' in name:
                        self._fim_tabular_epoch_sum[name] += (param.grad.data * param.grad.data) / len(batch) # 累加并按 batch_size 平均

            # 手动执行优化器步骤
            opt.step()
            
            return loss_fused # 返回主融合损失作为记录
        else:
            raise ValueError(f"Unsupported eval_datatype: {self.hparams.eval_datatype}")


    def training_epoch_end(self, outputs) -> None:
        # Metrics logging
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

        # FIM Traces的计算和记录
        if self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular'] and self._fim_imaging_epoch_sum is not None:
            # 计算每个参数的FIM平均值
            fim_trace_imaging = sum([fim_val.mean().item() for fim_val in self._fim_imaging_epoch_sum.values()])
            fim_trace_tabular = sum([fim_val.mean().item() for fim_val in self._fim_tabular_epoch_sum.values()])

            self.imaging_trace_list.append(fim_trace_imaging)
            self.tabular_trace_list.append(fim_trace_tabular)

            self.log('fim_trace_imaging', fim_trace_imaging, on_epoch=True)
            self.log('fim_trace_tabular', fim_trace_tabular, on_epoch=True)

            # 重置FIM累加器
            # 需要重新创建字典，因为直接修改其中的 Tensor 值可能会影响计算图或引用
            self._fim_imaging_epoch_sum = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters() if 'encoder_imaging' in name or 'head_imaging' in name}
            self._fim_tabular_epoch_sum = {name: torch.zeros_like(param.data) for name, param in self.model.named_parameters() if 'encoder_tabular' in name or 'head_tabular' in name}
        
    def validation_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        if self.hparams.eval_datatype in ['imaging', 'tabular']:
            x, y = batch
            if self.hparams.eval_datatype == 'imaging':
                input_tensor = x
            else: # 'tabular'
                input_tensor = x
            y_fused, _, _ = self.forward((input_tensor, torch.empty_like(input_tensor), None))
            loss = self.criterion(y_fused, y)
            
            # Metrics logging
            y_hat_log = torch.softmax(y_fused.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat_log = y_hat_log[:, 1]
            self.acc_val(y_hat_log, y)
            self.auc_val(y_hat_log, y)
            self.f1_val(y_hat_log, y)
            self.precision_val(y_hat_log, y)
            self.balanced_acc_val(y_hat_log, y)
            self.confmat_val(y_hat_log.argmax(dim=-1) if self.hparams.num_classes > 2 else y_hat_log > 0.5, y)
            self.log('eval.val.loss', loss, on_epoch=True, on_step=False)
            return loss
        elif self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']:
            loss_fused, _, _, _, _, _ = self._common_step(batch, 'val')
            return loss_fused
        else:
            raise ValueError(f"Unsupported eval_datatype: {self.hparams.eval_datatype}")

    def validation_epoch_end(self, outputs) -> None:
        # Metrics logging
        self.log('eval.val.f1', self.f1_val.compute(), on_epoch=True)
        self.log('eval.val.precision', self.precision_val.compute(), on_epoch=True)
        self.log('eval.val.balanced_acc', self.balanced_acc_val.compute(), on_epoch=True)
        self.log('eval.val.acc', self.acc_val.compute(), on_epoch=True)
        self.log('eval.val.auc', self.auc_val.compute(), on_epoch=True)
        
        # log confusion matrix only at the end of epoch
        # self.log('eval.val.confmat', self.confmat_val.compute(), on_epoch=True) # ConfusionMatrix often not directly loggable

        self.f1_val.reset()
        self.precision_val.reset()
        self.balanced_acc_val.reset()
        self.acc_val.reset()
        self.auc_val.reset()
        self.confmat_val.reset()

    def test_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        if self.hparams.eval_datatype in ['imaging', 'tabular']:
            x, y = batch
            if self.hparams.eval_datatype == 'imaging':
                input_tensor = x
            else: # 'tabular'
                input_tensor = x
            y_fused, _, _ = self.forward((input_tensor, torch.empty_like(input_tensor), None))
            loss = self.criterion(y_fused, y)
            
            # Metrics logging
            y_hat_log = torch.softmax(y_fused.detach(), dim=1)
            if self.hparams.num_classes == 2:
                y_hat_log = y_hat_log[:, 1]
            self.acc_test(y_hat_log, y)
            self.auc_test(y_hat_log, y)
            self.f1_test(y_hat_log, y)
            self.balanced_acc_test(y_hat_log, y)
            self.prc_auc_test(y_hat_log, y)
            self.confmat_test(y_hat_log.argmax(dim=-1) if self.hparams.num_classes > 2 else y_hat_log > 0.5, y)
            self.log('eval.test.loss', loss, on_epoch=True, on_step=False)
            return loss
        elif self.hparams.eval_datatype in ['multimodal', 'imaging_and_tabular']:
            loss_fused, _, _, _, _, _ = self._common_step(batch, 'test')
            return loss_fused
        else:
            raise ValueError(f"Unsupported eval_datatype: {self.hparams.eval_datatype}")

    def test_epoch_end(self, outputs) -> None:
        # Metrics logging
        self.log('eval.test.f1', self.f1_test.compute(), on_epoch=True)
        self.log('eval.test.balanced_acc', self.balanced_acc_test.compute(), on_epoch=True)
        self.log('eval.test.prc_auc', self.prc_auc_test.compute(), on_epoch=True)
        self.log('eval.test.acc', self.acc_test.compute(), on_epoch=True)
        self.log('eval.test.auc', self.auc_test.compute(), on_epoch=True)

        self.f1_test.reset()
        self.balanced_acc_test.reset()
        self.prc_auc_test.reset()
        self.acc_test.reset()
        self.auc_test.reset()
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
            },
            # 关键：明确告诉 PyTorch Lightning 您将手动控制优化器
            # 这需要在 Trainer 初始化时也传入 `manual_optimization=True`
            "monitor": 'eval.val.loss' # 仍然需要一个monitor
        }