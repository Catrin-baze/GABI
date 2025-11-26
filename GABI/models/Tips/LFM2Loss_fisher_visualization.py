
from typing import List, Tuple, Dict
from torchmetrics import Recall
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import math
from utils.clip_loss import CLIPLoss
from utils.reconstruct_loss import ReconstructionLoss

from models.Tip_utils.Tip_pretraining_balance import Pretraining 
import torchmetrics


class TIP3Loss(Pretraining):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

 
        self.initialize_imaging_encoder_and_projector()
        if self.hparams.imaging_pretrain_checkpoint:
            self.load_pretrained_imaging_weights()
        self.initialize_tabular_encoder_and_projector()
        self.initialize_multimodal_encoder_and_predictor()


        self.classifier = nn.Linear(self.hparams.multimodal_embedding_dim, self.hparams.num_classes)

        self.criterion_cls = nn.CrossEntropyLoss()

        self.criterion_val_itc = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
        self.criterion_train_itc = self.criterion_val_itc
        self.criterion_tr = ReconstructionLoss(
            num_cat=self.hparams.num_cat,
            cat_offsets=self.encoder_tabular.cat_offsets,
            num_con=self.hparams.num_con
        )
        self.criterion_itm = nn.CrossEntropyLoss(reduction='mean')

     
        self.loss_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

   
        num_classes = self.hparams.num_classes
        task = 'binary' if num_classes == 2 else 'multiclass'


        self.modes = ["joint", "tab_only", "image_only"]
        self.test_metrics = nn.ModuleDict()
        for mode in self.modes:
            self.test_metrics[mode] = nn.ModuleDict({
                "accuracy": torchmetrics.Accuracy(task=task, num_classes=num_classes),
                "auc": torchmetrics.AUROC(task=task, num_classes=num_classes),
                "balanced_accuracy": torchmetrics.Accuracy(task=task, num_classes=num_classes, average='macro'),
                "f1": torchmetrics.F1Score(task=task, num_classes=num_classes, average='macro'),
                "precision": torchmetrics.Precision(task=task, num_classes=num_classes, average='macro'),
                "Recall_accuracy": Recall(task=task, num_classes=num_classes, average="macro")
            })


    def compute_certified_robust_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        B, C = logits.shape
        rows = torch.arange(B, device=logits.device)
        f_true = logits[rows, labels]  #

     
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[rows, labels] = False
       
        very_neg = -1e9
        other_logits = logits.clone()
        other_logits[~mask] = very_neg
        f_max_other, _ = torch.max(other_logits, dim=1)  # [B]

        margin = f_true - f_max_other  # [B]
     
        loss_robust = - torch.mean(margin)
        return loss_robust


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        im_views, tab_views, y, _, original_tab = batch

     
        if optimizer_idx == 0:
            z0, image_embeddings = self.forward_imaging(im_views[1])
            z1, tabular_embeddings1 = self.forward_tabular(tab_views[0])

            fusion_features = self.forward_multimodal_feature(
                tabular_features=tabular_embeddings1,
                image_features=image_embeddings
            )
            loss_itc, logits_itc, labels_itc = self.criterion_train_itc(z0, z1, y)
            self.log("multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False)

            o_b = self.classifier(fusion_features)
            loss_cls = self.criterion_cls(o_b, y)
            self.log("multimodal.train.ClassificationLoss", loss_cls, on_epoch=True, on_step=False)

            weights = torch.softmax(self.loss_weights, dim=0)
            loss = weights[0] * loss_cls + weights[1] * loss_itc

            self.log("multimodal.train.total_loss", loss, on_epoch=True, on_step=False)
         
            self.log("multimodal.train.loss_weight_cls", weights[0], on_epoch=True, on_step=False)
            self.log("multimodal.train.loss_weight_itc", weights[1], on_epoch=True, on_step=False)

            predicted_labels = torch.argmax(o_b, dim=1)
            train_accuracy = (predicted_labels == y).float().mean()
            self.log("multimodal.train.accuracy", train_accuracy, on_epoch=True, on_step=False)
        
            self.train_accuracy.update(predicted_labels, y)
            self.train_balanced_accuracy.update(predicted_labels, y)

            return {"loss": loss}

     
        elif optimizer_idx == 1:
   
            if hasattr(self, "encoder_multimodal"):
                if hasattr(self.encoder_multimodal, "a_image"):
                    self.encoder_multimodal.a_image.requires_grad = True
                if hasattr(self.encoder_multimodal, "a_tabular"):
                    self.encoder_multimodal.a_tabular.requires_grad = True

            with torch.no_grad():
                z0, image_embeddings = self.forward_imaging(im_views[1])
                z1, tabular_embeddings1 = self.forward_tabular(tab_views[0])

            fusion_features = self.forward_multimodal_feature(
                tabular_features=tabular_embeddings1,
                image_features=image_embeddings
            )
            logits = self.classifier(fusion_features)

         
            w_image = torch.sigmoid(self.encoder_multimodal.a_image).mean()
            w_tab = torch.sigmoid(self.encoder_multimodal.a_tabular).mean()

           
            B, C = logits.shape
            rows = torch.arange(B, device=logits.device)
            f_true = logits[rows, y]
            very_neg = -1e9
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[rows, y] = False
            other_logits = logits.clone()
            other_logits[~mask] = very_neg
            f_max_other, _ = torch.max(other_logits, dim=1)
            margin = f_true - f_max_other
            loss_robust = -torch.mean(margin)

            lambda_reg = 0.01
            reg = lambda_reg * ((w_image - w_tab) ** 2)
            loss_robust = loss_robust + reg

            self.log("multimodal.weights.robust_loss", loss_robust, on_epoch=True, on_step=False)
            self.log("multimodal.weights.a_image", w_image, on_epoch=True, on_step=False)
            self.log("multimodal.weights.a_tabular", w_tab, on_epoch=True, on_step=False)

            return {"loss": loss_robust}

                

 
    def on_before_optimizer_step(self, optimizer, optimizer_idx):

        device = None
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        imaging_fisher_trace = torch.tensor(0., device=device)
        for param in self.encoder_imaging.parameters():
            if param.grad is not None:
                imaging_fisher_trace = imaging_fisher_trace + param.grad.detach().pow(2).sum()

        tabular_fisher_trace = torch.tensor(0., device=device)
        for param in self.encoder_tabular.parameters():
            if param.grad is not None:
                tabular_fisher_trace = tabular_fisher_trace + param.grad.detach().pow(2).sum()

        multimodal_fusion_encoder_fisher_trace = torch.tensor(0., device=device)
        for param in self.encoder_multimodal.parameters():
            if param.grad is not None:
                multimodal_fusion_encoder_fisher_trace = multimodal_fusion_encoder_fisher_trace + param.grad.detach().pow(2).sum()

        total_model_fisher_trace = torch.tensor(0., device=device)
        for param in self.parameters():
            if param.grad is not None:
                total_model_fisher_trace = total_model_fisher_trace + param.grad.detach().pow(2).sum()

 
        self.log("fisher_trace/imaging_encoder", imaging_fisher_trace, on_step=True, on_epoch=False, sync_dist=True)
        self.log("fisher_trace/tabular_encoder", tabular_fisher_trace, on_step=True, on_epoch=False, sync_dist=True)
        self.log("fisher_trace/multimodal_fusion_encoder", multimodal_fusion_encoder_fisher_trace, on_step=True, on_epoch=False, sync_dist=True)
        self.log("fisher_trace/total_model", total_model_fisher_trace, on_step=True, on_epoch=False, sync_dist=True)

        if total_model_fisher_trace.item() > 0:
            imaging_ratio = imaging_fisher_trace / total_model_fisher_trace
            tabular_ratio = tabular_fisher_trace / total_model_fisher_trace
            fusion_ratio = multimodal_fusion_encoder_fisher_trace / total_model_fisher_trace
            self.log("fisher_ratio/imaging", imaging_ratio, on_step=True, on_epoch=False)
            self.log("fisher_ratio/tabular", tabular_ratio, on_step=True, on_epoch=False)
            self.log("fisher_ratio/fusion", fusion_ratio, on_step=True, on_epoch=False)


    def validation_step(self, batch, batch_idx):
        im_views, tab_views, y, original_im, original_tab = batch
        z0, image_embeddings = self.forward_imaging(original_im) 
        z1, tabular_embeddings1 = self.forward_tabular(original_tab)

        loss_itc, _, _ = self.criterion_val_itc(z0, z1, y)
        self.log("multimodal.val.ITCloss", loss_itc, on_epoch=True, on_step=False)

        fusion_features = self.forward_multimodal_feature(
            tabular_features=tabular_embeddings1,
            image_features=image_embeddings
        )
        o_b = self.classifier(fusion_features)
        loss_cls = self.criterion_cls(o_b, y)
        self.log("multimodal.val.ClassificationLoss", loss_cls, on_epoch=True, on_step=False)

        weights = torch.softmax(self.loss_weights, dim=0)
        loss = weights[0] * loss_cls  + weights[1] * loss_itc

        self.log("supervised.val.loss", loss, on_epoch=True, on_step=False)
        self.log("supervised.val.accuracy", self.val_accuracy, on_epoch=True, on_step=False)
        self.log("supervised.val.balanced_accuracy", self.val_balanced_accuracy, on_epoch=True, on_step=False)

        self.log("multimodal.val.total_loss", loss, on_epoch=True, on_step=False)
        self.log("multimodal.val.loss_weight_cls", weights[0], on_epoch=True, on_step=False)
        self.log("multimodal.val.loss_weight_itc", weights[1], on_epoch=True, on_step=False)
        predicted_labels = torch.argmax(o_b, dim=1)
        val_accuracy = (predicted_labels == y).float().mean()
        self.val_accuracy.update(predicted_labels, y)
        self.val_balanced_accuracy.update(predicted_labels, y)
        self.log("multimodal.val.accuracy", val_accuracy, on_epoch=True, on_step=False)

        return {"sample_augmentation": im_views[1], "tabular_embeddings": tabular_embeddings1, "labels": y, "predictions": F.softmax(o_b, dim=1)}


    def test_step(self, batch, batch_idx):
        """计算并记录 joint/tab/image 三种模式的多指标"""
        data, y = batch
        im, tab, path = data
        im, tab = im.float(), tab.float()

        outputs = self.forward_for_eval(im, tab)
        results = {"labels": y, "path": path}

        for mode, preds in outputs.items():
            probs = F.softmax(preds, dim=1)
            loss = self.criterion_cls(preds, y)
            acc = (torch.argmax(preds, dim=1) == y).float().mean()

        
            self.log(f"test/{mode}_loss", loss, on_epoch=True, prog_bar=(mode == "joint"))
            self.log(f"test/{mode}_acc", acc, on_epoch=True, prog_bar=(mode == "joint"))

     
            preds_labels = torch.argmax(probs, dim=1)

            auc = self.test_metrics[mode]["auc"](probs, y)
            bal_acc = self.test_metrics[mode]["balanced_accuracy"](preds_labels, y)
            f1 = self.test_metrics[mode]["f1"](preds_labels, y)
            precision = self.test_metrics[mode]["precision"](preds_labels, y)
            recall = self.test_metrics[mode]["Recall_accuracy"](preds_labels, y)

            self.log(f"test/{mode}_auc", auc, on_epoch=True)
            self.log(f"test/{mode}_recall", recall, on_epoch=True)
            self.log(f"test/{mode}_balanced_accuracy", bal_acc, on_epoch=True)
            self.log(f"test/{mode}_f1", f1, on_epoch=True)
            self.log(f"test/{mode}_precision", precision, on_epoch=True)

            results[f"preds_{mode}"] = probs

        return results

      
    def configure_optimizers(self) -> Tuple[Dict, Dict]:

        optimizer_backbone = torch.optim.Adam(
            [
                {'params': self.encoder_imaging.parameters()},
                {'params': self.projector_imaging.parameters()},
                {'params': self.encoder_tabular.parameters()},
                {'params': self.projector_tabular.parameters()},
                {'params': self.encoder_multimodal.parameters()}, 
                {'params': self.predictor_tabular.parameters()},

                {'params': self.classifier.parameters()},
                {'params': self.loss_weights},
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )


        fusion_param_list = []
        if hasattr(self, "encoder_multimodal"):
            if hasattr(self.encoder_multimodal, "a_image"):
                fusion_param_list.append(self.encoder_multimodal.a_image)
            if hasattr(self.encoder_multimodal, "a_tabular"):
                fusion_param_list.append(self.encoder_multimodal.a_tabular)


        if len(fusion_param_list) == 0:
            fusion_param_list = [p for p in self.encoder_multimodal.parameters()][:1] 

        optimizer_weights = torch.optim.Adam(
            [{'params': fusion_param_list}],
            lr=self.hparams.lr * 0.5,  
            weight_decay=0.0
        )

        scheduler_backbone = self.initialize_scheduler(optimizer_backbone)

        return [optimizer_backbone, optimizer_weights], [scheduler_backbone]
