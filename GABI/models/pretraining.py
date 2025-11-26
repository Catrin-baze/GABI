from typing import List, Tuple, Dict, Any

import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
from sklearn.linear_model import LogisticRegression
from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
# import torchvision.models as models # 不再需要直接导入 models，因为我们不加载ImageNet权重

from models.TabularEncoder import TabularEncoder

class Pretraining(pl.LightningModule):
    
  def __init__(self, hparams) -> None:
    super().__init__()
    self.save_hyperparameters(hparams)
    self.encoder_imaging = None # 初始化为 None
    self.encoder_name_imaging = '' # 初始化为空字符串

  def initialize_imaging_encoder_and_projector(self) -> None:
    """
    Selects appropriate resnet50 encoder and loads weights from imaging_pretrain_checkpoint.
    """
    # 确保模型是 resnet50
    if self.hparams.model != 'resnet50':
        raise ValueError(f"Expected model 'resnet50', but got '{self.hparams.model}'.")


    self.pooled_dim = 2048 

    # 加载权重
    if hasattr(self.hparams, 'imaging_pretrain_checkpoint') and self.hparams.imaging_pretrain_checkpoint:
        print(f'Loading imaging checkpoint: {self.hparams.imaging_pretrain_checkpoint}')
        checkpoint = torch.load(self.hparams.imaging_pretrain_checkpoint)
        original_args = checkpoint['hyper_parameters'] # 这里 original_args 是 dict
        state_dict = checkpoint['state_dict']

        if 'encoder_imaging.0.weight' in state_dict:
            self.encoder_name_imaging = 'encoder_imaging.'
            resnet_base = torchvision.models.resnet50(pretrained=False) # 不加载 ImageNet 预训练权重
            self.encoder_imaging = torch.nn.Sequential(*list(resnet_base.children())[:-1])
            print("Initialized Imaging Encoder as a standard ResNet50 Sequential model.")
        else:
            encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
            # 使用 torchvision_ssl_encoder 来创建编码器
            self.encoder_imaging = torchvision_ssl_encoder(original_args['model'])
            print("Initialized Imaging Encoder using torchvision_ssl_encoder.")

            # Get the correct encoder name prefix
            # original_args['loss'] 必须存在于 checkpoint 的 hyper_parameters 中
            if 'loss' in original_args:
                self.encoder_name_imaging = encoder_name_dict[original_args['loss']]
            else:
                print("Warning: 'loss' key not found in checkpoint hyper_parameters. Assuming default 'encoder_imaging.' for prefix.")
                self.encoder_name_imaging = 'encoder_imaging.'

        # Remove prefix and fc layers, then load state_dict
        state_dict_encoder = {}
        for k in list(state_dict.keys()):
            if k.startswith(self.encoder_name_imaging) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_encoder[k[len(self.encoder_name_imaging):]] = state_dict[k]

        log = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=False)
        if len(log.missing_keys) > 0:
            print(f"Warning: Missing keys when loading imaging encoder: {log.missing_keys}")
        if len(log.unexpected_keys) > 0:
            print(f"Warning: Unexpected keys when loading imaging encoder: {log.unexpected_keys}")
        
        # Freeze if needed (using hparams.imaging_finetune_strategy or hparams.finetune_strategy)
        finetune_strategy_imaging = self.hparams.imaging_finetune_strategy if hasattr(self.hparams, 'imaging_finetune_strategy') else (self.hparams.finetune_strategy if hasattr(self.hparams, 'finetune_strategy') else None)
        
        if finetune_strategy_imaging == 'frozen':
            for _, param in self.encoder_imaging.named_parameters():
                param.requires_grad = False
            parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
            # assert len(parameters)==0 # 这条断言可能过于严格，因为一些 BN 层可能仍是可训练的
            if len(parameters) != 0:
                print(f"Warning: Imaging encoder frozen but still has {len(parameters)} trainable parameters (e.g., Batch Norm layers).")
            print('Freeze imaging encoder')
        elif finetune_strategy_imaging == 'trainable':
            print('Full finetune imaging encoder')
        elif finetune_strategy_imaging is None:
            print('No imaging finetune strategy specified, imaging encoder will be trainable by default.')
        else:
            raise ValueError(f'Unknown finetune strategy {finetune_strategy_imaging}')
    else:
        # 如果没有提供 imaging_pretrain_checkpoint，则使用默认的 torchvision_ssl_encoder
        # 这意味着编码器将随机初始化 (除非 torchvision_ssl_encoder 内部有其他默认行为)
        self.encoder_imaging = torchvision_ssl_encoder(self.hparams.model)
        self.encoder_name_imaging = 'encoder_imaging.' # 默认名称
        print("No imaging_pretrain_checkpoint provided, initializing imaging encoder with default torchvision_ssl_encoder (randomly initialized).")

    self.projector_imaging = SimCLRProjectionHead(self.pooled_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

  def initialize_tabular_encoder(self) -> None:
    self.field_lengths_tabular = torch.load(self.hparams.field_lengths_tabular)
    self.cat_lengths_tabular = []
    self.con_lengths_tabular = []
    for x in self.field_lengths_tabular:
      if x == 1:
        self.con_lengths_tabular.append(x) 
      else:
        self.cat_lengths_tabular.append(x)
    self.encoder_tabular = TabularEncoder(self.hparams)

  def initialize_tabular_encoder_and_projector(self) -> None:
    self.encoder_tabular = TabularEncoder(self.hparams)
    # self.projector_tabular = SimCLRProjectionHead(self.hparams.embedding_dim, self.hparams.embedding_dim, self.hparams.projection_dim)
    self.projector_tabular = SimCLRProjectionHead(self.hparams.tabular_embedding_dim, self.hparams.tabular_embedding_dim, self.hparams.projection_dim)

  def initialize_classifier_and_metrics(self, nclasses_train, nclasses_val):
    """
    Initializes classifier and metrics. Takes care to set correct number of classes for embedding similarity metric depending on loss.
    """
    # Classifier
    #self.estimator = None
        # === 为每种嵌入类型创建独立的分类器 ===
    # 将原来的 estimator 重命名为 multimodal_estimator 以明确其作用
    self.multimodal_estimator = None 
    self.image_estimator = None
    self.tabular_estimator = None

    # Accuracy calculated against all others in batch of same view except for self (i.e. -1) and all of the other view
    self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train)
    self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val)

    self.top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_train)
    self.top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_val)

    task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
    # 验证集准确率
    self.classifier_acc_val_multimodal = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_val_image = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_val_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

    # 验证集AUC
    self.classifier_auc_val_multimodal = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val_image = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    self.classifier_auc_val_tabular = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    
    # 训练集的指标也同样需要创建 (如果原来有的话)
    self.classifier_acc_train_multimodal = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_train_image = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
    self.classifier_acc_train_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)


  def load_pretrained_imaging_weights(self) -> None:
    """
    Can load imaging encoder with pretrained weights from previous checkpoint/run
    """
    loaded_chkpt = torch.load(self.hparams.imaging_pretrain_checkpoint)
    state_dict = loaded_chkpt['state_dict']
    state_dict_encoder = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
    _ = self.encoder_imaging.load_state_dict(state_dict_encoder, strict=True)
    print("Loaded imaging weights")
    if self.hparams.pretrained_imaging_strategy == 'frozen':
      for _, param in self.encoder_imaging.named_parameters():
        param.requires_grad = False
      parameters = list(filter(lambda p: p.requires_grad, self.encoder_imaging.parameters()))
      assert len(parameters)==0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates encoding of imaging data.
    """
    z, y = self.forward_imaging(x)
    return y

  def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of imaging data.
    """
    y = self.encoder_imaging(x)[0]
    z = self.projector_imaging(y)
    return z, y

  def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection and encoding of tabular data.
    """
    y = self.encoder_tabular(x).flatten(start_dim=1)
    z = self.projector_tabular(y)
    return z, y


  def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_train(logits, labels)
    self.top5_acc_train(logits, labels)
    
    self.log(f"{modality}.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
    self.log(f"{modality}.train.top5", self.top5_acc_train, on_epoch=True, on_step=False)

  def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
    self.top1_acc_val(logits, labels)
    self.top5_acc_val(logits, labels)
    
    self.log(f"{modality}.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
    self.log(f"{modality}.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)


  def training_epoch_end(self, train_step_outputs: List[Any]) -> None:
    """
    Train and log classifier, including training accuracy.
    """
    if self.current_epoch != 0 and self.current_epoch % self.hparams.classifier_freq == 0:
        def _stack_outputs_by_key(outputs, key):
            embeddings = torch.cat([x[key] for x in outputs], dim=0)
            labels = torch.cat([x['labels'] for x in outputs], dim=0)
            return embeddings.cpu().numpy(), labels.cpu().numpy()

        # 1. 训练图像分类器
        image_embeddings, labels_img = _stack_outputs_by_key(train_step_outputs, 'image_embeddings')
        self.image_estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(image_embeddings, labels_img)
        
        # 计算并记录图像分类器的训练准确率
        preds_img_train, _ = self.predict_live_estimator(self.image_estimator, torch.from_numpy(image_embeddings))
        self.classifier_acc_train_image(preds_img_train, torch.from_numpy(labels_img))
        self.log('classifier.train.image_accuracy', self.classifier_acc_train_image, on_epoch=True, on_step=False)
        
        # 2. 训练表格分类器
        tabular_embeddings, labels_tab = _stack_outputs_by_key(train_step_outputs, 'tabular_embeddings')
        self.tabular_estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(tabular_embeddings, labels_tab)
        
        # 计算并记录表格分类器的训练准确率
        preds_tab_train, _ = self.predict_live_estimator(self.tabular_estimator, torch.from_numpy(tabular_embeddings))
        self.classifier_acc_train_tabular(preds_tab_train, torch.from_numpy(labels_tab))
        self.log('classifier.train.tabular_accuracy', self.classifier_acc_train_tabular, on_epoch=True, on_step=False)
        
        # 3. 训练多模态分类器
        multimodal_embeddings, labels_multi = _stack_outputs_by_key(train_step_outputs, 'multimodal_embeddings')
        self.multimodal_estimator = LogisticRegression(class_weight='balanced', max_iter=1000).fit(multimodal_embeddings, labels_multi)

        # 计算并记录多模态分类器的训练准确率
        preds_multi_train, _ = self.predict_live_estimator(self.multimodal_estimator, torch.from_numpy(multimodal_embeddings))
        self.classifier_acc_train_multimodal(preds_multi_train, torch.from_numpy(labels_multi))
        self.log('classifier.train.multimodal_accuracy', self.classifier_acc_train_multimodal, on_epoch=True, on_step=False)



  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step and calc validation classifier performance
    """
    if self.hparams.log_images:
      self.logger.log_image(key="Image Example", images=[validation_step_outputs[0]['sample_augmentation']])

    # Validate classifier
    if self.multimodal_estimator is not None and self.current_epoch % self.hparams.classifier_freq == 0:
        
        # 同样使用辅助函数来提取数据

        # --- 评估 Image Embedding ---
        image_embeddings, image_labels = self.stack_outputs(validation_step_outputs, 'image_embeddings')
        # 使用 image_estimator 进行预测
        preds_img, probs_img = self.predict_live_estimator(self.image_estimator, image_embeddings) 
        self.classifier_acc_val_image(preds_img, image_labels)
        self.classifier_auc_val_image(probs_img, image_labels)
        self.log('classifier.val.image_accuracy', self.classifier_acc_val_image, on_epoch=True, on_step=False)
        self.log('classifier.val.image_auc', self.classifier_auc_val_image, on_epoch=True, on_step=False)

        # --- 评估 Tabular Embedding ---
        tabular_embeddings, tabular_labels = self.stack_outputs(validation_step_outputs, 'tabular_embeddings')
        # 使用 tabular_estimator 进行预测
        preds_tab, probs_tab = self.predict_live_estimator(self.tabular_estimator, tabular_embeddings)
        self.classifier_acc_val_tabular(preds_tab, tabular_labels)
        self.classifier_auc_val_tabular(probs_tab, tabular_labels)
        self.log('classifier.val.tabular_accuracy', self.classifier_acc_val_tabular, on_epoch=True, on_step=False)
        self.log('classifier.val.tabular_auc', self.classifier_auc_val_tabular, on_epoch=True, on_step=False)
        
        # --- 评估 Multimodal Embedding (原有的逻辑) ---
        multimodal_embeddings, multimodal_labels = self.stack_outputs(validation_step_outputs, 'multimodal_embeddings')
        # 使用 multimodal_estimator 进行预测
        preds_multi, probs_multi = self.predict_live_estimator(self.multimodal_estimator, multimodal_embeddings)
        self.classifier_acc_val_multimodal(preds_multi, multimodal_labels)
        self.classifier_auc_val_multimodal(probs_multi, multimodal_labels)
        self.log('classifier.val.multimodal_accuracy', self.classifier_acc_val_multimodal, on_epoch=True, on_step=False)
        self.log('classifier.val.multimodal_auc', self.classifier_auc_val_multimodal, on_epoch=True, on_step=False)


  def stack_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
    """
    Stack outputs from multiple steps
    """
    labels = outputs[0]['labels']
    embeddings = outputs[0]['embeddings']
    for i in range(1, len(outputs)):
      labels = torch.cat((labels, outputs[i]['labels']), dim=0)
      embeddings = torch.cat((embeddings, outputs[i]['embeddings']), dim=0)

    embeddings = embeddings.detach().cpu()
    labels = labels.cpu()

    return embeddings, labels

  def predict_live_estimator(self, estimator, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用训练好的分类器进行预测。
    """
    # 即使 embeddings 已经在 CPU 上，也必须将其转换为 NumPy 数组
    # 因为 scikit-learn 模型期望 NumPy 数组作为输入。
    embeddings_np = embeddings.numpy() # 这里不再需要 .cpu()，因为它已在 CPU 上

    # 使用 scikit-learn 分类器执行预测
    preds_np = estimator.predict(embeddings_np)
    probs_np = estimator.predict_proba(embeddings_np)

    # 将 NumPy 数组的预测结果和概率转换回 PyTorch 张量
    preds = torch.tensor(preds_np)
    probs = torch.tensor(probs_np)
    
    # 如果是二分类问题，只需要正类的概率
    if self.hparams.num_classes == 2:
        probs = probs[:, 1]

    return preds, probs


  def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
    else:
      raise ValueError('Valid schedulers are "cosine" and "anneal"')
    
    return scheduler