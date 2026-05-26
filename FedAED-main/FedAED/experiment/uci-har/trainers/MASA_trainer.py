
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
from torch.nn import functional as F

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientMASA(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model, 
        label_dict=None,
        num_class=None
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.num_class = num_class if num_class is not None else 10
        
        # 门控网络（延迟初始化）
        self.gate_network = None
        self._gate_initialized = False
        
        # 客户端状态
        self.client_cluster_id = None
        self.attention_weights = None
        self.result = None
        self.test_true = None
        self.test_pred = None
        self.train_groundtruth = None

    def _init_gate_network(self, feature_dim):
        """动态初始化门控网络"""
        self.gate_network = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # 初始化参数
        for layer in self.gate_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.constant_(layer.bias, 0.0)
        
        self._gate_initialized = True

    def get_parameters(self):
        return self.model.state_dict()

    def get_model_result(self):
        return self.result
    
    def get_test_true(self):
        return self.test_true
    
    def get_test_pred(self):
        return self.test_pred
    
    def get_train_groundtruth(self):
        return self.train_groundtruth

    def update_weights(self):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
        
        for epoch in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                
                # 数据准备
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device).float()
                    x_b = x_b.to(self.device).float()
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                    
                    outputs, features = self.model(x_a, x_b, l_a, l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device).float()
                    l = l.to(self.device)
                    y = y.to(self.device)
                    
                    outputs, features = self.model(x, l)
                
                # 初始化门控网络（仅首次）
                if not self._gate_initialized:
                    feature_dim = features.shape[1] if len(features.shape) > 1 else 64
                    self._init_gate_network(feature_dim)
                
                # 门控计算
                with torch.no_grad():
                    if len(features.shape) > 2:
                        features = features.mean(dim=1)
                    gate_weight = self.gate_network(features)
                    effective_lr = self.args.learning_rate * (0.1 + 0.9 * gate_weight.mean())
                
                # 调整学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = effective_lr.item()
                
                # 计算损失（确保loss保持为Tensor）
                if not self.multilabel:
                    outputs = F.log_softmax(outputs, dim=1)
                loss = self.criterion(outputs, y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                
                # 确保loss以Tensor形式传递（关键修复）
                loss_tensor = loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=self.device)
                
                # 保存结果
                if not self.multilabel:
                    self.eval.append_classification_results(
                        y.detach().cpu(),
                        outputs.detach().cpu(),
                        loss_tensor.detach().cpu()
                    )
                else:
                    self.eval.append_multilabel_results(
                        y.detach().cpu(),
                        outputs.detach().cpu(),
                        loss_tensor.detach().cpu()
                    )
        
        # 计算结果
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
