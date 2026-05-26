import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientFedPHP(object):
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
        
        # FedPHP specific parameters
        self.mu = 0.9  # 全局模型混合参数
        self.lamda = 0.1  # MMD正则化权重
        
        # 初始化服务器模型副本
        self.model_s = copy.deepcopy(self.model)
        for param in self.model_s.parameters():
            param.requires_grad = False
        
    def get_rep(self, x_a=None, x_b=None, l_a=None, l_b=None, x=None, l=None):
        """获取模型的特征表示"""
        if self.args.modality == "multimodal":
            # 获取多模态模型的倒数第二层输出作为特征
            outputs, features = self.model(x_a.float(), x_b.float(), l_a, l_b)
            return features
        else:
            # 获取单模态模型的倒数第二层输出作为特征
            outputs, features = self.model(x.float(), l)
            return features
    
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
    
    def extract_features(self):
        """提取模型特征用于联邦特征聚合"""
        self.model.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                    
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device)
                    x_b = x_b.to(self.device)
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                    
                    # 获取特征表示
                    rep = self.get_rep(x_a=x_a, x_b=x_b, l_a=l_a, l_b=l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device)
                    l = l.to(self.device)
                    y = y.to(self.device)
                    
                    rep = self.get_rep(x=x, l=l)
                
                all_features.append(rep.cpu().detach().numpy())
                all_labels.append(y.cpu().numpy())
        
        if len(all_features) > 0:
            self.all_features = np.vstack(all_features)
            
            # 修复标签拼接问题
            if self.multilabel:
                # 对于多标签分类，标签是二维的，使用vstack
                self.all_labels = np.vstack(all_labels)
            else:
                # 对于单标签分类，标签是一维的，使用hstack
                self.all_labels = np.hstack(all_labels)
        else:
            self.all_features = np.array([])
            self.all_labels = np.array([])
        return self.all_features, self.all_labels
    
    def MMD(self, x, y, kernel='rbf'):
        """计算最大均值差异(MMD)"""
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        
        XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
                     torch.zeros(xx.shape).to(self.device),
                     torch.zeros(xx.shape).to(self.device))
        
        if kernel == "rbf":
            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5*dxx/a)
                YY += torch.exp(-0.5*dyy/a)
                XY += torch.exp(-0.5*dxy/a)
        
        return torch.mean(XX + YY - 2. * XY)

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
                
                # 准备数据
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device)
                    x_b = x_b.to(self.device)
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                    
                    # 前向传播
                    outputs, features = self.model(x_a.float(), x_b.float(), l_a, l_b)
                    
                    # 获取服务器模型特征
                    with torch.no_grad():
                        _, features_s = self.model_s(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device)
                    l = l.to(self.device)
                    y = y.to(self.device)
                    
                    # 前向传播
                    outputs, features = self.model(x.float(), l)
                    
                    # 获取服务器模型特征
                    with torch.no_grad():
                        _, features_s = self.model_s(x.float(), l)
                
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                
                # 计算损失
                loss = self.criterion(outputs, y) * (1 - self.lamda)
                
                # 计算MMD损失
                if features is not None and features_s is not None:
                    mmd_loss = self.MMD(features, features_s) * self.lamda
                    total_loss = loss + mmd_loss
                else:
                    total_loss = loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                
                # 保存结果
                if not self.multilabel:
                    self.eval.append_classification_results(y, outputs, total_loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, total_loss)
        
        # 提取特征用于联邦聚合
        self.extract_features()
        
        # 计算最终评估指标
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
    
    def set_parameters(self, global_model):
        """更新本地模型参数"""
        # 更新服务器模型副本
        for new_param, old_param in zip(global_model.parameters(), self.model_s.parameters()):
            old_param.data = new_param.data.clone()
        
        # 混合全局模型和本地模型
        for new_param, old_param in zip(global_model.parameters(), self.model.parameters()):
            old_param.data = new_param.data * (1 - self.mu) + old_param.data * self.mu
