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


class Clientperavg(object):
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
        
        # PerAvg 特定参数
        self.beta = args.beta if hasattr(args, 'beta') else 0.01  # 个性化学习率
        self.personalized_model = copy.deepcopy(self.model)  # 个性化模型
        self.personalized_model.to(self.device)
        
        # 保存历史模型参数用于平均
        self.model_history = []
        self.max_history = 3  # 保存最近3个模型
        
    def get_parameters(self):
        # 返回全局模型参数
        return self.model.state_dict()
    
    def get_personalized_parameters(self):
        # 返回个性化模型参数
        return self.personalized_model.state_dict()
    
    def get_model_result(self):
        return self.result
    
    def get_test_true(self):
        return self.test_true
    
    def get_test_pred(self):
        return self.test_pred
    
    def get_train_groundtruth(self):
        return self.train_groundtruth

    def update_weights(self):
        # PerAvg 算法核心：先训练全局模型，然后进行个性化平均
        
        # 阶段1: 训练全局模型
        self.model.train()
        self.eval = EvalMetric(self.multilabel)
        
        # 优化器
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )
            
        # 保存初始模型用于 FedProx
        last_global_model = copy.deepcopy(self.model)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: 
                    continue
                    
                self.model.zero_grad()
                optimizer.zero_grad()
                
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(
                        x.float(), l
                    )
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                # backward
                loss = self.criterion(outputs, y)

                # 如果是FedProx，添加正则化项
                if hasattr(optimizer, 'proximal_regularization'):
                    fedprox_loss = optimizer.proximal_regularization(last_global_model)
                    loss += fedprox_loss

                # backward
                loss.backward()
                
                # clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    10.0
                )
                optimizer.step()
                
                # save results
                if not self.multilabel: 
                    self.eval.append_classification_results(
                        y, 
                        outputs, 
                        loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        loss
                    )
        
        # 保存当前训练后的全局模型到历史
        self.model_history.append(copy.deepcopy(self.model.state_dict()))
        if len(self.model_history) > self.max_history:
            self.model_history.pop(0)  # 保持最近的历史
        
        # 阶段2: 更新个性化模型（模型平均）
        self.update_personalized_model()
        
        # 阶段3: 使用个性化模型进行评估
        self.evaluate_personalized_model()
        
        # 返回全局模型参数（用于服务器聚合）
        return self.model.state_dict()
    
    def update_personalized_model(self):
        """PerAvg核心：更新个性化模型通过加权平均"""
        if not self.model_history:
            return
        
        # 计算加权平均
        personalized_params = {}
        
        # 初始化个性化模型参数
        for name, param in self.personalized_model.named_parameters():
            personalized_params[name] = torch.zeros_like(param.data)
        
        # 对历史模型进行加权平均（越近的模型权重越高）
        total_weight = 0
        for i, model_params in enumerate(self.model_history):
            weight = i + 1  # 越近的模型权重越高
            total_weight += weight
            
            for name, param in self.personalized_model.named_parameters():
                if name in model_params:
                    personalized_params[name] += weight * model_params[name]
        
        # 应用加权平均
        for name, param in self.personalized_model.named_parameters():
            param.data = personalized_params[name] / total_weight
        
        # 可选：在个性化模型上进行少量微调
        self.fine_tune_personalized_model()
    
    def fine_tune_personalized_model(self):
        """在个性化模型上进行少量微调"""
        self.personalized_model.train()
        optimizer = torch.optim.SGD(
            self.personalized_model.parameters(), 
            lr=self.beta,  # 使用较小的个性化学习率
            momentum=0.9,
            weight_decay=1e-5
        )
        
        # 只进行少量迭代
        for batch_idx, batch_data in enumerate(self.dataloader):
            if batch_idx >= 2:  # 只微调2个批次
                break
                
            if self.args.dataset == 'extrasensory' and batch_idx > 5:
                break
                
            self.personalized_model.zero_grad()
            optimizer.zero_grad()
            
            if self.args.modality == "multimodal":
                x_a, x_b, l_a, l_b, y = batch_data
                x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                
                outputs, _ = self.personalized_model(
                    x_a.float(), x_b.float(), l_a, l_b
                )
            else:
                x, l, y = batch_data
                x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                
                outputs, _ = self.personalized_model(
                    x.float(), l
                )
            
            if not self.multilabel: 
                outputs = torch.log_softmax(outputs, dim=1)
                
            loss = self.criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.personalized_model.parameters(), 
                10.0
            )
            optimizer.step()
    
    def evaluate_personalized_model(self):
        """使用个性化模型进行评估"""
        self.personalized_model.eval()
        eval_metric = EvalMetric(self.multilabel)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                    
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    outputs, _ = self.personalized_model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    outputs, _ = self.personalized_model(
                        x.float(), l
                    )
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                
                loss = self.criterion(outputs, y)
                
                if not self.multilabel: 
                    eval_metric.append_classification_results(y, outputs, loss)
                else:
                    eval_metric.append_multilabel_results(y, outputs, loss)
        
        # 使用个性化模型的评估结果
        if not self.multilabel:
            self.result = eval_metric.classification_summary()
        else:
            self.result = eval_metric.multilabel_summary()
