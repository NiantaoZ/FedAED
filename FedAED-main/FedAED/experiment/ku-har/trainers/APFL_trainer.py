import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientAPFL(object):
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
        
        # APFL specific parameters
        self.alpha = args.alpha if hasattr(args, 'alpha') else 0.5
        self.model_per = copy.deepcopy(self.model)  # 个性化模型
        
        # 优化器
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
        self.optimizer_per = torch.optim.SGD(
            self.model_per.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )
        
        # 存储训练结果
        self.result = None
        self.test_true = None
        self.test_pred = None
        self.train_groundtruth = None
        
    def get_parameters(self):
        """返回全局模型参数"""
        return self.model.state_dict()
    
    def get_personalized_parameters(self):
        """返回个性化模型参数"""
        return self.model_per.state_dict()
    
    def get_model_result(self):
        return self.result
    
    def get_test_true(self):
        return self.test_true
    
    def get_test_pred(self):
        return self.test_pred
    
    def get_train_groundtruth(self):
        return self.train_groundtruth

    def alpha_update(self, model_grads, model_per_grads):
        """更新混合权重alpha - 使用显式传入的梯度"""
        grad_alpha = 0.0
        valid_params_count = 0
        
        for l_grad, p_grad, l_param, p_param in zip(model_grads, model_per_grads, 
                                                   self.model.parameters(), self.model_per.parameters()):
            # 使用显式传入的梯度，避免直接访问.grad属性
            if l_grad is None or p_grad is None:
                continue
                
            # 计算差异和梯度
            dif = p_param.data - l_param.data
            grad = self.alpha * p_grad + (1 - self.alpha) * l_grad
            
            # 确保张量形状匹配
            dif_flat = dif.view(-1)
            grad_flat = grad.view(-1)
            
            if dif_flat.shape[0] == grad_flat.shape[0]:
                grad_alpha += dif_flat.dot(grad_flat).item()
                valid_params_count += 1
        
        if valid_params_count == 0:
            # 如果没有有效参数，跳过alpha更新
            return
        
        # 添加正则化项并更新alpha
        grad_alpha += 0.02 * self.alpha
        
        # 使用适当的学习率更新alpha
        alpha_lr = getattr(self.args, 'alpha_lr', 0.001)
        self.alpha = self.alpha - alpha_lr * grad_alpha
        
        # 确保alpha在[0,1]范围内
        self.alpha = max(0.0, min(1.0, self.alpha))

    def _get_model_gradients(self, model):
        """安全地获取模型梯度"""
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.clone())
            else:
                gradients.append(None)
        return gradients

    def update_weights(self):
        """更新模型权重 - APFL训练过程"""
        # 设置模型为训练模式
        self.model.train()
        self.model_per.train()
        
        # 初始化评估器
        self.eval = EvalMetric(self.multilabel)
        
        # 存储所有批次的真实标签和预测结果
        all_true = []
        all_pred = []
        
        for epoch in range(int(self.args.local_epochs)):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch_data in enumerate(self.dataloader):
                # 对于extrasensory数据集，限制批次数量
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue
                
                # 准备数据
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device).float()
                    x_b = x_b.to(self.device).float()
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device).float()
                    l = l.to(self.device)
                    y = y.to(self.device)
                
                # ===== 训练全局模型 =====
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                # 前向传播
                if self.args.modality == "multimodal":
                    outputs, _ = self.model(x_a, x_b, l_a, l_b)
                else:
                    outputs, _ = self.model(x, l)
                
                # 对于分类任务应用log_softmax
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                
                # 计算损失和反向传播
                loss_global = self.criterion(outputs, y)
                loss_global.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                
                # 在优化器step之前保存全局模型梯度
                model_grads = self._get_model_gradients(self.model)
                self.optimizer.step()
                
                # ===== 训练个性化模型 =====
                self.model_per.zero_grad()
                self.optimizer_per.zero_grad()
                
                # 前向传播
                if self.args.modality == "multimodal":
                    outputs_per, _ = self.model_per(x_a, x_b, l_a, l_b)
                else:
                    outputs_per, _ = self.model_per(x, l)
                
                # 对于分类任务应用log_softmax
                if not self.multilabel:
                    outputs_per = torch.log_softmax(outputs_per, dim=1)
                
                # 计算损失和反向传播
                loss_per = self.criterion(outputs_per, y)
                loss_per.backward()
                torch.nn.utils.clip_grad_norm_(self.model_per.parameters(), 10.0)
                
                # 在优化器step之前保存个性化模型梯度
                model_per_grads = self._get_model_gradients(self.model_per)
                self.optimizer_per.step()
                
                # ===== 使用保存的梯度更新alpha混合权重 =====
                self.alpha_update(model_grads, model_per_grads)
                
                # 保存结果（使用个性化模型的输出）
                if not self.multilabel:
                    self.eval.append_classification_results(y, outputs_per, loss_per)
                    _, predicted = torch.max(outputs_per.data, 1)
                    all_pred.extend(predicted.cpu().numpy())
                else:
                    self.eval.append_multilabel_results(y, outputs_per, loss_per)
                    predicted = (torch.sigmoid(outputs_per) > 0.5).int()
                    all_pred.extend(predicted.cpu().numpy())
                
                all_true.extend(y.cpu().numpy())
                
                epoch_loss += loss_per.item()
                batch_count += 1
            
            # 打印epoch统计信息
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                #print(f'Client Epoch [{epoch+1}/{self.args.local_epochs}], Average Loss: {avg_loss:.4f}, Alpha: {self.alpha:.4f}')
        
        # 混合全局模型和个性化模型的参数
        self._mix_models()
        
        # 计算最终评估指标
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
        
        # 保存真实标签和预测结果
        self.train_groundtruth = all_true
        self.test_pred = all_pred
        
        return self.result

    def _mix_models(self):
        """混合全局模型和个性化模型的参数"""
        with torch.no_grad():
            for lp, p in zip(self.model_per.parameters(), self.model.parameters()):
                lp.data = (1 - self.alpha) * p.data + self.alpha * lp.data

    def test(self, test_dataloader):
        """在测试集上评估模型"""
        self.model_per.eval()  # 使用个性化模型进行测试
        self.eval = EvalMetric(self.multilabel)
        
        test_true = []
        test_pred = []
        
        with torch.no_grad():
            for batch_data in test_dataloader:
                # 准备数据
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device).float()
                    x_b = x_b.to(self.device).float()
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device).float()
                    l = l.to(self.device)
                    y = y.to(self.device)
                
                # 前向传播
                if self.args.modality == "multimodal":
                    outputs, _ = self.model_per(x_a, x_b, l_a, l_b)
                else:
                    outputs, _ = self.model_per(x, l)
                
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)
                    self.eval.append_classification_results(y, outputs, torch.tensor(0.0))
                    _, predicted = torch.max(outputs.data, 1)
                    test_pred.extend(predicted.cpu().numpy())
                else:
                    self.eval.append_multilabel_results(y, outputs, torch.tensor(0.0))
                    predicted = (torch.sigmoid(outputs) > 0.5).int()
                    test_pred.extend(predicted.cpu().numpy())
                
                test_true.extend(y.cpu().numpy())
        
        # 计算测试指标
        if not self.multilabel:
            test_result = self.eval.classification_summary()
        else:
            test_result = self.eval.multilabel_summary()
        
        self.test_true = test_true
        self.test_pred = test_pred
        
        return test_result

    def set_parameters(self, parameters):
        """设置全局模型参数"""
        self.model.load_state_dict(parameters)

    def set_personalized_parameters(self, parameters):
        """设置个性化模型参数"""
        self.model_per.load_state_dict(parameters)

    def get_alpha(self):
        """返回当前的alpha值"""
        return self.alpha

    def set_alpha(self, alpha):
        """设置alpha值"""
        self.alpha = max(0.0, min(1.0, alpha))
