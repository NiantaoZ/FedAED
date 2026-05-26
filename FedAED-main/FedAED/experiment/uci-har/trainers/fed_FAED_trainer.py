import collections
import numpy as np
from torch import nn
import torch.nn.functional as F
from .fed_avg_trainer import ClientFedAvg
from .optimizer import FedProxOptimizer
import pandas as pd
import copy, pdb, time, warnings, torch
import math

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class BalancedAdaptiveModalityAlignment(nn.Module):
    """平衡的AMA模块：适度增强特征表示"""
    def __init__(self, num_modalities=2, temperature=0.1):
        super(BalancedAdaptiveModalityAlignment, self).__init__()
        self.num_modalities = num_modalities
        self.temperature = temperature
        
        # 模态重要性权重
        self.modal_importance = nn.Parameter(torch.ones(num_modalities, dtype=torch.float32))
        
    def forward(self, modality_features, modality_mask):
        batch_size = modality_mask.size(0)
        
        # 计算模态质量分数
        quality_scores = []
        for i in range(self.num_modalities):
            if i < len(modality_features) and modality_features[i] is not None:
                feat = modality_features[i].float()
                # 综合质量评估：范数 + 方差 + 稀疏度
                norm_score = torch.norm(feat, p=2, dim=1, keepdim=True)
                var_score = torch.var(feat, dim=1, keepdim=True)
                # 避免除零
                norm_score = norm_score / (torch.max(norm_score) + 1e-8)
                var_score = var_score / (torch.max(var_score) + 1e-8)
                
                quality = norm_score * 0.6 + var_score * 0.4
                quality_scores.append(quality)
            else:
                quality_scores.append(torch.zeros(batch_size, 1, device=modality_mask.device))
        
        quality_scores = torch.cat(quality_scores, dim=1)
        
        # 结合模态重要性和质量分数
        combined_scores = quality_scores * self.modal_importance.unsqueeze(0)
        availability_dist = F.softmax(combined_scores / self.temperature, dim=1)
        availability_dist = availability_dist * modality_mask.float()
        
        # 归一化
        sum_weights = torch.sum(availability_dist, dim=1, keepdim=True)
        sum_weights[sum_weights == 0] = 1
        availability_dist = availability_dist / sum_weights
        
        # 平衡的特征增强
        enhanced_features = []
        for i in range(self.num_modalities):
            if i < len(modality_features) and modality_features[i] is not None:
                base_feature = modality_features[i].float()
                enhanced_feature = base_feature.clone()
                
                # 只从质量较高的模态获取信息
                for j in range(self.num_modalities):
                    if i != j and modality_features[j] is not None and availability_dist[:, j].mean() > 0.3:
                        transfer_weight = availability_dist[:, j].unsqueeze(1) * 0.5  # 适中的权重
                        enhanced_feature = enhanced_feature + transfer_weight * modality_features[j].float()
                
                enhanced_features.append(enhanced_feature)
            else:
                enhanced_features.append(None)
        
        return enhanced_features, availability_dist


class BalancedCrossModalContrastive(nn.Module):
    """平衡的跨模态对比学习"""
    def __init__(self, margin=0.6, alpha=0.3):
        super(BalancedCrossModalContrastive, self).__init__()
        self.margin = margin
        self.alpha = alpha
        
    def safe_feature_alignment(self, feat1, feat2):
        """安全的特征对齐"""
        min_dim = min(feat1.shape[1], feat2.shape[1])
        feat1_aligned = feat1[:, :min_dim]
        feat2_aligned = feat2[:, :min_dim]
        
        feat1_norm = F.normalize(feat1_aligned, dim=1)
        feat2_norm = F.normalize(feat2_aligned, dim=1)
        
        # 使用余弦相似度和MSE的混合
        cosine_sim = F.cosine_similarity(feat1_norm, feat2_norm, dim=1).mean()
        mse_loss = F.mse_loss(feat1_norm, feat2_norm)
        
        return (1 - cosine_sim) * 0.7 + mse_loss * 0.3
    
    def intra_modal_alignment(self, student_features, teacher_features):
        """模态内对齐 - 平衡实现"""
        loss = 0.0
        count = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            if s_feat is not None and t_feat is not None:
                try:
                    alignment_loss = self.safe_feature_alignment(s_feat.float(), t_feat.float().detach())
                    loss += alignment_loss
                    count += 1
                except:
                    # 备用方案：简单的MSE
                    min_dim = min(s_feat.shape[1], t_feat.shape[1])
                    loss += F.mse_loss(s_feat[:, :min_dim].float(), t_feat[:, :min_dim].float().detach())
                    count += 1
        
        return loss / max(count, 1) if count > 0 else torch.tensor(0.0)
    
    def inter_modal_alignment(self, features, labels):
        """模态间对齐 - 平衡实现"""
        valid_features = [feat for feat in features if feat is not None]
        if len(valid_features) < 2:
            return torch.tensor(0.0)
        
        loss = 0.0
        count = 0
        
        # 对齐维度
        min_dim = min(feat.shape[1] for feat in valid_features)
        aligned_features = [feat[:, :min_dim].float() for feat in valid_features]
        
        # 创建标签相似度矩阵
        if len(labels.shape) > 1:
            label_sim = torch.sigmoid(torch.mm(labels.float(), labels.float().t()))
        else:
            label_sim = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        for i in range(len(aligned_features)):
            for j in range(i + 1, len(aligned_features)):
                feat_i = F.normalize(aligned_features[i], dim=1)
                feat_j = F.normalize(aligned_features[j], dim=1)
                
                similarity = torch.mm(feat_i, feat_j.t())
                
                # 平衡的对比损失
                positive_pairs = label_sim > 0.5
                negative_pairs = label_sim < 0.5
                
                pos_loss = torch.where(positive_pairs, self.margin - similarity, torch.zeros_like(similarity))
                neg_loss = torch.where(negative_pairs, similarity - (1 - self.margin), torch.zeros_like(similarity))
                
                loss += (pos_loss.mean() + neg_loss.mean()) * 0.5
                count += 1
        
        return loss / max(count, 1) if count > 0 else torch.tensor(0.0)
    
    def forward(self, student_features, teacher_features, labels):
        intra_loss = self.intra_modal_alignment(student_features, teacher_features)
        inter_loss = self.inter_modal_alignment(student_features, labels)
        
        return intra_loss * 0.6 + inter_loss * 0.4  # 加权平衡


class BalancedKnowledgeDistillation(nn.Module):
    """平衡的知识蒸馏"""
    def __init__(self, temperature=2.0):
        super(BalancedKnowledgeDistillation, self).__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits):
        """平衡的蒸馏损失"""
        # 软化概率分布
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL散度损失
        kl_loss = self.kl_loss(student_probs.log(), teacher_probs.detach())
        
        return kl_loss * (self.temperature ** 2)  # 温度缩放补偿


class ClientFAED(ClientFedAvg):
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
        
        # 平衡的FAED模块配置
        self.num_modalities = 2
        self.use_faed = getattr(args, 'use_faed', True)
        
        if self.use_faed and self.args.modality == "multimodal":
            # 平衡的模块初始化
            self.ama_module = BalancedAdaptiveModalityAlignment(
                num_modalities=self.num_modalities,
                temperature=getattr(args, 'ama_temperature', 0.1)
            ).to(device)
            
            self.ccr_module = BalancedCrossModalContrastive(
                margin=getattr(args, 'ccr_margin', 0.6),
                alpha=getattr(args, 'ccr_alpha', 0.3)
            ).to(device)
            
            self.kd_module = BalancedKnowledgeDistillation(
                temperature=getattr(args, 'kd_temperature', 2.0)
            )
            
            # 平衡的损失权重
            self.lambda_ccr = getattr(args, 'lambda_ccr', 0.15)  # 适中的权重
            self.lambda_ama = getattr(args, 'lambda_ama', 0.1)
            self.lambda_kd = getattr(args, 'lambda_kd', 0.1)
            
            # 教师模型
            self.teacher_model = copy.deepcopy(model)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.to(device)
            
            # 训练状态跟踪
            self.training_stage = 0  # 0: 初始, 1: 稳定, 2: 增强
    
    def get_parameters(self):
        return self.model.state_dict()
    
    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)
        if hasattr(self, 'teacher_model'):
            # 教师模型也更新，但保持EMA关系
            with torch.no_grad():
                for t_param, s_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                    t_param.data.copy_(s_param.data)
    
    def get_model_result(self):
        return self.result
    
    def get_test_true(self):
        return self.test_true
    
    def get_test_pred(self):
        return self.test_pred
    
    def get_train_groundtruth(self):
        return self.train_groundtruth

    def extract_balanced_features(self, x_a, x_b):
        """平衡的特征提取"""
        modality_features = []
        
        # 模态A
        if x_a is not None:
            x_a = x_a.float()
            if len(x_a.shape) > 2:
                # 多种特征提取方式的平衡
                feat_a = x_a.mean(dim=1)  # 时间维度平均
                if feat_a.shape[1] < 10:  # 如果维度太小，使用展平
                    feat_a = x_a.view(x_a.size(0), -1)
            else:
                feat_a = x_a
            modality_features.append(feat_a)
        else:
            modality_features.append(None)
            
        # 模态B
        if x_b is not None:
            x_b = x_b.float()
            if len(x_b.shape) > 2:
                feat_b = x_b.mean(dim=1)
                if feat_b.shape[1] < 10:
                    feat_b = x_b.view(x_b.size(0), -1)
            else:
                feat_b = x_b
            modality_features.append(feat_b)
        else:
            modality_features.append(None)
            
        return modality_features

    def update_training_stage(self, batch_idx, total_batches):
        """动态调整训练阶段"""
        progress = batch_idx / total_batches
        
        if progress < 0.3:
            self.training_stage = 0  # 初始阶段
        elif progress < 0.7:
            self.training_stage = 1  # 稳定阶段
        else:
            self.training_stage = 2  # 增强阶段

    def get_dynamic_weights(self):
        """动态调整损失权重"""
        if self.training_stage == 0:
            # 初始阶段：主要关注分类损失
            return 0.05, 0.05, 0.05
        elif self.training_stage == 1:
            # 稳定阶段：平衡使用所有模块
            return self.lambda_ccr, self.lambda_ama, self.lambda_kd
        else:
            # 增强阶段：适度加强正则化
            return self.lambda_ccr * 1.2, self.lambda_ama * 1.2, self.lambda_kd * 1.1

    def update_weights(self):
        self.model.train()
        if hasattr(self, 'ama_module'):
            self.ama_module.train()

        self.eval = EvalMetric(self.multilabel)
        total_batches = len(self.dataloader)
        
        # 优化器配置
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
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: 
                    continue
                
                # 更新训练阶段
                self.update_training_stage(batch_idx, total_batches)
                
                optimizer.zero_grad()
                
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    # 主要前向传播
                    outputs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                    
                    # 主要分类损失
                    if not self.multilabel: 
                        log_outputs = torch.log_softmax(outputs, dim=1)
                    else:
                        log_outputs = outputs
                    
                    cls_loss = self.criterion(log_outputs, y)
                    total_loss = cls_loss
                    
                    # FAED增强（在所有阶段都使用，但权重不同）
                    if self.use_faed:
                        try:
                            modality_mask = torch.ones(y.size(0), self.num_modalities, 
                                                     device=self.device, dtype=torch.float32)
                            
                            # 提取特征
                            student_features = self.extract_balanced_features(x_a, x_b)
                            
                            with torch.no_grad():
                                teacher_features = self.extract_balanced_features(x_a, x_b)
                                teacher_outputs, _ = self.teacher_model(x_a.float(), x_b.float(), l_a, l_b)
                            
                            # AMA特征增强
                            enhanced_features, availability_dist = self.ama_module(
                                student_features, modality_mask
                            )
                            
                            # CCR对比学习
                            ccr_loss = self.ccr_module(student_features, teacher_features, y)
                            
                            # KD知识蒸馏
                            kd_loss = self.kd_module(outputs, teacher_outputs)
                            
                            # 动态权重
                            lambda_ccr, lambda_ama, lambda_kd = self.get_dynamic_weights()
                            
                            # 平衡的总损失
                            total_loss = (cls_loss + 
                                        lambda_ccr * ccr_loss + 
                                        lambda_kd * kd_loss)
                            
                        except Exception as e:
                            # 优雅降级
                            print(f"FAED模块降级: {e}")
                            pass
                    
                else:
                    # 单模态情况
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    outputs, _ = self.model(x.float(), l)
                    
                    if not self.multilabel: 
                        outputs = torch.log_softmax(outputs, dim=1)
                    
                    cls_loss = self.criterion(outputs, y)
                    total_loss = cls_loss
                
                # 反向传播
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    8.0  # 适中的梯度裁剪
                )
                optimizer.step()
                
                # 更新教师模型（EMA）
                if hasattr(self, 'teacher_model'):
                    self._update_teacher_model()
                
                # 记录结果
                if not self.multilabel: 
                    self.eval.append_classification_results(y, outputs, cls_loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, cls_loss)
        
        # 最终结果
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
    
    def _update_teacher_model(self):
        """EMA更新教师模型"""
        alpha = 0.99
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), 
                                                  self.model.parameters()):
                teacher_param.data = alpha * teacher_param.data + (1 - alpha) * student_param.data


# 服务器端的平衡ABC机制
class BalancedAsynchronousCollaboration:
    """平衡的异步协作机制"""
    def __init__(self, buffer_size=5, decay_factor=0.95):
        self.buffer_size = buffer_size
        self.decay_factor = decay_factor
        self.knowledge_buffer = collections.deque(maxlen=buffer_size)
        
    def add_knowledge(self, knowledge, round_idx):
        """添加知识到缓冲区"""
        self.knowledge_buffer.append((knowledge, round_idx))
    
    def get_balanced_knowledge(self):
        """获取平衡的聚合知识"""
        if not self.knowledge_buffer:
            return None
            
        total_weight = 0
        balanced_knowledge = None
        
        for i, (knowledge, round_idx) in enumerate(self.knowledge_buffer):
            weight = math.pow(self.decay_factor, len(self.knowledge_buffer) - i - 1)
            
            if balanced_knowledge is None:
                balanced_knowledge = {k: v * weight for k, v in knowledge.items()}
            else:
                for k in balanced_knowledge:
                    if k in knowledge:
                        balanced_knowledge[k] += knowledge[k] * weight
            
            total_weight += weight
        
        # 归一化
        for k in balanced_knowledge:
            balanced_knowledge[k] /= total_weight
        
        return balanced_knowledge
