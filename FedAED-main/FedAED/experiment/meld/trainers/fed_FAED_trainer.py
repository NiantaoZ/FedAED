import torch
import copy
import collections
from torch import nn
import torch.nn.functional as F
from .fed_avg_trainer import ClientFedAvg
from .optimizer import FedProxOptimizer
from .evaluation import EvalMetric
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# 设置日志
logger = logging.getLogger(__name__)

class DynamicReconfigurableDistillationGraph(nn.Module):
    """
    动态重构蒸馏图 - 适当优化版
    - 保持核心架构
    - 提升特征融合效果
    """

    def __init__(self, num_modalities=2):
        super().__init__()
        self.num_modalities = num_modalities
        # 适当改进权重初始化
        initial_weights = torch.eye(num_modalities) * 0.8 + torch.ones(num_modalities, num_modalities) * 0.1
        self.path_weights = nn.Parameter(initial_weights)
        
        # 添加模态重要性参数
        self.modality_importance = nn.Parameter(torch.ones(num_modalities))

    def forward(self, modality_features, modality_mask):
        """
        适当优化的前向传播
        """
        if not isinstance(modality_features, (list, tuple)):
            modality_features = [modality_features]

        B = modality_features[0].size(0)

        # 1. 改进的可用性评估
        availability_scores = []
        for i, feat in enumerate(modality_features):
            # 使用特征范数和方差综合评估
            norm_score = torch.norm(feat, p=2, dim=-1, keepdim=True)
            var_score = torch.std(feat, dim=-1, keepdim=True)
            # 结合模态重要性
            quality_score = 0.7 * norm_score + 0.3 * var_score
            importance = torch.sigmoid(self.modality_importance[i])
            final_score = quality_score * importance
            availability_scores.append(final_score)
        
        availability_scores = torch.cat(availability_scores, dim=-1)
        availability_scores = F.softmax(availability_scores, dim=-1) * modality_mask

        # 2. 路径矩阵构建（添加对称性）
        path_weights = self.path_weights
        symmetric_weights = (path_weights + path_weights.T) / 2.0
        normalized_weights = F.softmax(symmetric_weights, dim=-1)
        path_matrix = normalized_weights.unsqueeze(0) * availability_scores.unsqueeze(-1)

        # 3. 改进的跨模态融合
        aligned_features = []
        for target_idx in range(self.num_modalities):
            target_feat = modality_features[target_idx]
            
            fusion_contributions = []
            fusion_weights = []
            
            for source_idx in range(self.num_modalities):
                if target_idx != source_idx:
                    weight = path_matrix[:, source_idx, target_idx].unsqueeze(-1)
                    # 温和的权重限制
                    weight = torch.clamp(weight, min=0.2, max=0.8)
                    fusion_contributions.append(weight * modality_features[source_idx])
                    fusion_weights.append(weight)
            
            if fusion_contributions:
                # 加权融合
                total_contribution = torch.sum(torch.stack(fusion_contributions), dim=0)
                total_weight = torch.sum(torch.stack(fusion_weights), dim=0) + 1e-8
                
                fused_feature = total_contribution / total_weight
                # 门控融合机制
                gate = torch.sigmoid(total_weight * 2.5)
                aligned_feature = target_feat + gate * fused_feature
            else:
                aligned_feature = target_feat
            
            aligned_features.append(aligned_feature)

        return aligned_features, path_matrix


class CrossModalContrastiveRegularizer(nn.Module):
    """
    跨模态对比正则化 - 适当优化版
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_features, labels):
        """
        适当优化的对比损失
        """
        # 同模态对齐损失
        intra_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 特征归一化提升稳定性
            s_feat_norm = F.normalize(s_feat, p=2, dim=-1)
            t_feat_norm = F.normalize(t_feat, p=2, dim=-1)
            # 结合cosine和MSE损失
            cos_loss = 1.0 - F.cosine_similarity(s_feat_norm, t_feat_norm, dim=-1).mean()
            mse_loss = F.mse_loss(s_feat_norm, t_feat_norm)
            intra_loss += 0.7 * cos_loss + 0.3 * mse_loss
        
        intra_loss = intra_loss / len(student_features)

        # 跨模态对比损失
        inter_loss = 0.0
        num_modalities = len(student_features)
        batch_size = labels.size(0)
        
        if num_modalities > 1 and batch_size > 1:
            count = 0
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    # 特征归一化
                    feat_i = F.normalize(student_features[i], p=2, dim=-1)
                    feat_j = F.normalize(student_features[j], p=2, dim=-1)
                    
                    # 正样本相似度
                    pos_sim = F.cosine_similarity(feat_i, feat_j, dim=-1).mean()
                    
                    # 负样本采样
                    if batch_size > 2:
                        # 随机负样本
                        neg_indices = torch.randperm(batch_size, device=labels.device)
                        neg_sim = F.cosine_similarity(feat_i, feat_j[neg_indices], dim=-1).mean()
                        
                        # InfoNCE风格损失
                        numerator = torch.exp(pos_sim / self.temperature)
                        denominator = numerator + torch.exp(neg_sim / self.temperature)
                        inter_loss += -torch.log(numerator / denominator)
                        count += 1
            
            if count > 0:
                inter_loss = inter_loss / count

        # 平衡损失权重
        total_loss = intra_loss + 0.25 * inter_loss
        return total_loss


class ClientFAED(ClientFedAvg):
    """FAED客户端 - 适当优化版"""

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
        self.label_dict = label_dict
        self.num_class = num_class

        # 初始化FAED模块
        self.drdg = DynamicReconfigurableDistillationGraph(num_modalities=2).to(device)
        self.ccr = CrossModalContrastiveRegularizer(temperature=0.5).to(device)

        # 优化器设置
        self.drdg_optimizer = torch.optim.Adam(
            self.drdg.parameters(),
            lr=args.learning_rate * 0.15,
            weight_decay=1e-4
        )

    def update_weights(self):
        """适当优化的训练过程"""
        self.model.train()
        self.drdg.train()
        self.ccr.train()

        self.eval = EvalMetric(self.multilabel)

        # 主模型优化器
        optimizer = FedProxOptimizer(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
            mu=self.args.mu
        )

        last_global_model = copy.deepcopy(self.model)

        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue

                # 清零梯度
                self.model.zero_grad()
                optimizer.zero_grad()
                self.drdg_optimizer.zero_grad()

                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                    # 模型前向传播
                    outputs, features = self.model(x_a.float(), x_b.float(), l_a, l_b)

                    # 特征处理
                    if isinstance(features, torch.Tensor):
                        chunk_size = features.size(-1) // 2
                        modality_features = [
                            features[..., :chunk_size],
                            features[..., chunk_size:]
                        ]
                    else:
                        modality_features = features[:2]

                    # 蒸馏图处理
                    modality_mask = torch.stack([l_a, l_b], dim=1).float()
                    aligned_features, _ = self.drdg(modality_features, modality_mask)

                    # 教师特征
                    with torch.no_grad():
                        global_outputs, global_features = last_global_model(x_a.float(), x_b.float(), l_a, l_b)
                        if isinstance(global_features, torch.Tensor):
                            chunk_size = global_features.size(-1) // 2
                            global_modality_features = [
                                global_features[..., :chunk_size],
                                global_features[..., chunk_size:]
                            ]
                        else:
                            global_modality_features = global_features[:2]

                    # 对比损失
                    contrastive_loss = self.ccr(aligned_features, global_modality_features, y)

                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)

                    outputs, features = self.model(x.float(), l)
                    modality_features = [features]
                    modality_mask = l.unsqueeze(1).float()

                    aligned_features, _ = self.drdg(modality_features, modality_mask)

                    with torch.no_grad():
                        global_outputs, global_features = last_global_model(x.float(), l)
                        global_modality_features = [global_features]

                    contrastive_loss = self.ccr(aligned_features, global_modality_features, y)

                # 分类损失
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)

                cls_loss = self.criterion(outputs, y)

                # 动态损失权重
                contrastive_weight = 0.02  # 适当增加权重
                total_loss = cls_loss + contrastive_weight * contrastive_loss

                # 反向传播
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 6.0)
                torch.nn.utils.clip_grad_norm_(self.drdg.parameters(), 6.0)

                # 优化步骤
                optimizer.step()
                self.drdg_optimizer.step()

                # 评估
                if not self.multilabel:
                    self.eval.append_classification_results(y, outputs, cls_loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, cls_loss)

        # 返回结果
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

        return self.result
