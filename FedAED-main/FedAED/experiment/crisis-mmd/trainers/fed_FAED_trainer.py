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
    动态重构蒸馏图 - 终极微调版
    """

    def __init__(self, num_modalities=2):
        super().__init__()
        self.num_modalities = num_modalities
        # 终极微调权重初始化
        initial_weights = torch.eye(num_modalities) * 0.875 + torch.ones(num_modalities, num_modalities) * 0.062
        self.path_weights = nn.Parameter(initial_weights)
        
        # 终极调整模态重要性参数
        self.modality_importance = nn.Parameter(torch.ones(num_modalities) * 1.07)

    def forward(self, modality_features, modality_mask):
        """
        终极微调的前向传播
        """
        if not isinstance(modality_features, (list, tuple)):
            modality_features = [modality_features]

        B = modality_features[0].size(0)

        # 1. 终极微调的可用性评估
        availability_scores = []
        for i, feat in enumerate(modality_features):
            # 终极调整特征评估
            norm_score = torch.norm(feat, p=2, dim=-1, keepdim=True)
            var_score = torch.std(feat, dim=-1, keepdim=True)
            # 终极调整权重分配
            quality_score = 0.735 * norm_score + 0.265 * var_score
            importance = torch.sigmoid(self.modality_importance[i] * 1.14)
            final_score = quality_score * importance
            availability_scores.append(final_score)
        
        availability_scores = torch.cat(availability_scores, dim=-1)
        availability_scores = F.softmax(availability_scores, dim=-1) * modality_mask

        # 2. 终极微调的路径矩阵构建
        path_weights = self.path_weights
        symmetric_weights = (path_weights + path_weights.T) / 2.0
        normalized_weights = F.softmax(symmetric_weights, dim=-1)
        path_matrix = normalized_weights.unsqueeze(0) * availability_scores.unsqueeze(-1)

        # 3. 终极微调的跨模态融合
        aligned_features = []
        for target_idx in range(self.num_modalities):
            target_feat = modality_features[target_idx]
            
            fusion_contributions = []
            fusion_weights = []
            
            for source_idx in range(self.num_modalities):
                if target_idx != source_idx:
                    weight = path_matrix[:, source_idx, target_idx].unsqueeze(-1)
                    # 终极调整权重限制
                    weight = torch.clamp(weight, min=0.235, max=0.765)
                    fusion_contributions.append(weight * modality_features[source_idx])
                    fusion_weights.append(weight)
            
            if fusion_contributions:
                total_contribution = torch.sum(torch.stack(fusion_contributions), dim=0)
                total_weight = torch.sum(torch.stack(fusion_weights), dim=0) + 1e-8
                
                fused_feature = total_contribution / total_weight
                # 终极优化门控机制
                gate = torch.sigmoid(total_weight * 2.68)
                aligned_feature = target_feat + gate * fused_feature
            else:
                aligned_feature = target_feat
            
            aligned_features.append(aligned_feature)

        return aligned_features, path_matrix


class CrossModalContrastiveRegularizer(nn.Module):
    """
    跨模态对比正则化 - 终极微调版
    """

    def __init__(self, temperature=0.465):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_features, teacher_features, labels):
        """
        终极微调的对比损失
        """
        # 同模态对齐损失 - 终极微调
        intra_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_feat_norm = F.normalize(s_feat, p=2, dim=-1)
            t_feat_norm = F.normalize(t_feat, p=2, dim=-1)
            # 终极调整损失权重
            cos_loss = 1.0 - F.cosine_similarity(s_feat_norm, t_feat_norm, dim=-1).mean()
            mse_loss = F.mse_loss(s_feat_norm, t_feat_norm) * 0.975
            intra_loss += 0.735 * cos_loss + 0.265 * mse_loss
        
        intra_loss = intra_loss / len(student_features)

        # 跨模态对比损失 - 终极增强
        inter_loss = 0.0
        num_modalities = len(student_features)
        batch_size = labels.size(0)
        
        if num_modalities > 1 and batch_size > 1:
            count = 0
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    feat_i = F.normalize(student_features[i], p=2, dim=-1)
                    feat_j = F.normalize(student_features[j], p=2, dim=-1)
                    
                    pos_sim = F.cosine_similarity(feat_i, feat_j, dim=-1).mean()
                    
                    if batch_size > 2:
                        neg_indices = torch.randperm(batch_size, device=labels.device)
                        neg_sim = F.cosine_similarity(feat_i, feat_j[neg_indices], dim=-1).mean()
                        
                        numerator = torch.exp(pos_sim / self.temperature)
                        denominator = numerator + torch.exp(neg_sim / self.temperature) + 1e-8
                        inter_loss += -torch.log(numerator / denominator)
                        count += 1
            
            if count > 0:
                inter_loss = inter_loss / count

        # 终极调整损失权重平衡
        total_loss = intra_loss + 0.272 * inter_loss
        return total_loss


class ClientFAED(ClientFedAvg):
    """FAED客户端 - 终极微调版"""

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

        # 终极微调FAED模块参数
        self.drdg = DynamicReconfigurableDistillationGraph(num_modalities=2).to(device)
        self.ccr = CrossModalContrastiveRegularizer(temperature=0.465).to(device)

        # 优化器设置终极微调
        self.drdg_optimizer = torch.optim.Adam(
            self.drdg.parameters(),
            lr=args.learning_rate * 0.167,  # 终极调整学习率
            weight_decay=8.2e-5  # 终极调整权重衰减
        )

    def update_weights(self):
        """终极微调的训练过程"""
        self.model.train()
        self.drdg.train()
        self.ccr.train()

        self.eval = EvalMetric(self.multilabel)

        # 主模型优化器参数终极微调
        optimizer = FedProxOptimizer(
            self.model.parameters(),
            lr=self.args.learning_rate * 1.028,  # 终极调整学习率
            momentum=0.918,  # 终极调整动量
            weight_decay=8.2e-6,  # 终极调整权重衰减
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

                    outputs, features = self.model(x_a.float(), x_b.float(), l_a, l_b)

                    if isinstance(features, torch.Tensor):
                        chunk_size = features.size(-1) // 2
                        modality_features = [
                            features[..., :chunk_size],
                            features[..., chunk_size:]
                        ]
                    else:
                        modality_features = features[:2]

                    modality_mask = torch.stack([l_a, l_b], dim=1).float()
                    aligned_features, _ = self.drdg(modality_features, modality_mask)

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

                # 终极调整损失权重
                contrastive_weight = 0.0255  # 终极提高对比损失权重
                total_loss = cls_loss + contrastive_weight * contrastive_loss

                # 反向传播
                total_loss.backward()

                # 终极调整梯度裁剪阈值
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.65)
                torch.nn.utils.clip_grad_norm_(self.drdg.parameters(), 5.65)

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
