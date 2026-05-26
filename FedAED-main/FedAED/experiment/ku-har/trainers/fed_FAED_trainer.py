import torch
import copy
import collections
from torch import nn
import torch.nn.functional as F
from .fed_avg_trainer import ClientFedAvg
from .optimizer import FedProxOptimizer
from .evaluation import EvalMetric
import numpy as np
import pdb


class DynamicReconfigurableDistillationGraph(nn.Module):
    """
    Lightweight DRDG: Dynamic Reconfigurable Distillation Graph
    - 不使用 feature_dim，也不需要 Linear 投影
    - 用 L2 范数做模态可用性估计
    - 保留模态间路径建图机制
    """

    def __init__(self, num_modalities=2):
        super().__init__()
        self.num_modalities = num_modalities

        # 模态间传递路径权重（可学习）
        self.path_weights = nn.Parameter(torch.ones(num_modalities, num_modalities))

    def forward(self, modality_features, modality_mask):
        """
        Args:
            modality_features: List of [B, D] tensors
            modality_mask: [B, M] 模态可用性 mask（0/1）
        Returns:
            aligned_features: List of aligned feature tensors
            path_matrix: [B, M, M] 模态蒸馏图
        """
        if not isinstance(modality_features, (list, tuple)):
            modality_features = [modality_features]

        B = modality_features[0].size(0)

        # 1. 用每模态特征范数计算可用性打分
        availability_scores = []
        for feat in modality_features:
            score = torch.norm(feat, p=2, dim=-1, keepdim=True)  # [B, 1]
            availability_scores.append(score)
        availability_scores = torch.cat(availability_scores, dim=-1)  # [B, M]
        availability_scores = F.softmax(availability_scores, dim=-1) * modality_mask  # [B, M]

        # 2. 构建路径图
        normalized_weights = F.softmax(self.path_weights, dim=-1)  # [M, M]
        path_matrix = normalized_weights * availability_scores.unsqueeze(-1)  # [B, M, M]

        # 3. 跨模态融合（不做投影）
        aligned_features = []
        for i in range(self.num_modalities):
            weighted_features = []
            for j in range(self.num_modalities):
                if i != j:
                    weight = path_matrix[:, j, i].unsqueeze(-1)  # [B, 1]
                    weighted_features.append(weight * modality_features[j])
            if weighted_features:
                cross_features = torch.sum(torch.stack(weighted_features), dim=0)
                aligned_feature = modality_features[i] + cross_features
            else:
                aligned_feature = modality_features[i]
            aligned_features.append(aligned_feature)

        return aligned_features, path_matrix



class CrossModalContrastiveRegularizer:
    """Cross-modal Contrastive Regularization (CCR) module"""
    # 功能：	通过对比学习加强模态间语义对齐，缓解异构问题
    # 服务器侧跨客户端非线性语义正则

    def __init__(self, temperature=0.5, margin=1):
        self.temperature = temperature #控制对比损失中的分布形状（
        self.margin = margin #控制正负对比之间的间隔。

    def __call__(self, student_features, teacher_features, labels):
        """
        Args:
            student_features: List of modality features from student [B, D]
            teacher_features: List of modality features from teacher [B, D]
            labels: Ground truth labels for contrastive learning
        Returns:
            contrastive_loss: Cross-modal contrastive loss
        """
        # 同模态（Student vs Teacher）对齐损失
        # 每个模态下的学生特征与教师特征做 MSE 对齐，保持语义一致性
        intra_loss = 0
        for s_feat, t_feat in zip(student_features, teacher_features):
            intra_loss += F.mse_loss(s_feat, t_feat)

        # Inter-modal alignment (cross-modality)
        inter_loss = 0
        num_modalities = len(student_features)

        # 不同模态之间的对比损失
        # 对于同一样本的两个不同模态，计算它们的 cosine 相似度（越高越好）
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Positive pairs (same sample, different modalities)
                pos_sim = F.cosine_similarity(
                    student_features[i], student_features[j], dim=-1
                ).mean()

                # Negative pairs (different samples)
                neg_sim = 0
                count = 0
                for k in range(len(labels)):
                    for l in range(len(labels)):
                        if labels[k] != labels[l]:
                            neg_sim += F.cosine_similarity(
                                student_features[i][k],
                                student_features[j][l],
                                dim=-1
                            )
                            count += 1
                if count > 0:
                    neg_sim = neg_sim / count
                # 对于不同标签的样本，作为负样本进行对比（应当相异）
                # 如果负样本比正样本还相似，则产生惩罚（margin-based对比）
                inter_loss += F.relu(self.margin - pos_sim + neg_sim)

        return intra_loss + inter_loss / (num_modalities * (num_modalities - 1) / 2)


class AsyncBidirectionalEvolution:
    """Asynchronous Bidirectional Evolution (ABC) module"""
    # 功能： 通过时间加权的双缓冲机制，提升知识演化稳定性
    # 提升全局学生模型蒸馏的鲁棒性

    def __init__(self, buffer_size=2, decay_factor=0.1):
        self.buffer_size = buffer_size # 维护过去多少轮的模型历史
        self.decay_factor = decay_factor # 越新的模型越重要
        self.teacher_buffer = collections.deque(maxlen=buffer_size)
        self.student_buffer = collections.deque(maxlen=buffer_size)

    #  更新教师模型的缓冲区
    def update_teacher_buffer(self, teacher_params):
        """Update teacher knowledge buffer"""
        self.teacher_buffer.append(teacher_params)
    #  更新学生模型的缓冲区
    def update_student_buffer(self, student_params):
        """Update student knowledge buffer"""
        self.student_buffer.append(student_params)

    # 从两个历史缓冲中获取演化后的教师知识（加权平均）
    # 对每一轮的历史参数做时间衰减平均，构成新的“演化教师”。
    def get_evolved_knowledge(self):
        """Combine knowledge from both buffers with time decay"""
        if not self.teacher_buffer or not self.student_buffer:
            return None

        # Time-weighted average of teacher knowledge
        teacher_knowledge = {}
        decay_weights = [self.decay_factor ** i for i in range(len(self.teacher_buffer))]
        total_weight = sum(decay_weights)

        for key in self.teacher_buffer[0].keys():
            teacher_knowledge[key] = sum(
                buf[key] * w for buf, w in zip(self.teacher_buffer, decay_weights)
            ) / total_weight

        return {'teacher': teacher_knowledge}


class ClientFAED(ClientFedAvg):
    """FAED client implementing all three modules"""

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

        # 初始化 FAED 模块

        # 用于对加速度计（acc）和陀螺仪（gyro）两模态进行特征对齐和蒸馏路径重构，适应模态可用性的动态变化
        # feature_dim = args.hid_size  # Using hidden size as feature dimension
        # 注意力机制下可能会改变特征维度
        self.drdg = DynamicReconfigurableDistillationGraph(
            num_modalities=2
        ).to(device)

        # 跨模态对比正则器
        # 强化不同模态之间的语义一致性，缓解语义漂移
        self.ccr = CrossModalContrastiveRegularizer(
            temperature=0.5,
            margin=2
        )
        # 异步双向演化机制
        # 维护历史知识缓冲池（教师），提升蒸馏过程稳定性并防止异常客户端污染全局模型。
        self.abc = AsyncBidirectionalEvolution(
            buffer_size=2,
            decay_factor=0.1
        )

        # DRDG优化器
        # 用更小学习率独立优化 DRDG 模块，避免扰动主模型。
        self.drdg_optimizer = torch.optim.Adam(
            self.drdg.parameters(),
            lr=args.learning_rate * 0.1  # Lower learning rate for distillation
        )

    def update_weights(self):
        # 开启训练模式，初始化评估器
        self.model.train()
        self.drdg.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)

        # optimizer
        optimizer = FedProxOptimizer(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
            mu=self.args.mu
        )

        # 获取当前的全局模型（作为 Student）
        # 用于蒸馏对比，即 当前轮客户端教师 vs 上一轮学生（全局模型）。
        last_global_model = copy.deepcopy(self.model)

        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue

                # Zero gradients
                self.model.zero_grad()
                optimizer.zero_grad()
                self.drdg_optimizer.zero_grad()

                if self.args.modality == "multimodal":
                    # x_a/x_b 代表 acc/gyro，l_a/l_b 是其可用性 mask，y 是标签
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)

                    # 模型前向传播
                    outputs, features = self.model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )

                    # 特征分离（acc + gyro），将融合后的特征按通道切分成 acc 和 gyro。
                    if isinstance(features, torch.Tensor):
                        # 如果特征连接在一起，则将其拆分为两种模态
                        chunk_size = features.size(-1) // 2
                        modality_features = [
                            features[..., :chunk_size],
                            features[..., chunk_size:]
                        ]
                    else:
                        modality_features = features[:2]  # 前两个是加速度计和陀螺仪功能

                    #  蒸馏图前向传播（DRDG）
                    modality_mask = torch.ones((x_a.size(0), 2), device=self.device)  # [B, 2]
                    # 应用 DRDG 进行跨模态知识对齐
                    # 在 DRDG 中感知可用模态并重构跨模态增强特征，输出对齐后的表示
                    aligned_features, _ = self.drdg(modality_features, modality_mask)

                    # 通过全局模型（学生）进行前向传递，用于构建“教师 vs 学生”的对比关系
                    with torch.no_grad():
                        global_outputs, global_features = last_global_model(
                            x_a.float(), x_b.float(), l_a, l_b
                        )
                        if isinstance(global_features, torch.Tensor):
                            chunk_size = global_features.size(-1) // 2
                            global_modality_features = [
                                global_features[..., :chunk_size],
                                global_features[..., chunk_size:]
                            ]
                        else:
                            global_modality_features = global_features[:2]

                    #  CCR计算对比损失
                    contrastive_loss = self.ccr(
                        aligned_features,
                        global_modality_features,
                        y
                    )


                else:
                    # 只用一个模态输入，仍然会应用 DRDG（退化为单模态图）和 CCR。
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)

                    outputs, features = self.model(x.float(), l)

                    # 对于单模态，我们仍然应用 DRDG
                    modality_features = [features]
                    modality_mask = torch.ones((x.size(0), 1), device=self.device)

                    aligned_features, _ = self.drdg(modality_features, modality_mask)

                    with torch.no_grad():
                        global_outputs, global_features = last_global_model(x.float(), l)
                        global_modality_features = [global_features]

                    contrastive_loss = self.ccr(
                        aligned_features,
                        global_modality_features,
                        y
                    )


                # Classification loss
                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)


                # 对于单模态，我们仍然应用 DRDG
                cls_loss = self.criterion(outputs, y)

                # 总损失整合并反向传播
                total_loss = cls_loss + 0.01 * contrastive_loss  # Weight could be tuned

                # Backward pass
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    10.0
                )
                torch.nn.utils.clip_grad_norm_(
                    self.drdg.parameters(),
                    10.0
                )

                # Optimization step
                optimizer.step()
                self.drdg_optimizer.step()

                # Save results
                if not self.multilabel:
                    self.eval.append_classification_results(
                        y,
                        outputs,
                        cls_loss  # Only track classification loss for metrics
                    )
                else:
                    self.eval.append_multilabel_results(
                        y,
                        outputs,
                        cls_loss
                    )

                # Update ABC buffer with teacher knowledge
                #teacher_params = {
                #    'features': [f.detach() for f in aligned_features],
                #    'outputs': outputs.detach()
                #}
                #self.abc.update_teacher_buffer(teacher_params)

        # Get evolved knowledge from ABC (could be used in server aggregation)
        #evolved_knowledge = self.abc.get_evolved_knowledge()

        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

        # Store additional FAED-specific information
        #self.result['faed_knowledge'] = evolved_knowledge
