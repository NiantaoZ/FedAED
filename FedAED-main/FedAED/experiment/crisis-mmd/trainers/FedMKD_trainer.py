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


class ClientFedMKD(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model, 
        label_dict=None,
        num_class=None,
        global_logits=None,  # 添加全局logits
        client_id=None       # 添加客户端ID
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.global_logits = global_logits  # 存储全局知识
        self.client_id = client_id
        self.num_class = num_class
        
        # FedMKD参数
        self.temperature_teacher = getattr(args, 'temperature_teacher', 4.0)  # 教师模型温度
        self.temperature_student = getattr(args, 'temperature_student', 1.0)  # 学生模型温度
        self.alpha = getattr(args, 'alpha', 0.5)  # KD损失权重
        self.beta = getattr(args, 'beta', 0.3)   # CRKD损失权重
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth
        
    def get_local_logits(self):
        # 返回本地logits用于服务器聚合
        return self.local_logits
    
    def set_global_logits(self, global_logits):
        # 设置全局logits
        self.global_logits = global_logits

    def temperature_adaptive_knowledge_distillation(self, student_logits, teacher_logits, labels):
        """
        温度自适应知识蒸馏 (TAKD)
        - 为教师和学生模型提供不同的温度参数
        - 最大化知识转移
        """
        # 应用不同的温度参数
        teacher_softmax = torch.softmax(teacher_logits / self.temperature_teacher, dim=1)
        student_softmax = torch.softmax(student_logits / self.temperature_student, dim=1)
        
        # 计算KL散度损失
        kd_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(student_softmax + 1e-8),
            teacher_softmax.detach()
        ) * (self.temperature_student ** 2)
        
        return kd_loss

    def class_related_knowledge_distillation(self, student_logits, teacher_logits, labels):
        """
        类相关知识蒸馏 (CRKD)
        - 引入批次级样本相关性损失
        - 减少对特定样本或类的过度依赖
        - 提高模型对整体数据特征的理解
        """
        batch_size = student_logits.size(0)
        
        # 计算批次内样本相关性矩阵
        with torch.no_grad():
            # 教师模型的相关性矩阵
            teacher_similarity = torch.mm(
                teacher_logits, 
                teacher_logits.t()
            ) / torch.norm(teacher_logits, dim=1).unsqueeze(1) / torch.norm(teacher_logits, dim=1).unsqueeze(0)
        
        # 学生模型的相关性矩阵
        student_similarity = torch.mm(
            student_logits,
            student_logits.t()
        ) / torch.norm(student_logits, dim=1).unsqueeze(1) / torch.norm(student_logits, dim=1).unsqueeze(0)
        
        # 计算相关性损失 (MSE)
        correlation_loss = nn.MSELoss()(student_similarity, teacher_similarity)
        
        return correlation_loss

    def compute_fedmkd_loss(self, student_logits, teacher_logits, labels, task_loss):
        """
        计算FedMKD总损失
        """
        # 温度自适应知识蒸馏损失
        takd_loss = self.temperature_adaptive_knowledge_distillation(
            student_logits, teacher_logits, labels
        )
        
        # 类相关知识蒸馏损失
        crkd_loss = self.class_related_knowledge_distillation(
            student_logits, teacher_logits, labels
        )
        
        # 总损失 = 任务损失 + α * TAKD损失 + β * CRKD损失
        total_loss = task_loss + self.alpha * takd_loss + self.beta * crkd_loss
        
        return total_loss, takd_loss, crkd_loss

    def get_global_teacher_logits(self, batch_data):
        """
        根据批次数据获取全局教师logits
        使用CLIA（类粒度logits交互架构）
        """
        if self.global_logits is None:
            return None
            
        # 这里需要根据batch_data中的样本信息来匹配全局logits
        # 在实际实现中，需要根据数据索引或其他标识来匹配
        # 这里简化为返回一个适当形状的零张量
        batch_size = batch_data[0].size(0) if self.args.modality == "multimodal" else batch_data[0].size(0)
        
        if self.global_logits is not None and len(self.global_logits) > 0:
            # 在实际应用中，这里需要实现样本级别的logits匹配
            # 这里简化为返回平均logits或随机logits
            if isinstance(self.global_logits, list):
                # 如果是logits列表，取平均
                avg_logits = torch.mean(torch.stack(self.global_logits), dim=0)
                # 扩展到批次大小
                teacher_logits = avg_logits.unsqueeze(0).repeat(batch_size, 1)
            else:
                teacher_logits = self.global_logits[:batch_size]
            return teacher_logits.to(self.device)
        
        return None

    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # 存储本地logits用于服务器聚合
        self.local_logits = []
        
        # optimizer
        if self.args.fed_alg in ['fed_avg', 'fed_opt', 'fed_mkd']:
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

        # last global model
        last_global_model = copy.deepcopy(self.model)

        for iter in range(int(self.args.local_epochs)):
            epoch_logits = []
            
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
                    outputs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    # forward
                    outputs, _ = self.model(x.float(), l)
                
                # 存储本地logits
                epoch_logits.append(outputs.detach().cpu())
                
                if not self.multilabel: 
                    outputs_softmax = torch.log_softmax(outputs, dim=1)
                    task_loss = self.criterion(outputs_softmax, y)
                else:
                    task_loss = self.criterion(outputs, y)
                
                # FedMKD知识蒸馏
                teacher_logits = self.get_global_teacher_logits(batch_data)
                
                if teacher_logits is not None and teacher_logits.size(0) == outputs.size(0):
                    # 应用FedMKD损失
                    total_loss, takd_loss, crkd_loss = self.compute_fedmkd_loss(
                        outputs, teacher_logits, y, task_loss
                    )
                else:
                    # 如果没有教师logits，使用原始任务损失
                    total_loss = task_loss
                    takd_loss = torch.tensor(0.0)
                    crkd_loss = torch.tensor(0.0)
                
                # backward
                total_loss.backward()
                
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
                        total_loss  # 使用总损失进行评估记录
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        total_loss  # 使用总损失进行评估记录
                    )
            
            # 聚合本epoch的logits
            if epoch_logits:
                self.local_logits = torch.cat(epoch_logits, dim=0)

        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
            
        # 添加FedMKD特定的评估指标
        if hasattr(self, 'takd_loss'):
            self.result['takd_loss'] = takd_loss.item()
            self.result['crkd_loss'] = crkd_loss.item()
