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


class ClientpFedMe(object):
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
        
        # 非常温和的 pFedMe 参数
        self.lamda = args.lamda if hasattr(args, 'lamda') else 1.0  # 极低的正则化
        self.K = args.K if hasattr(args, 'K') else 1  # 仅1次迭代
        self.beta = args.beta if hasattr(args, 'beta') else 0.0001  # 极低的学习率
        
        # 个性化模型（主要用于评估，训练影响很小）
        self.personalized_model = copy.deepcopy(self.model)
        self.personalized_model.to(self.device)
        
    def get_parameters(self):
        return self.model.state_dict()
    
    def get_personalized_parameters(self):
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
        """主要训练全局模型，轻微的个人化"""
        
        # 首先正常训练全局模型（这是主要的）
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
                    
                    outputs, _ = self.model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    outputs, _ = self.model(x.float(), l)
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                loss = self.criterion(outputs, y)

                if hasattr(optimizer, 'proximal_regularization'):
                    fedprox_loss = optimizer.proximal_regularization(last_global_model)
                    loss += fedprox_loss

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                
                if not self.multilabel: 
                    self.eval.append_classification_results(y, outputs, loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, loss)
        
        # 轻微的个人化步骤（可选，影响很小）
        self._mild_personalization()
        
        # 使用个性化模型进行评估（但训练主要影响全局模型）
        self._evaluate_personalized_model()
        
        return self.model.state_dict()
    
    def _mild_personalization(self):
        """非常轻微的个人化，几乎不影响主要训练"""
        if self.lamda <= 0 or self.beta <= 0:
            return
            
        # 只在少量数据上进行轻微调整
        self.personalized_model.load_state_dict(self.model.state_dict())
        self.personalized_model.train()
        
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        
        # 只在一个批次上进行轻微调整
        try:
            batch_data = next(iter(self.dataloader))
        except:
            return
            
        if self.args.modality == "multimodal":
            x_a, x_b, l_a, l_b, y = batch_data
            x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
            l_a, l_b = l_a.to(self.device), l_b.to(self.device)
            
            with torch.no_grad():
                global_outputs, _ = global_model(x_a.float(), x_b.float(), l_a, l_b)
            personalized_outputs, _ = self.personalized_model(x_a.float(), x_b.float(), l_a, l_b)
        else:
            x, l, y = batch_data
            x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
            
            with torch.no_grad():
                global_outputs, _ = global_model(x.float(), l)
            personalized_outputs, _ = self.personalized_model(x.float(), l)
        
        if not self.multilabel: 
            personalized_outputs = torch.log_softmax(personalized_outputs, dim=1)
        
        data_loss = self.criterion(personalized_outputs, y)
        
        # 极轻微的正则化
        reg_loss = 0.0
        for p_param, g_param in zip(self.personalized_model.parameters(), global_model.parameters()):
            reg_loss += torch.sum((p_param - g_param) ** 2)
        
        total_loss = data_loss + 0.1 * self.lamda * reg_loss  # 进一步降低影响
        
        self.personalized_model.zero_grad()
        total_loss.backward()
        
        # 极轻微的更新
        with torch.no_grad():
            for p_param, g_param in zip(self.personalized_model.parameters(), global_model.parameters()):
                if p_param.grad is not None:
                    p_param.data -= 0.01 * self.beta * p_param.grad  # 极小的更新
    
    def _evaluate_personalized_model(self):
        """使用个性化模型进行评估"""
        self.personalized_model.eval()
        personalized_eval = EvalMetric(self.multilabel)
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 15:
                    continue
                    
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    outputs, _ = self.personalized_model(x_a.float(), x_b.float(), l_a, l_b)
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    outputs, _ = self.personalized_model(x.float(), l)
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                
                loss = self.criterion(outputs, y)
                
                if not self.multilabel: 
                    personalized_eval.append_classification_results(y, outputs, loss)
                else:
                    personalized_eval.append_multilabel_results(y, outputs, loss)
        
        # 使用个性化模型的结果，但主要训练还是全局模型
        if not self.multilabel:
            self.result = personalized_eval.classification_summary()
        else:
            self.result = personalized_eval.multilabel_summary()
