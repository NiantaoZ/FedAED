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


class ClientFedBN(object):
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
        
        # FedBN: 识别批归一化层
        self.bn_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.bn_layers.append(name)
        
    def get_parameters(self):
        """FedBN: 只返回非批归一化层的参数"""
        params = {}
        for name, param in self.model.named_parameters():
            # 检查参数是否属于批归一化层
            is_bn_param = False
            for bn_layer in self.bn_layers:
                if name.startswith(bn_layer):
                    is_bn_param = True
                    break
            if not is_bn_param:
                params[name] = param.data.clone()
        return params
    
    def get_model_result(self):
        return self.result
    
    def get_test_true(self):
        return self.test_true
    
    def get_test_pred(self):
        return self.test_pred
    
    def get_train_groundtruth(self):
        return self.train_groundtruth

    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # optimizer
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
            
        # last global model
        last_global_model = copy.deepcopy(self.model)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
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
                
        # epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
