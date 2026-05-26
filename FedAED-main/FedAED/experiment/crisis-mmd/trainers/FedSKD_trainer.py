
import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
from torch.nn import functional as F

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric


class ClientFedSKD(object):
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
        self.model = model  # Personalized model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.num_class = num_class if num_class is not None else 10
        
        # FedSKD specific components
        self.circulated_models = {}  # Stores models from other clients
        self.current_teacher = None   # Current teacher model for knowledge distillation
        self.kd_loss_weights = {     # Multi-dimensional distillation loss weights
            'batch': 1.0,
            'pixel': 0.5, 
            'region': 0.3
        }
        
        # Initialize result containers
        self.result = None
        self.test_true = None
        self.test_pred = None
        self.train_groundtruth = None

    def receive_model(self, client_id, model):
        """Receive models from other clients for knowledge distillation"""
        self.circulated_models[client_id] = copy.deepcopy(model).to(self.device)

    def set_current_teacher(self, client_id):
        """Set the teacher model for current round"""
        self.current_teacher = self.circulated_models.get(client_id)

    def _batch_level_distill(self, student_logits, teacher_logits):
        """Batch-level knowledge distillation"""
        return F.kl_div(
            F.log_softmax(student_logits / self.args.temperature, dim=1),
            F.softmax(teacher_logits / self.args.temperature, dim=1),
            reduction='batchmean'
        ) * (self.args.temperature ** 2)

    def _pixel_level_distill(self, student_feats, teacher_feats):
        """Pixel/voxel-level knowledge distillation"""
        return F.mse_loss(student_feats, teacher_feats)

    def _region_level_distill(self, student_feats, teacher_feats, region_size=8):
        """Region-level knowledge distillation"""
        # Use average pooling to extract region features
        student_pool = F.avg_pool2d(student_feats, region_size)
        teacher_pool = F.avg_pool2d(teacher_feats, region_size)
        return F.mse_loss(student_pool, teacher_pool)

    def get_parameters(self):
        """Get model parameters"""
        return self.model.state_dict()
    
    def get_model_result(self):
        """Get evaluation results"""
        return self.result
    
    def get_test_true(self):
        """Get ground truth test labels"""
        return self.test_true
    
    def get_test_pred(self):
        """Get predicted test labels"""
        return self.test_pred
    
    def get_train_groundtruth(self):
        """Get training ground truth labels"""
        return self.train_groundtruth

    def update_weights(self):
        """Update model weights with local training and knowledge distillation"""
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
                
                # Data preparation
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device).float()
                    x_b = x_b.to(self.device).float()
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)
                    
                    # Student model forward pass
                    student_outputs, student_feats = self.model(x_a, x_b, l_a, l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device).float()
                    l = l.to(self.device)
                    y = y.to(self.device)
                    
                    student_outputs, student_feats = self.model(x, l)
                
                # Teacher model forward pass (if exists)
                teacher_outputs = teacher_feats = None
                if self.current_teacher is not None:
                    with torch.no_grad():
                        if self.args.modality == "multimodal":
                            teacher_outputs, teacher_feats = self.current_teacher(x_a, x_b, l_a, l_b)
                        else:
                            teacher_outputs, teacher_feats = self.current_teacher(x, l)
                
                # Calculate task loss
                if not self.multilabel:
                    student_outputs = F.log_softmax(student_outputs, dim=1)
                    task_loss = self.criterion(student_outputs, y)
                else:
                    task_loss = self.criterion(student_outputs, y)
                
                # Multi-dimensional knowledge distillation
                kd_loss = 0
                if teacher_outputs is not None:
                    # Batch-level distillation
                    kd_loss += self.kd_loss_weights['batch'] * self._batch_level_distill(
                        student_outputs, teacher_outputs)
                    
                    # Feature-level distillation
                    if teacher_feats is not None and student_feats is not None:
                        # Pixel/voxel-level
                        kd_loss += self.kd_loss_weights['pixel'] * self._pixel_level_distill(
                            student_feats, teacher_feats)
                        
                        # Region-level (if spatial features)
                        if len(student_feats.shape) == 4:  # [B,C,H,W]
                            kd_loss += self.kd_loss_weights['region'] * self._region_level_distill(
                                student_feats, teacher_feats)
                
                total_loss = task_loss + kd_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                
                # Save results (pass the tensor directly, let EvalMetric handle conversion)
                if not self.multilabel:
                    self.eval.append_classification_results(y, student_outputs, total_loss)
                else:
                    self.eval.append_multilabel_results(y, student_outputs, total_loss)
        
        # Calculate final results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()
