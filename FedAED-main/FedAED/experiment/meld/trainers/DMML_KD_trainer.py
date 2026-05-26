
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


class ClientDMML_KD(object):
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

        # DMML-KD specific components
        self.generator = self._init_adaptive_generator()  # Changed to adaptive generator
        self.modality_balance_metrics = {'modality_a': 1.0, 'modality_b': 1.0}
        self.local_iterations = {'modality_a': args.local_epochs, 'modality_b': args.local_epochs}

        # Get modality-specific parameters
        self.modality_a_params = []
        self.modality_b_params = []
        for name, param in self.model.named_parameters():
            if 'modality_a' in name:
                self.modality_a_params.append(param)
            elif 'modality_b' in name:
                self.modality_b_params.append(param)

    def _init_adaptive_generator(self):
        """Initialize an adaptive generator that can handle varying input sizes"""
        return AdaptiveGenerator().to(self.device)

    def get_parameters(self, modality_common_only=False):
        if modality_common_only:
            return {name: param.clone() for name, param in self.model.named_parameters()
                    if 'common' in name or 'generator' in name}
        return self.model.state_dict()

    def update_weights(self, teacher_common_features=None, remaining_energy=100):
        self.model.train()
        self.eval = EvalMetric(self.multilabel)

        # Main optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=1e-5
        )

        # Modality-specific optimizers
        optimizers = {}
        if len(self.modality_a_params) > 0:
            optimizers['modality_a'] = torch.optim.SGD(
                self.modality_a_params,
                lr=self.args.learning_rate
            )
        if len(self.modality_b_params) > 0:
            optimizers['modality_b'] = torch.optim.SGD(
                self.modality_b_params,
                lr=self.args.learning_rate
            )

        prev_params = {n: p.clone() for n, p in self.model.named_parameters()}

        for epoch in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20:
                    continue

                # Prepare data
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a = x_a.to(self.device).float()
                    x_b = x_b.to(self.device).float()
                    l_a = l_a.to(self.device)
                    l_b = l_b.to(self.device)
                    y = y.to(self.device)

                    # Feature extraction
                    with torch.no_grad():
                        _, features = self.model(x_a, x_b, l_a, l_b)
                        common_features = self.generator(features)

                    # Forward pass
                    outputs, _ = self.model(x_a, x_b, l_a, l_b)
                else:
                    x, l, y = batch_data
                    x = x.to(self.device).float()
                    l = l.to(self.device)
                    y = y.to(self.device)

                    # Simplified processing for unimodal case
                    outputs, _ = self.model(x, l)
                    common_features = None

                if not self.multilabel:
                    outputs = torch.log_softmax(outputs, dim=1)

                # Calculate loss
                cls_loss = self.criterion(outputs, y)
                total_loss = cls_loss

                if teacher_common_features is not None and common_features is not None:
                    kd_loss = F.mse_loss(common_features, teacher_common_features)
                    total_loss += self.args.kd_weight * kd_loss

                # Backward pass
                optimizer.zero_grad()
                if 'modality_a' in optimizers:
                    optimizers['modality_a'].zero_grad()
                if 'modality_b' in optimizers:
                    optimizers['modality_b'].zero_grad()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

                # Update based on balance metrics
                if 'modality_a' in optimizers and epoch < self.local_iterations['modality_a']:
                    optimizers['modality_a'].step()
                if 'modality_b' in optimizers and epoch < self.local_iterations['modality_b']:
                    optimizers['modality_b'].step()

                optimizer.step()

                # Save results
                if not self.multilabel:
                    self.eval.append_classification_results(y, outputs, total_loss)
                else:
                    self.eval.append_multilabel_results(y, outputs, total_loss)

            # Update balance metrics
            current_params = {n: p.clone() for n, p in self.model.named_parameters()}
            for modality in ['modality_a', 'modality_b']:
                if modality in optimizers:
                    self.modality_balance_metrics[modality] = self._calculate_param_diff(
                        prev_params, current_params, modality)
            prev_params = current_params

        # Adjust iterations for next training
        self._adjust_iterations(remaining_energy)

        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()

        return common_features

    def _calculate_param_diff(self, prev_params, current_params, modality):
        """Calculate parameter differences"""
        diff = 0
        count = 0
        for name in prev_params:
            if modality in name:
                diff += torch.norm(prev_params[name] - current_params[name]).item()
                count += 1
        return diff / max(1, count)  # Avoid division by zero

    def _adjust_iterations(self, remaining_energy):
        """Adjust iterations for each modality"""
        total = sum(self.modality_balance_metrics.values())
        if total > 0:
            for modality in self.modality_balance_metrics:
                ratio = self.modality_balance_metrics[modality] / total
                self.local_iterations[modality] = min(
                    self.args.local_epochs,
                    max(1, int((1 - ratio) * self.args.local_epochs * remaining_energy / 100)))


class AdaptiveGenerator(nn.Module):
    """Adaptive generator that can handle varying input sizes"""
    def __init__(self, hidden_size=512, output_size=256):
        super(AdaptiveGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Dynamic layers that adapt to input size
        self.fc1 = nn.LazyLinear(hidden_size)  # Automatically adapts to input size
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Flatten input if necessary
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
