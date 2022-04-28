#coding: utf-8
import torch
from utils import clamp

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGSM1(): #局部Embedding
    def __init__(self, model):
        self.model = model

    def attack(self, is_first_attack=False, delta=None):
        # emb_name这个参数要换成你模型中embedding的参数名
        if is_first_attack:
            delta.uniform_(-1.0, 1.0)
            delta.requires_grad = True
        else:
            grad = delta.grad.detach()
            norm = torch.norm(grad)
            if norm != 0 and not torch.isnan(norm):
                delta.data = clamp(delta + 0.3 * grad / norm, torch.tensor((-1.0)).cuda(), torch.tensor((1.0)).cuda())
                delta = delta.detach()
        return delta 
    
class FGSM2(): #全局Embedding
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., alpha = 0.3, emb_name='embedding', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.backup[name] = param.data.clone()
                    delta = torch.zeros_like(param.data).cuda()
                    delta.uniform_(-epsilon, epsilon)
                    param.data.add_(delta)
                else:
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        delta = clamp(0.3 * param.grad / norm, torch.tensor((-1.0)).cuda(), torch.tensor((1.0)).cuda())
                        param.data.add_(delta)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FreeAT2(): #全局Embedding
    def __init__(self, model):
        self.model = model

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = clamp(epsilon * param.grad / norm, torch.tensor((-epsilon)).cuda(), torch.tensor((epsilon)).cuda())
                    param.data.add_(r_at)

class FreeAT1(): #局部Embedding
    def __init__(self, model):
        self.model = model

    def attack(self, epsilon=1., delta=None):
        grad = delta.grad.detach()
        norm  = torch.norm(grad)
        if norm != 0 and not torch.isnan(norm):
            delta.data = clamp(delta + epsilon * grad / norm, torch.tensor((-epsilon)).cuda(), torch.tensor((epsilon)).cuda())
        return delta

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embedding', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]