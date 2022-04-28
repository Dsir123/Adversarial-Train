# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import metrics
import time
from utils import get_time_dif, clamp
from adversarial import FGM, FGSM1, FGSM2, PGD, FreeAT1, FreeAT2
from tensorboardX import SummaryWriter

AdversarialMethods = {
    "free": FreeAT1,
    "free2": FreeAT2, 
    "pgd": PGD,
    "fgm": FGM,
    "fgsm": FGSM1,
    "fgsm2": FGSM2, #扰动计算方式不一样
    "base": "base"
}

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def fgm(config, model, trains, labels, adversarial_method, optimizer, scheduler):
    outputs = model(model.embedding(trains[0]))
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    adversarial_method.attack()
    outputs = model(model.embedding(trains[0]))
    loss = F.cross_entropy(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    adversarial_method.restore() # 恢复embedding参数
    scheduler.step()
    model.zero_grad()
    return loss, outputs

def fgsm(config, model, trains, labels, adversarial_method, optimizer, scheduler):
    delta = torch.zeros((trains[0].size()[0], trains[0].size()[1], config.embed), dtype=torch.float32).to("cuda")
    delta = adversarial_method.attack(is_first_attack=True, delta=delta)
    outputs = model(model.embedding(trains[0])+delta)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    delta = adversarial_method.attack(delta=delta)
    outputs = model(model.embedding(trains[0])+delta)
    loss = F.cross_entropy(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.zero_grad()
    return loss, outputs

def fgsm2(config, model, trains, labels, adversarial_method, optimizer, scheduler):
    adversarial_method.attack(is_first_attack=True)
    outputs = model(model.embedding(trains[0]))
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    adversarial_method.attack()
    outputs = model(model.embedding(trains[0]))
    loss = F.cross_entropy(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    adversarial_method.restore() # 恢复embedding参数
    scheduler.step()
    model.zero_grad()
    return loss, outputs

def pgd(config, model, trains, labels, adversarial_method, optimizer, scheduler):
    outputs = model(model.embedding(trains[0]))
    loss = F.cross_entropy(outputs, labels)
    loss.backward()

    adversarial_method.backup_grad()
    for t in range(config.attack_iters):
        adversarial_method.attack(is_first_attack=(t==0))
        if t != config.attack_iters - 1:
            model.zero_grad()
        else:
            adversarial_method.restore_grad()
        outputs = model(model.embedding(trains[0]))
        loss = F.cross_entropy(outputs, labels)
        loss.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    adversarial_method.restore() # 恢复embedding参数
    optimizer.step()
    scheduler.step()
    model.zero_grad()

    return loss, outputs

def free2(config, model, trains, labels, adversarial_method, optimizer, scheduler):
    for _ in range(config.minibatch_replays):
        outputs = model(model.embedding(trains[0]))
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adversarial_method.attack()
        scheduler.step()
        model.zero_grad()
    return loss, outputs

def free(config, model, trains, labels, adversarial_method, optimizer, scheduler, delta):
    for _ in range(config.minibatch_replays):
        outputs = model(model.embedding(trains[0])+delta[:trains[0].size(0), :trains[0].size(1), :])
        loss = F.cross_entropy(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        delta = adversarial_method.attack(delta=delta)
        #delta.grad.zero_()
        scheduler.step()
        model.zero_grad()
    return loss, outputs, delta

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=config.lr_max, momentum=config.momentum, weight_decay=config.weight_decay)

    lr_steps = config.num_epochs * len(train_iter)
    if config.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.lr_min, max_lr=config.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif config.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    adversarial_method = ""
    if config.adversarial != "base":
        adversarial_method = AdversarialMethods[config.adversarial](model)
    if config.adversarial == "free":
        delta = torch.zeros((config.batch_size, config.pad_size, config.embed)).to("cuda")
        delta.requires_grad = True

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            if config.adversarial == "fgsm":
                loss, outputs = fgsm(config, model, trains, labels, adversarial_method, optimizer, scheduler)
            elif config.adversarial == "fgsm2":
                loss, outputs = fgsm2(config, model, trains, labels, adversarial_method, optimizer, scheduler)
            elif config.adversarial == "fgm":
                loss, outputs = fgm(config, model, trains, labels, adversarial_method, optimizer, scheduler)
            elif config.adversarial == "pgd":
                loss, outputs = pgd(config, model, trains, labels, adversarial_method, optimizer, scheduler)
            elif config.adversarial == "free":
                loss, outputs, delta = free(config, model, trains, labels, adversarial_method, optimizer, scheduler, delta)
            elif config.adversarial == "free2":
                loss, outputs = free2(config, model, trains, labels, adversarial_method, optimizer, scheduler)
            else:
                outputs = model(model.embedding(trains[0]))
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)

                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(model.embedding(texts[0]))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)