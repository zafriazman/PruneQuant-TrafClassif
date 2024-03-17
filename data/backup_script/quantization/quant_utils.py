import os
import torch
from networks.cnn1d import CNN1D_TrafficClassification

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for rawpacket, target in data_loader:
            output = model(rawpacket)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], rawpacket.size(0))
            top5.update(acc5[0], rawpacket.size(0))
    print('')

    return top1, top5

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (kB):", os.path.getsize("temp.p")/1e3)
    os.remove("temp.p")


def load_fp32_model(input_ch, num_classes, device):
    net = CNN1D_TrafficClassification(input_ch=input_ch, num_classes=num_classes).to(device)
    checkpoint = torch.load(os.path.join('networks', 'pretrained_models', 'iscx2016vpn', 'best_model_without_aux.pth'), map_location=device)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    net.load_state_dict(sd, strict=True)
    return net