# Take reference from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import torch

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

    def get_state(self):
        return {self.name: self.avg}


class AccMeter(object):
    """Computes and stores the average and current values of accuracy"""
    def __init__(self, name, topk=(1,), fmt=':f'):
        self.name = name
        self.topk = topk
        self.maxk = max(topk)
        self.names = ["{name}@{k}".format(name=name, k=k) for k in self.topk]
        self.meters = {n: AverageMeter(n, fmt) for n in self.names}
        self.fmt = fmt
        for meter in self.meters.values():
            meter.reset()

    def get_meter(self, name):
        if name not in self.meters:
            raise RuntimeError("Can not find meter {}".format(name))
        return self.meters[name]

    def update(self, output, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = output.topk(self.maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for k, n in zip(self.topk, self.names):
                meter = self.meters[n]
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res = correct_k.mul_(100.0 / batch_size)
                meter.update(res.item(), batch_size)

    def __str__(self):
        return '\t'.join([str(meter) for meter in self.meters.values()])

    def get_state(self):
        state = {}
        for meter in self.meters.values():
            state.update(meter.get_state())
        return state


class ProgressMeter(object):
    def __init__(self, meters=None, prefix=""):
        self.epoch_fmtstr = 'Epoch: {:3d}'
        self.batch_fmtstr = 'Batch: {:5d}'
        self.meters = dict()
        if meters is not None:
            for meter in meters:
                add_meter(meter)
        self.prefix = prefix

    def add_meter(self, meter):
        if meter.name in self.meters:
            raise ValueError("Can not add meter name {} as it is already taken".format(meter.name))
        self.meters[meter.name] = meter

    def get_meter(self, name):
        if name not in self.meters:
            raise RuntimeError("Can not find meter {}".format(name))
        return self.meters[name]

    def update(self, name, val, n=1):
        if name not in self.meters:
            raise RuntimeError("Can not find meter {}".format(name))
        self.meters[name].update(val, n)

    def display(self, epoch, batch_idx=None, exclude=None):
        prefix = self.prefix + ' ' + self.epoch_fmtstr.format(epoch)
        if batch_idx is not None:
            prefix += ' ' + self.batch_fmtstr.format(batch_idx)
        if exclude is None:
            exclude = []
        entries = [prefix] + [str(m) for n, m in self.meters.items() if n not in exclude]
        print('\t'.join(entries))

    def get_state(self):
        state = {}
        for meter in self.meters.values():
            state.update(meter.get_state())
        return state
