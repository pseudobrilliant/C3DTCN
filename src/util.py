import torch
import time
from urllib.request import urlretrieve
from progressbar import ProgressBar, Percentage, Bar

def download_url(url, path, progress=True):
    if progress:
        pbar = ProgressBar(widgets=[Percentage(), Bar()])
        pbar.update(100)
        def progress_update(count, blockSize, totalSize):
            val = max(0, min(int(count * blockSize * float(100.0 / totalSize)), 100))
            pbar.update(val)

        urlretrieve(url, path, reporthook=progress_update)
        time.sleep(0.5)
    else:
        urlretrieve(url, path)

def euclidean(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_config_list(config_option):
    if "," in config_option:
        return config_option.split(",")
    else:
        return [config_option]
