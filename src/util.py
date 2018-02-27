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

