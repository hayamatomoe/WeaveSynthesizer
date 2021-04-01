import torch
import math
import functools
from . import functions as func
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime as dt
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
def getTime():
    return dt.now().strftime('%y-%m%d-%H%M-%S%f')
def hello():
    print('hello')
class Wav(object):
    def __init__(self, f):
        self.func = f
    def val(self, wd=1024, norm=True, smooth=False):
        t_ = torch.arange(0., wd, 1.)/ wd
        f = self.func
        if norm:
            t = self.normalize(f(t_))
        else:
            t = f(t_)
        if smooth:
            t =func.smoothstep(t)
        return t
    def normalize(self, t, scale=1.):
        t -= torch.min(t)
        t /= torch.max(t)*scale + (1.0 - scale)/2
        return t
    def plot(self, twill= False, wd=1024, smooth=False, norm=True, aspect=1.):
        t = self.val(wd=wd, norm=norm,smooth=smooth)
        if twill:
            t = self.twill()
        t = t.cpu().numpy()
        t_ = np.arange(0., 1., 1. / wd)
        plt.figure(figsize=(5*aspect, 5))
        plt.plot(t_,t)
        plt.show()
    def twill(self, wd=1024, ht=512, n=16):
        t = self.val(wd=wd)
        t_ = (torch.arange(0., wd, 1.)/ht) % 1.
        t -= t_
        t=torch.round(n * t) * ht / n + torch.arange(0, wd) % ht
        t %= ht
        return t.long()
    def sparse(self, wd=1024, ht=512, n=16):
        t = self.twill(wd=wd, ht=ht, n=n)
        t = torch.unsqueeze(t, 0)
        t_ = torch.arange(0, wd).long()
        t_ = torch.unsqueeze(t_, 0)
        t = torch.cat([t, t_], 0)
        ones = torch.ones(wd, dtype=torch.short)
        t = torch.sparse.LongTensor(t, ones, torch.Size([ht, wd])).to_dense()
        # zeros = torch.zeros(1, ht - t.shape[1], wd)
        # t=torch.cat([t,zeros], 1)
        t.bool()
        return Tensor(t.cuda())
    def fft(self, N=1024, time=1, norm=True, smooth=True, B=None, log=True, fMax=None):
        if fMax == None:
            fMax = int(N / 2)
        t = self.val(wd=N, norm=norm, smooth=smooth)
        t = t.cpu().numpy()
        F = np.fft.fft(t)
        freq = np.fft.fftfreq(N, d=time/N)
        Amp = np.abs(F / (N / 2))**2
        # plt.loglog(freq[1:int(N / 2)], Amp[1:int(N / 2)])
        if log:
            plt.loglog(freq[1:fMax], Amp[1:fMax])
        else:
            plt.plot(freq[1:fMax], Amp[1:fMax])
        if B != None:
            fInv = freq ** B
            if log:
                plt.loglog(freq[1:fMax], fInv[1:fMax])
            else:
                plt.plot(freq[1:fMax], fInv[1:fMax])
        # plt.loglog(freq[1:100], Amp[1:100])
        plt.show()
def densityplot(t, aspect=1., bar=False):
    plt.figure(figsize=(12.0, 12.0))
    plt.imshow(t.cpu().numpy(), aspect=aspect)
    if bar:
        plt.colorbar()
def density(w_, w, a=1, levels=None):
    t = density_(w_, w, a)
    if levels != None:
        t = torch.logical_and((t > levels[0]), (t < levels[1]))
    return t.float()
def density_(w_, w, a=1):
    w = torch.unsqueeze(w.val(), 0)
    w = w.repeat(w.shape[1], 1)
    w_ = torch.unsqueeze(w_.val(), 1)
    w_ = w_.repeat(1, w_.shape[0])
    t = w - a * w_ % 1.
    # return torch.abs(torch.fmod(2.*t+2., 2.)-torch.ones(t.shape))
    return torch.abs(t)
def colorize(t, color):
    tdim = list(t.shape)
    tdim.append(3)
    t_=torch.unsqueeze(t,dim=2)
    return torch.lerp(color[0].expand(tdim), color[1].expand(tdim), t_)


class Seq(object):
    def __init__(self, t):
        if torch.is_tensor(t):
            self.tensor = t.short()
        else:
            self.tensor = torch.tensor(t, device='cpu', dtype=torch.short)
        while len(self.tensor.shape) < 2:
            self.tensor = torch.unsqueeze(self.tensor, 0).short()
    def copy(self):
        return Seq(self.tensor)
    def append(self, t):
        self.tensor = self.append_(self.tensor, t.tensor)
    def append_(self, t, t_):
        if t.shape[-1] == 0:
            out = t_
        else:
            d = t.shape[-1]
            d_ = t_.shape[-1]
            dMax = max(d, d_)
            dMin = min(d, d_)
            a = F.pad(t, (0, dMax - d), "constant", 0)
            b = F.pad(t_, (0, dMax - d_), "constant", 0)
            out = torch.cat([a, b], dim=0)
        return out
    def dotSeq_(self, i, lng=9):
        nlist = []
        for j in list(i):
            if j < lng:
                nlist.append(j)
            else:
                nlist += [math.ceil(j / 2),
                    1,
                    j - math.ceil(j / 2) - 1]
        lay = torch.tensor(nlist, dtype=torch.short)
        return lay
    def dotSeq(self):
        self.tensor = self.substitute(self.dotSeq_)
    def toBinTensor_(self, i, wd=512):
        elm = []
        for ind, j in enumerate(list(i)):
            if ind % 2 == 0:
                b = 0
            else:
                b = 1
            elm += [b] * int(j)
        nlist = elm * math.ceil(wd / len(elm))
        lay = torch.tensor(nlist, dtype=torch.bool)
        return lay
    def toBinTensor(self, wd=512):
        func=functools.partial(self.toBinTensor_, wd=wd)
        t = self.substitute(func)
        t = torch.unsqueeze(t, dim=1)
        t = t[...,:wd]
        return Tensor(t)
    def substitute(self, func):
        # make dot
        t = torch.tensor([[]], dtype=torch.short)
        for i in self.tensor:
            lay = func(i)
            lay = lay.reshape([1, -1])
            t = self.append_(t, lay)
        return t
    def plot(self, wd = 512, save = False, color = None, name=None):
        v = self.toBinTensor(wd=wd)
        v.plot(aspect=50., save=save, color = color, wd=wd, name=name)
    def roll(self, rot, diff=[0,1], init= 0, wd=512, ht=None, sq=True):
        t = self.tensor
        if torch.is_tensor(rot):
            rot = list(rot)
        if ht == None:
            ht = len(rot)
        if sq:
            ht = wd
        t_ = []
        for i in t:
            lay_ = []
            for j in range(ht):
                shiftNum = init + j * rot[(j - 1) %
                                          len(rot)] + (j* diff[0]) % diff[1]
                row = self.toBinTensor_(i, wd=wd)
                row = row.roll(shiftNum, dims=0)
                row = row[:wd]
                row = row.reshape([1, -1])
                lay_.append(row)
            lay = torch.cat(lay_, dim=0)
            lay = torch.unsqueeze(lay, dim=0)
            t_.append(lay)
        t = torch.cat(t_, dim=0).bool()
        return Tensor(t.to(device = 'cuda'))
def getTime():
    return dt.now().strftime('%y-%m%d-%H%M-%S%f')
class Tensor(object):
    def __init__(self, t):
        if torch.is_tensor(t):
            self.tensor = t.bool()
        else:
            self.tensor = torch.tensor(t, dtype=torch.bool)
        while len(self.tensor.shape) < 3:
            self.tensor = torch.unsqueeze(self.tensor, 0).bool()
    def copy(self):
        return Tensor(self.tensor)
    def append(self, t):
        self.tensor = torch.cat([self.tensor,t.tensor], dim=0)
    def reshape(self, wd):
        t = self.tensor
        if wd != None:
            s = t.shape[-1]
            if s > wd:
                t = t[..., :wd]
            elif s < wd:
                t = F.pad(t, (0, wd - s))
        return t
    def htJoint(self):
        t = self.tensor
        tdim = list(t.shape)
        t_ = [i for i in t]
        t = torch.cat(t_, dim=1).flatten().view(1, tdim[0] * tdim[1], tdim[2])
        return Tensor(t)
    def wdJoint(self, n=2):
        t = self.htJoint().tensor
        tdim = list(t.shape)
        t = t.flatten().view(tdim[1]//n, n, tdim[2])
        t = torch.transpose(t, 1, 2).flatten(start_dim=1)
        return Tensor(t)
    def ones(self):
        t = torch.ones(self.tensor.shape, dtype=torch.bool, device='cuda')
        return Tensor(t)
    def zeros(self):
        t = torch.zeros(self.tensor.shape, dtype=torch.bool, device='cuda')
        return Tensor(t)
    def plot(self, wd=None, aspect=1.0, save=False, color=None, sum=False, name=None, bmp=False):
        t = self.reshape(wd)
        if color != None:
            t = self.colorize(t, color=color)
            tdim = list(t.shape)
            if sum:
                sum_ = torch.zeros(tdim[1], tdim[2], tdim[3], dtype=torch.uint8, device='cuda')
                for i in t:
                    sum_ = sum_ + i
                t = torch.unsqueeze(sum_, 0)
            else:
                t_ = [i for i in t]
                t = torch.cat(t_, dim=1)
                t = t.reshape(1, tdim[0] * tdim[1], tdim[2], tdim[3])
        t = t.cpu().numpy()
        if save:
            timeStamp = getTime() + '_'
            if color != None:
                timeStamp += 'color_'
            if name != None:
                timeStamp += name
                timeStamp += '_'
        for i, lay in enumerate(t):
            plt.imshow(lay, aspect=aspect)
            plt.figure(figsize=(20.0, 20.0))
            plt.show()
            if save:
                if color == None:
                    img = Image.fromarray(lay.astype(np.bool))
                else:
                    img = Image.fromarray(lay.astype(np.uint8))
                if bmp:
                    img.save('img/' + timeStamp + str(i) + '.bmp')
                else:
                    img.save('img/' + timeStamp + str(i) + '.png')
                print(timeStamp)        
    def diag(self, wd, n=2):
        a = torch.ones([wd])
        b = torch.zeros([wd, wd])
        for i in range(-n, n+1):
            b += torch.diag(torch.ones([wd - abs(i)]), i)
        return b.cuda()
    def countNbd(self, nbd=2):
        nbdt = []
        for i in self.tensor:
            x = torch.matmul(self.diag(wd=i.shape[0], n=nbd), i.float())
            x = torch.matmul(x, self.diag(wd=i.shape[1], n=nbd))
            x = torch.unsqueeze(x, 0)
            nbdt.append(x.short())
        return torch.cat(nbdt, 0)
    def geNbd(self, thr=3, nbd=2):
        bound = (2*nbd+1)**2-thr
        return torch.ge(self.countNbd(nbd=nbd), bound)
    def leNbd(self, thr=3, nbd=2):
        bound = thr
        return torch.le(self.countNbd(nbd=nbd), bound)
    def satin(self, n, m, j):
        s = torch.tensor([n, m])
        s = s.repeat(self.tensor.shape[0], 1)
        s = Seq(s)
        t = s.roll([j], wd=self.tensor.shape[2], ht=self.tensor.shape[1], sq=False)
        return t
    def mask(self, t, nbd=2, low=3, high=3):
        h = self.geNbd(thr=high, nbd=nbd)
        l = self.leNbd(thr=low, nbd=nbd)
        out = torch.logical_xor(torch.logical_xor(self.tensor, h * t.tensor),l * t.tensor)
        return out
    def unisolate(self, nbd=3, low=1, high=1):
        a = self.countNbd(nbd=nbd)
        out = torch.logical_and(torch.logical_not(a <= low), self.tensor)
        out = torch.logical_or((a >= (2 * nbd + 1)** 2 - high), out)
        return out
    def colorize(self, t, color=None):
        if color == None or len(color) != len(t):
            color = torch.randint(
                0, 255, (len(t), 1, 1, 3), device='cuda')
        tdim = list(t.shape)
        tdim.append(3)
        color = color.expand(tdim)
        t = torch.unsqueeze(t, dim=3).expand(tdim)
        t = t.to(device='cuda')
        t = t*color
        return t

def matmul(a, b, c):
    x = torch.transpose(a.tensor, 1, 2)
    y = torch.matmul(x.float(), b.tensor.float())
    return Tensor(torch.matmul(y, c.tensor.float()))
def catTensor(t):
    t_ = [i.tensor for i in t]
    t = torch.cat(t_, dim=0)
    return Tensor(t)
def b_xor(t, t_):
    return Tensor(torch.bitwise_xor(t.tensor, t_.tensor))
def b_and(t, t_):
    return Tensor(torch.bitwise_and(t.tensor, t_.tensor))
def b_or(t, t_):
    return Tensor(torch.bitwise_or(t.tensor, t_.tensor))
def b_not(t):
    return Tensor(torch.bitwise_not(t.tensor.bool()))
def clamp(t):
    return Tensor(torch.clamp(t, 0, 1)) 
def mult(t, t_):
    return Tensor(t.tensor * t_.tensor)
def minus(t, t_):
    return t.tensor.short() - t_.tensor.short()
def plus(t, t_):
    return t.tensor.short() + t_.tensor.short()
