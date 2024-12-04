'''
Based on: https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss_adaptive_gamma.py

Implementation of Focal Loss with adaptive gamma.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gammas=[3.0], probs=[1], device=None, adaptive=False):
        super(FocalLoss, self).__init__()
        self.device = device
        self.adaptive = adaptive
        self.ps = probs
        self.gammas = gammas
        self.gamma_dic = {}
        if len(probs) != len(gammas):
            raise RuntimeError('Oops, make sure probs and gammas have the same size!!!')
        
        for key, value in zip(self.ps, self.gammas):
            self.gamma_dic[key] = value

    def get_gamma_list(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            if self.adaptive:
                # Choosing the gamma for the sample
                for key in sorted(self.gamma_dic.keys()):
                    if pt_sample < key or pt_sample == 1:
                        gamma_list.append(self.gamma_dic[key])
                        break
            else:
                gamma_list.append(self.gammas[0])
        return torch.tensor(gamma_list).to(self.device)

    def forward(self, input, target):
        # if input.dim()>2:
        #     input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        #     input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()
        gamma = self.get_gamma_list(pt)
        loss = -1 * (1-pt)**gamma * logpt
        return loss.mean()


