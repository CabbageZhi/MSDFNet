import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int,
                 group_num:int = 16,
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.gamma      = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.beta       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps

    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class MSFI(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = F.softmax(self.gn.gamma, dim=0)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = w_gamma > self.gate_treshold
        noninfo_mask = w_gamma <= self.gate_treshold
        x_1 = info_mask * reweigts * x
        x_2 = noninfo_mask * reweigts * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class DFWE(nn.Module):
    def __init__(self, channel):
        """ Detail Emphasis Module """
        super(DFWE, self).__init__()
        
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=0),
                                   MSFI(channel),
                                   nn.BatchNorm2d(channel),
                                   nn.ReLU(True))
        
        self.global_path = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.ReLU(True),
                                         nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
                                         nn.Sigmoid())

    def forward(self, x):
	    """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : recalibrated feature + input feature
                attention: B X C X 1 X 1
        """
	    out = self.conv1(x)
	    attention = self.global_path(out)      
        
	    return out + out * attention.expand_as(out) 
