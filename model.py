from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion
# from SoftPool import soft_pool1d, SoftPool1d
from module import farthest_point_sampling, index2point_converter
sys.path.append("./MDS/")
import MDS_module

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x



class MultiKernelConv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list of int): List of kernel sizes.
        """
        super(MultiKernelConv1DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
            for kernel_size in kernel_sizes
        ])
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length)
        Returns:
            torch.Tensor: Output tensor after applying multi-kernel convolutions.
        """
        # Apply each convolution kernel and store the results
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Sum the results element-wise (you can also concatenate if desired)
        out = sum(conv_outputs)  # Element-wise sum of the outputs
        
        return out
    
class SoftPool1d(nn.Module):
    def __init__(self):
        super(SoftPool1d, self).__init__()

    def forward(self, x):
        # Ensure input is 3D (batch_size, channels, num_points)
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
            no_batch = True
        else:
            no_batch = False

        B, C, D = x.size()

        # Apply softmax along the num_points dimension
        max_val, _ = torch.max(x, dim=-1, keepdim=True)
        x_exp = torch.exp(x - max_val)
        x_softmax = x_exp / torch.sum(x_exp, dim=-1, keepdim=True)
        
        # Weighted sum of input values according to softmax weights
        x_weighted_sum = torch.sum(x * x_softmax, dim=-1, keepdim=True)

        if no_batch:
            return x_weighted_sum.squeeze(0)
        return x_weighted_sum
    
class PointNetfeat(nn.Module):
    def __init__(self, num_points = 8192, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        
        self.soft_pool = SoftPool1d()

        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        x1 = F.relu(self.bn1(self.conv1(x))) # B x 256 x n
        x2 = F.relu(self.bn2(self.conv2(x1))) # B x 512 x n
        x3 = self.bn3(self.conv3(x2)) # B x 1024 x n
        
        # x1,_ = torch.max(x1, 2) # B x 256
        # x2,_ = torch.max(x2, 2) # B x 512
        # x3,_ = torch.max(x3, 2) # B x 1024
        
        
        #Modification B (Adding SoftPool instead of max pooling)
        #x_soft = soft_pool1d(x, x.size(2)) # B x 1024 x 1
        # x_soft = self.soft_pool(x)
        
        x1 = self.soft_pool(x1) # B x 256
        x2 = self.soft_pool(x2) # B x 512
        x3 = self.soft_pool(x3) # B x 1024
        
        #Print technical comparison between max pooling and soft pooling
        #TODO: Generate images with the two pooling methods for presentation.
        
        x1 = x1.view(-1, 256) #reshape tensor to 1D tensor B x 256
        x2 = x2.view(-1, 512) #reshape tensor to 1D tensor B x 512
        x3 = x3.view(-1, 1024) #reshape tensor to 1D tensor B x 1024
        return x3, x2, x1

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size + 2, self.bottleneck_size+2, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size + 2, self.bottleneck_size//2 + 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2 + 2, self.bottleneck_size//4 + 2, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4 + 2, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size + 2)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2 + 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4 + 2)

    def forward(self, x, x3, x2, x1):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))#B x 1026 x n
        x = x + x3
        x = F.relu(self.bn2(self.conv2(x)))#B x 514 x n
        x = x + x2
        x = F.relu(self.bn3(self.conv3(x)))#B x 258 x n
        x = x + x1
        x = self.th(self.conv4(x))
        return x

class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)


        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)
        x = torch.cat([x, pointfeat], 1)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x

class MSN(nn.Module):
    def __init__(self, num_points = 8192, bottleneck_size = 1024, n_primitives = 16):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.encoder = PointNetfeat(num_points, global_feat=True)
        self.encoder_linear = nn.Linear(1024, self.bottleneck_size)
        self.encoder_bn = nn.BatchNorm1d(self.bottleneck_size)
        self.encoder_relu = nn.ReLU()
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = self.bottleneck_size) for i in range(0,self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, x):
        partial = x # B x 3 x N
        x3, x2, x1 = self.encoder(x) # B x 1024
        x = self.encoder_relu(self.encoder_bn(self.encoder_linear(x3)))
        
        outs = []
        for i in range(0,self.n_primitives): #Loop on surface elements 16
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0),2,self.num_points//self.n_primitives)) # B x 2 x N/16
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            
            y3 = x3.unsqueeze(2).expand(x3.size(0),x3.size(1), rand_grid.size(2)).contiguous()
            y2 = x2.unsqueeze(2).expand(x2.size(0),x2.size(1), rand_grid.size(2)).contiguous()
            y1 = x1.unsqueeze(2).expand(x1.size(0),x1.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous() # B x 2+1024 x N/16 
            #y3 B x 1026 x N/16 y2 B x 514 x N/16 y1 B x 258 x N/16
            y3 = torch.cat((rand_grid, y3), 1).contiguous() # B x 2+1024 x N/16
            y2 = torch.cat((rand_grid, y2), 1).contiguous() # B x 2+512 x N/16
            y1 = torch.cat((rand_grid, y1), 1).contiguous() # B x 2+256 x N/16
            outs.append(self.decoder[i](y, y3, y2, y1)) 

        outs = torch.cat(outs,2).contiguous() # B x 3 x 4*N (Bx3x16*N/4)
        out1 = outs.transpose(1, 2).contiguous() # B x 4*N x 3
        
        dist, _, mean_mst_dis = self.expansion(out1, self.num_points//self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)

        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).cuda().contiguous() # B x 1 x 4*N
        outs = torch.cat( (outs, id0), 1) # B x 4 x 4*N
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous() # B x 1 x N
        partial = torch.cat( (partial, id1), 1) # B x 4 x N
        xx = torch.cat( (outs, partial), 2) # B x 4 x 5*N

        
        ## Condition on sampling method
        # FPS_indices = farthest_point_sampling(xx[:, 0:3, :], outs.shape[2])
        # xx = index2point_converter(xx, FPS_indices)
        resampled_idx = MDS_module.minimum_density_sample(xx[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1], mean_mst_dis) 
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :] 
        out2 = (xx + delta).transpose(2,1).contiguous()  
        return out1, out2, loss_mst