import sys
import open3d as o3d
from logs.skip.model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import os
# import visdom
sys.path.append("./emd/")
import emd_module as emd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = './trained_model/network.pth',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 8192,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives in the atlas')
parser.add_argument('--env', type=str, default ="MSN_VAL"   ,  help='visdom environment') 

opt = parser.parse_args()
print (opt)

network = MSN(num_points = opt.num_points, n_primitives = opt.n_primitives) 
network.cuda()
network.apply(weights_init)

# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

network.eval()
with open(os.path.join('/home/stud/ahah/storage/slurm/cvai/3D-FUTURE/3D-FUTURE-model-val.txt')) as file:
    model_list = [line.strip() for line in file]

partial_dir = "/home/stud/ahah/storage/slurm/cvai/3D-FUTURE/partial"
gt_dir = "/home/stud/ahah/storage/slurm/cvai/3D-FUTURE/complete"
# vis = visdom.Visdom(port = 8097, env=opt.env) # set your port

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

EMD = emd.emdModule()

labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)
total_emd1 = 0
total_emd2 = 0
total_expansion_penalty = 0

brk = False
with torch.no_grad():
    for i, model in enumerate(model_list):
        print(model)
        partial = torch.zeros((20, 5000, 3), device='cuda')
        gt = torch.zeros((20, opt.num_points, 3), device='cuda')
        for j in range(20):
            pcd = o3d.io.read_point_cloud(os.path.join(partial_dir, model + '/' + str(j) + '.pcd'))
            if np.array(pcd.points).shape[0] == 0:
                print("Empty partial point cloud")
                brk = True
                break
            
            
            partial[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
            gt_pcd = o3d.io.read_point_cloud(os.path.join(gt_dir, "00000001-"+model + '.pcd'))
            if np.array(gt_pcd.points).shape[0] == 0:
                print("Empty complete point cloud")
                brk = True
                break
            gt[j, :, :] = torch.from_numpy(np.array(gt_pcd.points))
        if brk:
            continue
        output1, output2, expansion_penalty = network(partial.transpose(2,1).contiguous())
        dist, _ = EMD(output1, gt, 0.002, 10000)
        emd1 = torch.sqrt(dist).mean()
        dist, _ = EMD(output2, gt, 0.002, 10000)
        emd2 = torch.sqrt(dist).mean()
        idx = random.randint(0, 19)
        
        
        
        # vis.scatter(X = gt[idx].data.cpu(), win = 'GT',
        #             opts = dict(title = model, markersize = 2))
        # vis.scatter(X = partial[idx].data.cpu(), win = 'INPUT',
        #             opts = dict(title = model, markersize = 2))
        # vis.scatter(X = output1[idx].data.cpu(),
        #             Y = labels_generated_points[0:output1.size(1)],
        #             win = 'COARSE',
        #             opts = dict(title = model, markersize=2))
        # vis.scatter(X = output2[idx].data.cpu(),
        #             win = 'OUTPUT',
        #             opts = dict(title = model, markersize=2))
        print(opt.env + ' val [%d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd1.item(), emd2.item(), expansion_penalty.mean().item()))
        total_emd1 += emd1.item()
        total_emd2 += emd2.item()
        total_expansion_penalty += expansion_penalty.mean().item()
        avg_emd1 = total_emd1 / (i + 1)
        avg_emd2 = total_emd2 / (i + 1)
        avg_expansion_penalty = total_expansion_penalty / (i + 1)
        print('Average EMD1: %f, EMD2: %f, Expansion Penalty: %f' %(avg_emd1, avg_emd2, avg_expansion_penalty))
        
        
        #create folder in output directory for model
        if not os.path.exists('./output/skip/' + model):
            os.makedirs('./output/skip/' + model)
        
        output_dir = './output/skip/' + model + '/'
        #save to disk coarse point cloud, full output and gt point clouds using open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(output1[idx].data.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(output_dir, 'coarse_output.pcd'), pcd)
        pcd.points = o3d.utility.Vector3dVector(output2[idx].data.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(output_dir, 'output.pcd'), pcd)
        pcd.points = o3d.utility.Vector3dVector(gt[idx].data.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(output_dir, 'gt.pcd'), pcd)
        
        #save input partial point cloud
        pcd.points = o3d.utility.Vector3dVector(partial[idx].data.cpu().numpy())
        o3d.io.write_point_cloud(os.path.join(output_dir, 'input.pcd'), pcd)
        
        print('Saved output point clouds to disk')
        