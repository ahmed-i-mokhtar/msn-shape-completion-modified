import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm

inputs = glob.glob("./*/input.pcd")
outputs = glob.glob("./*/output.pcd")
coarse_outputs = glob.glob("./*/coarse_output.pcd")
gt = glob.glob("./*/gt.pcd")

for i in range(len(inputs)):
    input = inputs[i]
    output = outputs[i]
    course_output = coarse_outputs[i]
    gt = gt[i]

    input_pcd = o3d.io.read_point_cloud(input)
    output_pcd = o3d.io.read_point_cloud(output)
    course_output_pcd = o3d.io.read_point_cloud(course_output)
    gt_pcd = o3d.io.read_point_cloud(gt)

    #show one by one
    o3d.visualization.draw_geometries([input_pcd])
    o3d.visualization.draw_geometries([course_output_pcd])
    o3d.visualization.draw_geometries([output_pcd])
    o3d.visualization.draw_geometries([gt_pcd])
    
    break
    