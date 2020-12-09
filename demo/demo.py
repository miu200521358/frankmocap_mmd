# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from demo.demo_options import DemoOptions
from bodymocap.constants import J49_FLIP_PERM, JOINT_NAMES
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer

from renderer.viewer2D import ImShow

def run_body_mocap(args, body_bbox_detector, body_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    timer = Timer()
    while True:
        timer.tic()
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':    
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        if load_bbox:
            body_pose_list = None
        else:
            body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
                img_original_bgr)
        hand_bbox_list = [None, ] * len(body_bbox_list)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        #Sort the bbox using bbox size 
        # (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
        if args.single_person and len(body_bbox_list)>0:
            body_bbox_list = [body_bbox_list[0], ]       

        # Body Pose Regression
        pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        # pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # # visualization
        # res_img = visualizer.visualize(
        #     img_original_bgr,
        #     pred_mesh_list = pred_mesh_list, 
        #     body_bbox_list = body_bbox_list)

        # # show result in the screen
        # if not args.no_display:
        #     res_img = res_img.astype(np.uint8)
        #     ImShow(res_img)

        # # save result image
        # if args.out_dir is not None:
        #     demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'body'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

            for pidx, pred_output in enumerate(pred_output_list):
                # 描画設定
                fig = plt.figure(figsize=(15,15),dpi=100)
                # 3DAxesを追加
                ax = fig.add_subplot(111, projection='3d')

                joints = pred_output['pred_joints_img']
                min_joint = np.min(joints, axis=0)
                max_joint = np.max(joints, axis=0)

                # ジョイント出力                    
                ax.set_xlim3d(int(min_joint[0]), int(max_joint[0]))
                ax.set_ylim3d(int(min_joint[2]), int(max_joint[2]))
                ax.set_zlim3d(int(-max_joint[1]), int(-min_joint[1]))
                ax.set(xlabel='x', ylabel='y', zlabel='z')

                for jidx, (jsname, jename) in enumerate([('OP Nose', 'OP Neck'), ('OP Neck', 'OP RShoulder'), ('OP RShoulder', 'OP RElbow'), ('OP RElbow', 'OP RWrist'), \
                                                         ('OP Neck', 'OP LShoulder'), ('OP LShoulder', 'OP LElbow'), ('OP LElbow', 'OP LWrist'), \
                                                         ('OP MidHip', 'OP RHip'), ('OP RHip', 'OP RKnee'), ('OP RKnee', 'OP RAnkle'), ('OP RAnkle', 'OP RHeel'), ('OP RHeel', 'OP RSmallToe'), ('OP RHeel', 'OP RBigToe'), \
                                                         ('OP MidHip', 'OP LHip'), ('OP LHip', 'OP LKnee'), ('OP LKnee', 'OP LAnkle'), ('OP LAnkle', 'OP LHeel'), ('OP LHeel', 'OP LSmallToe'), ('OP LHeel', 'OP LBigToe'), \
                                                         ('OP Nose', 'OP REye'), ('OP REye', 'OP REar'), ('OP Nose', 'OP LEye'), ('OP LEye', 'OP LEar'), ('OP Neck', 'OP MidHip')]):
                    xs = [joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][0], joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][0]]
                    ys = [joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][2], joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][2]]
                    zs = [-joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][1], -joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][1]]

                    ax.plot3D(xs, ys, zs, marker="o", ms=2, c="#0000FF")


                for jidx, (jsname, jename) in enumerate([('Nose', 'Neck (LSP)'), ('Neck (LSP)', 'Right Shoulder'), ('Right Shoulder', 'Right Elbow'), ('Right Elbow', 'Right Wrist'), \
                                                         ('Neck (LSP)', 'Left Shoulder'), ('Left Shoulder', 'Left Elbow'), ('Left Elbow', 'Left Wrist'), \
                                                         ('Pelvis (MPII)', 'Right Hip'), ('Right Hip', 'Right Knee'), ('Right Knee', 'Right Ankle'), \
                                                         ('Pelvis (MPII)', 'Left Hip'), ('Left Hip', 'Left Knee'), ('Left Knee', 'Left Ankle'), \
                                                         ('Nose', 'Right Eye'), ('Right Eye', 'Right Ear'), ('Nose', 'Left Eye'), ('Left Eye', 'Left Ear'), \
                                                         ('Pelvis (MPII)', 'Thorax (MPII)'), ('Thorax (MPII)', 'Spine (H36M)'), ('Spine (H36M)', 'Jaw (H36M)'), ('Jaw (H36M)', 'Nose')]):
                    xs = [joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][0], joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][0]]
                    ys = [joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][2], joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][2]]
                    zs = [-joints[J49_FLIP_PERM[JOINT_NAMES.index(jsname)]][1], -joints[J49_FLIP_PERM[JOINT_NAMES.index(jename)]][1]]

                    ax.plot3D(xs, ys, zs, marker="o", ms=2, c="#FF0000")
                                
                plt.savefig(osp.join(args.out_dir, "frames", f"joints_{pidx:02d}_{cur_frame:05d}.jpg"))
                plt.close()

        timer.toc(bPrint=True,title="Time")
        print(f"Processed : {image_path}")

    #save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


def main():
    args = DemoOptions().parse()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()

    # Set mocap regressor
    use_smplx = args.use_smplx
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # # Set Visualizer
    # if args.renderer_type in ['pytorch3d', 'opendr']:
    #     from renderer.screen_free_visualizer import Visualizer
    # else:
    #     from renderer.visualizer import Visualizer
    # visualizer = Visualizer(args.renderer_type)
  
    run_body_mocap(args, body_bbox_detector, body_mocap, None)


if __name__ == '__main__':
    main()