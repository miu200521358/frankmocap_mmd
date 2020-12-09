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
import matplotlib.pyplot as plt

############# input parameters  #############
from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from handmocap.hand_mocap_api import HandMocap
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer
from datetime import datetime

from bodymocap.body_bbox_detector import BodyPoseEstimator
from handmocap.hand_bbox_detector import HandBboxDetector
from bodymocap.constants import J49_FLIP_PERM, JOINT_NAMES
from integration.copy_and_paste import integration_copy_paste


def __filter_bbox_list(body_bbox_list, hand_bbox_list, single_person):
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    hand_bbox_list = [hand_bbox_list[i] for i in idx_big2small]

    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]
        hand_bbox_list = [hand_bbox_list[0], ]

    return body_bbox_list, hand_bbox_list


def run_regress(
    args, img_original_bgr, 
    body_bbox_list, hand_bbox_list, bbox_detector,
    body_mocap, hand_mocap
):
    cond1 = len(body_bbox_list) > 0 and len(hand_bbox_list) > 0
    cond2 = not args.frankmocap_fast_mode

    # use pre-computed bbox or use slow detection mode
    if cond1 or cond2:
        if not cond1 and cond2:
            # run detection only when bbox is not available
            body_pose_list, body_bbox_list, hand_bbox_list, _ = \
                bbox_detector.detect_hand_bbox(img_original_bgr.copy())
        else:
            print("Use pre-computed bounding boxes")
        assert len(body_bbox_list) == len(hand_bbox_list)

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        body_bbox_list, hand_bbox_list = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # hand & body pose regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(hand_bbox_list) == len(pred_hand_list)
        assert len(pred_hand_list) == len(pred_body_list)

    else:
        _, body_bbox_list = bbox_detector.detect_body_bbox(img_original_bgr.copy())

        if len(body_bbox_list) < 1: 
            return list(), list(), list()

        # sort the bbox using bbox size 
        # only keep on bbox if args.single_person is set
        hand_bbox_list = [None, ] * len(body_bbox_list)
        body_bbox_list, _ = __filter_bbox_list(
            body_bbox_list, hand_bbox_list, args.single_person)

        # body regression first 
        pred_body_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_body_list)

        # get hand bbox from body
        hand_bbox_list = body_mocap.get_hand_bboxes(pred_body_list, img_original_bgr.shape[:2])
        assert len(pred_body_list) == len(hand_bbox_list)

        # hand regression
        pred_hand_list = hand_mocap.regress(
            img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(pred_hand_list) 

    # integration by copy-and-paste
    integral_output_list = integration_copy_paste(
        pred_body_list, pred_hand_list, body_mocap.smpl, img_original_bgr.shape)
    
    return body_bbox_list, hand_bbox_list, integral_output_list


def run_frank_mocap(args, bbox_detector, body_mocap, hand_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)

    cur_frame = args.start_frame
    video_frame = 0
    while True:
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
        
        # bbox detection
        if not load_bbox:
            body_bbox_list, hand_bbox_list = list(), list()
        
        # regression (includes integration)
        body_bbox_list, hand_bbox_list, pred_output_list = run_regress(
            args, img_original_bgr, 
            body_bbox_list, hand_bbox_list, bbox_detector,
            body_mocap, hand_mocap)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    #     # visualization
    #     res_img = visualizer.visualize(
    #         img_original_bgr,
    #         pred_mesh_list = pred_mesh_list,
    #         body_bbox_list = body_bbox_list,
    #         hand_bbox_list = hand_bbox_list)

    #    # show result in the screen
    #     if not args.no_display:
    #         res_img = res_img.astype(np.uint8)
    #         ImShow(res_img)

    #     # save result image
    #     if args.out_dir is not None:
    #         demo_utils.save_res_img(args.out_dir, image_path, res_img)

        demo_type = 'frank'
        demo_utils.save_pred_to_pkl(
            args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)

        for pidx, pred_output in enumerate(pred_output_list):
            # 描画設定
            fig = plt.figure(figsize=(15,15),dpi=100)
            # 3DAxesを追加
            ax = fig.add_subplot(111, projection='3d')

            joints = pred_output['pred_joints_img']
            left_hand_joints = pred_output['pred_left_hand_joints_img']
            right_hand_joints = pred_output['pred_right_hand_joints_img']
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
                            
            plt.savefig(osp.join(args.out_dir, "frames", f"3d_{pidx:02d}_{cur_frame:05d}.jpg"))
            plt.close()

            img = Vis_Skeleton_2D_SPIN49(joints, image=img_original_bgr)
            cv2.imwrite(osp.join(args.out_dir, "frames", f"2d_{pidx:02d}_{cur_frame:05d}.png"), img)

            # json出力
            joint_dict = {}
            joint_dict["image"] = {"width": img_original_bgr.shape[1], "height": img_original_bgr.shape[0]}
            joint_dict["bbox"] = {"x": float(body_bbox_list[pidx][0]), "y": float(body_bbox_list[pidx][1]), \
                                  "width": float(body_bbox_list[pidx][2]), "height": float(body_bbox_list[pidx][3])}
            joint_dict["joints"] = {}

            for jidx, jname in enumerate(JOINT_NAMES):
                joint_dict["joints"][jname] = {'x': float(joints[J49_FLIP_PERM[jidx]][0]), 'y': float(joints[J49_FLIP_PERM[jidx]][1]), 'z': float(joints[J49_FLIP_PERM[jidx]][2])}

            for jidx, jname in enumerate(RIGHT_HAND_NAMES):
                joint_dict["joints"][jname] = {'x': float(right_hand_joints[jidx][0]), 'y': float(right_hand_joints[jidx][1]), 'z': float(right_hand_joints[jidx][2])}

            for jidx, jname in enumerate(LEFT_HAND_NAMES):
                joint_dict["joints"][jname] = {'x': float(left_hand_joints[jidx][0]), 'y': float(left_hand_joints[jidx][1]), 'z': float(left_hand_joints[jidx][2])}

            params_json_path = osp.join(args.out_dir, "frames", f"joints_{pidx:02d}_{cur_frame:05d}.json")

            with open(params_json_path, 'w') as f:
                json.dump(joint_dict, f, indent=4)

        print(f"Processed : {image_path}")

    # save images as a video
    if not args.no_video_out and input_type in ['video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()


CONNECTIONS = [
    # Right Thumb
    ['right_wrist', 'right_thumb1'],
    ['right_thumb1', 'right_thumb2'],
    ['right_thumb2', 'right_thumb3'],
    ['right_thumb3', 'right_thumb'],
    # Right Index
    ['right_wrist', 'right_index1'],
    ['right_index1', 'right_index2'],
    ['right_index2', 'right_index3'],
    ['right_index3', 'right_index'],
    # Right Middle
    ['right_wrist', 'right_middle1'],
    ['right_middle1', 'right_middle2'],
    ['right_middle2', 'right_middle3'],
    ['right_middle3', 'right_middle'],
    # Right Ring
    ['right_wrist', 'right_ring1'],
    ['right_ring1', 'right_ring2'],
    ['right_ring2', 'right_ring3'],
    ['right_ring3', 'right_ring'],
    # Right Pinky
    ['right_wrist', 'right_pinky1'],
    ['right_pinky1', 'right_pinky2'],
    ['right_pinky2', 'right_pinky3'],
    ['right_pinky3', 'right_pinky'],

    # Left Thumb
    ['left_wrist', 'left_thumb1'],
    ['left_thumb1', 'left_thumb2'],
    ['left_thumb2', 'left_thumb3'],
    ['left_thumb3', 'left_thumb'],
    # Left Index
    ['left_wrist', 'left_index1'],
    ['left_index1', 'left_index2'],
    ['left_index2', 'left_index3'],
    ['left_index3', 'left_index'],
    # Left Middle
    ['left_wrist', 'left_middle1'],
    ['left_middle1', 'left_middle2'],
    ['left_middle2', 'left_middle3'],
    ['left_middle3', 'left_middle'],
    # Left Ring
    ['left_wrist', 'left_ring1'],
    ['left_ring1', 'left_ring2'],
    ['left_ring2', 'left_ring3'],
    ['left_ring3', 'left_ring'],
    # Left Pinky
    ['left_wrist', 'left_pinky1'],
    ['left_pinky1', 'left_pinky2'],
    ['left_pinky2', 'left_pinky3'],
    ['left_pinky3', 'left_pinky'],
]


RIGHT_HAND_NAMES = [
    'right_wrist',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky']


LEFT_HAND_NAMES = [
    'left_wrist',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky']




def Vis_Skeleton_2D_SPIN49(pt2d, pt2d_visibility = None, image = None, color=None):
    #Draw via opencv
    if not isinstance(image, np.ndarray):#not image: #If no image is given, generate Blank image
        image = np.ones((1000,1000,3),np.uint8) *255

    radius = 4

    #Openpose25 in Spin Defition + SPIN global 24
    # 'OP Nose', 'OP Neck', 'OP RShoulder',           #0,1,2
    # 'OP RElbow', 'OP RWrist', 'OP LShoulder',       #3,4,5
    # 'OP LElbow', 'OP LWrist', 'OP MidHip',          #6, 7,8
    # 'OP RHip', 'OP RKnee', 'OP RAnkle',             #9,10,11
    # 'OP LHip', 'OP LKnee', 'OP LAnkle',             #12,13,14
    # 'OP REye', 'OP LEye', 'OP REar',                #15,16,17
    # 'OP LEar', 'OP LBigToe', 'OP LSmallToe',        #18,19,20
    # 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',  #21, 22, 23, 24  ##Total 25 joints  for openpose
    link_openpose = [  [8,1], [1,0] , [0,16] , [16,18] , [0,15], [15,17],
                [1,2],[2,3],[3,4],      #Right Arm
                [1,5], [5,6], [6,7],       #Left Arm
                [8,12], [12,13], [13,14], [14,19], [19,20], [20,21],    #Left Leg
                [8,9], [9,10], [10,11], [11,22], [22,23], [23,24]       #Right left
                ]

    link_spin24 =[  [14,16], [16,12], [12,17] , [17,18] ,
                [12,9],[9,10],[10,11],      #Right Arm
                [12,8], [8,7], [7,6],       #Left Arm
                [14,3], [3,4], [4,5],
                [14,2], [2,1], [1,0]]


    link_spin24 = np.array(link_spin24) + 25

    # bLeft = [ 1,1,1,1,0,0,
    #     0,0,0,
    #     1,1,1,
    #     1,1,1,1,1,1,
    #     0,0,0,0,0,0]
    bLeft = [ 0,0,0,0,
        1,1,1,
        0,0,0,
        1,1,1,
        0,0,0]



    # for i in np.arange( len(link) ):
    for k in np.arange( 25,len(pt2d) ):
        if color is not None:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, color,-1)
        else:
            if pt2d_visibility is None:
                cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)
            else:
                if pt2d_visibility[k]:
                    cv2.circle(image, (int(pt2d[k][0]), int(pt2d[k][1]) ), radius, (0,0,255),-1)

    # #Openpose joint drawn as blue
    for k in np.arange( len(link_openpose) ):
        parent = link_openpose[k][0]
        child = link_openpose[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (255,0,0)#BGR, Blue
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)


    #SPIN24 joint drawn as red
    for k in np.arange( len(link_spin24) ):
        parent = link_spin24[k][0]
        child = link_spin24[k][1]
        if color is not None:
            c = color
        else:
            if True:#bLeft[k]:
                c = (0,0,255)#BGR, RED
            else:
                c = (0,0,0)
        if pt2d_visibility is None:
            cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)
        else:
            if pt2d_visibility[parent] and pt2d_visibility[child]:
                cv2.line(image, (int(pt2d[parent][0]), int(pt2d[parent][1])), (int(pt2d[child][0]), int(pt2d[child][1])), c, radius - 2)

    return image


def main():
    args = DemoOptions().parse()
    args.use_smplx = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    hand_bbox_detector =  HandBboxDetector('third_view', device)
    
    #Set Mocap regressor
    body_mocap = BodyMocap(args.checkpoint_body_smplx, args.smpl_dir, device = device, use_smplx= True)
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # # Set Visualizer
    # if args.renderer_type in ['pytorch3d', 'opendr']:
    #     from renderer.screen_free_visualizer import Visualizer
    # else:
    #     from renderer.visualizer import Visualizer
    # visualizer = Visualizer(args.renderer_type)

    run_frank_mocap(args, hand_bbox_detector, body_mocap, hand_mocap, None)


if __name__ == '__main__':
    main()