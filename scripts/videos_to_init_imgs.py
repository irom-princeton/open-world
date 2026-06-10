# frames need to be 320 x 192

import os
import argparse
import cv2
import h5py
import yaml

TARGET_W = 320
TARGET_H = 192

def video_to_init_imgs(input_dir, instruction, suite, scene, task_type):
    videos = ["left_base_cam.mp4", "right_base_cam.mp4", "wrist_cam.mp4"]
    imgs = ["exterior_left.png", "exterior_right.png", "wrist.png"]

    output_parent_dir = os.path.join(input_dir, "init_frames")
    os.mkdir(output_parent_dir)

    for init_cond in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, init_cond)) or os.path.join(input_dir, init_cond) == output_parent_dir:
            continue
        output_dir = os.path.join(input_dir, "init_frames", init_cond)
        os.mkdir(output_dir)

        # get frames from videos
        for i in range(len(videos)):
            video_path = os.path.join(input_dir, init_cond, videos[i])
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"Error: Could not open video file at path {video_path}.")
                continue
            success, frame = video.read()
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            cv2.imwrite(f"{output_dir}/{imgs[i]}", frame)
        print("successfully extracted frames")

        # make yaml
        h5_path = os.path.join(input_dir, init_cond, "trajectory.h5")
        yaml_path = os.path.join(output_dir, "initialization.yaml")

        with h5py.File(h5_path, "r") as h5_file:
            state = [float(x) for x in h5_file["data"]["cartesian_position"][0]]
            gripper_pos = [float(x) for x in h5_file["data"]["gripper_position"][0]]
            state = state + gripper_pos
            gripper_pos = gripper_pos[0]
            joint_pos = [float(x) for x in h5_file["data"]["joint_position"][0]]
        
        data = {
            "initial_state": {
                "robot": {
                    "state_representation": "cartesian_position_with_gripper",
                    "state": state,
                    "joint_position": joint_pos,
                    "gripper_position": gripper_pos
                }
            },
            "instruction": instruction,
            "metadata": {
                "suite": suite,
                "scene": scene,
                "task_type": task_type,
                "state_length": 7
            }
        }
        
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
            print("successfully created initialization.yaml")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--instruction")
    parser.add_argument("--suite")
    parser.add_argument("--scene")
    parser.add_argument("--task_type")
    args = parser.parse_args()
    video_to_init_imgs(args.input_dir, args.instruction, args.suite, args.scene, args.task_type)

if __name__ == "__main__":
    main()