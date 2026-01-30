import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import queue
import pickle
import os
from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation
from kitti_odometry import KittiEvalOdom, umeyama_alignment


def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file
    Args:
        poses {ndarray}: list with all 4x4 pose matrix
        sequence {str}: sequence of KITTI dataset
        save_dir {str}: path to save pose
    """
    # create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = os.path.join(save_dir, "{}.txt".format(sequence))
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            f.write(line)


def post_processing(pred_poses, args):

    if args["window_size"] == 2:
        pred_poses = pred_poses.squeeze(1)
        return np.asarray(pred_poses)

    num_batchs = pred_poses.shape[0]

    # get poses in overlaped frames
    q = queue.Queue(args["window_size"]-1)  # The max size is 5.
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx, :, :])
        idx = idx + 1

    while idx < num_batchs:
        # process first full queue
        if idx == (args["window_size"]-1):
            poses.append(q.queue[0][0, :])

            # implemented for specific case window_size = 3 and overlap = 2
            avg_pose = (q.queue[0][1, :] + q.queue[1][0, :])/2
            poses.append(avg_pose)

            if args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1]
                            [1, :] + q.queue[2][0, :])/3
                poses.append(avg_pose)

        elif idx < (num_batchs - 1):
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                avg_pose = (q.queue[0][1, :] + q.queue[1][0, :])/2
                poses.append(avg_pose)

            elif args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 3
                avg_pose = (q.queue[0][2, :] + q.queue[1]
                            [1, :] + q.queue[2][0, :])/3
                poses.append(avg_pose)

        # process last full queue (idx == num_batchs-1)
        else:
            if args["window_size"] == 3:
                # implemented for specific case window_size = 3 and overlap = 2
                poses.append(q.queue[1][1, :])

            elif args["window_size"] == 4:
                # implemented for specific case window_size = 4 and overlap = 2
                avg_pose = (q.queue[1][2, :] + q.queue[2][1, :])/2
                poses.append(avg_pose)
                poses.append(q.queue[2][2, :])

            idx = idx + 1

        # update queue
        if idx < (num_batchs-1):
            idx = idx + 1
            first = q.get()  # dequeue first element
            q.put(pred_poses[idx, :, :])

    return np.asarray(poses)


def recover_trajectory_and_poses(poses):

    predicted_poses = []
    # recover predicted trajectory
    predicted_trajectory = []
    for i in range(len(poses)-1):
        if i == 0:
            T = np.eye(4)

        angles = poses[i, :3]
        t = poses[i, 3:]

        # undo normalization
        mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        [x, y, z] = np.multiply(angles, std_angles) + mean_angles
        t = np.multiply(t, std_t) + mean_t
        R = np.asarray(euler_to_rotation(x, y, z, seq='zyx'))

        T_r = np.concatenate((np.concatenate([R, np.reshape(t, (3, 1))], axis=1), [
                             [0.0, 0.0, 0.0, 1.0]]), axis=0)
        T_abs = np.dot(T, T_r)
        T = T_abs

        predicted_poses.append(T)
        predicted_trajectory.append(T_abs[:3, 3])

    return predicted_poses, predicted_trajectory


def align_trajectory_7dof(pred_poses, gt_poses):
    """Align predicted poses to ground truth using 7DoF transformation."""
    # Convert poses to dictionaries
    poses_result = {i: pose for i, pose in enumerate(pred_poses)}
    poses_gt = {i: pose for i, pose in enumerate(gt_poses)}

    # Extract common frame indices
    common_length = min(len(poses_result), len(poses_gt))
    gt_frames = sorted(poses_gt.keys())[:common_length]
    pred_frames = sorted(poses_result.keys())[:common_length]

    # Extract xyz coordinates
    xyz_gt = np.array([poses_gt[i][:3, 3] for i in gt_frames]).T
    xyz_pred = np.array([poses_result[i][:3, 3] for i in pred_frames]).T

    # Compute 7DoF alignment
    R, t, scale = umeyama_alignment(xyz_pred, xyz_gt, with_scale=True)

    # Create transformation matrix
    align_transformation = np.eye(4)
    align_transformation[:3, :3] = R
    align_transformation[:3, 3] = t

    # Apply alignment to each predicted pose
    aligned_poses = []
    for pose in pred_poses:
        scaled_pose = np.copy(pose)
        scaled_pose[:3, 3] *= scale
        aligned_pose = align_transformation @ scaled_pose
        aligned_poses.append(aligned_pose)
    
    return aligned_poses

if __name__ == "__main__":
    ckpt_path = "/Users/epheriami/Downloads/4B/Code/SYDE 673/P/tsformer/checkpoints/Exp3"
    ckpt_name = "kitti_pruned_model"
    sequences = ["01", "03", "04", "05", "06", "07", "10"]

    # Load arguments and setup
    with open(os.path.join(ckpt_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
        args['checkpoint_path'] = "/Users/epheriami/Downloads/4B/Code/SYDE 673/P/tsformer/checkpoints/Exp3/global_unstruct_pruning"
        args['checkpoint'] = f"{ckpt_name}.pth"
    f.close()

    all_metrics = []

    for sequence in sequences:
        # Load predicted poses
        pred_path = os.path.join(args["checkpoint_path"], f"pred_poses_{sequence}.npy")
        pred_poses = np.load(pred_path)

        # Post-process and recover trajectory
        poses = post_processing(pred_poses, args)
        pred_poses, pred_trajectory = recover_trajectory_and_poses(poses)

        # Load ground truth poses
        kitti_eval = KittiEvalOdom()
        gt_poses_path = os.path.join("/Users/epheriami/Downloads/4B/Code/SYDE 673/P/data/poses", f"{sequence}.txt")
        gt_poses = kitti_eval.load_poses_from_txt(gt_poses_path)
        gt_poses_list = [gt_poses[i] for i in sorted(gt_poses.keys())]

        # Align using 7DoF
        aligned_pred_poses = align_trajectory_7dof(pred_poses, gt_poses_list)

        # Truncate to common length
        common_length = min(len(aligned_pred_poses), len(gt_poses_list))
        aligned_pred_poses = aligned_pred_poses[:common_length]
        gt_poses_list = gt_poses_list[:common_length]

        # Save aligned poses
        save_trajectory(aligned_pred_poses, sequence,
                        save_dir=os.path.join(args["checkpoint_path"], "pred_poses"))

        # Create dictionaries for evaluation
        poses_result = {i: pose for i, pose in enumerate(aligned_pred_poses)}
        gt_poses_common = {i: gt_poses_list[i] for i in range(common_length)}

        # Compute metrics
        seq_err = kitti_eval.calc_sequence_errors(gt_poses_common, poses_result)
        ave_t_err, ave_r_err = kitti_eval.compute_overall_err(seq_err)
        ate = kitti_eval.compute_ATE(gt_poses_common, poses_result)
        rpe_trans, rpe_rot = kitti_eval.compute_RPE(gt_poses_common, poses_result)

        # Append metrics
        all_metrics.append({
            'Sequence': sequence,
            'Average Translational Error (%)': ave_t_err * 100,
            'Average Rotational Error (deg/100m)': ave_r_err / np.pi * 180 * 100,
            'ATE (m)': ate,
            'RPE Translational (m)': rpe_trans,
            'RPE Rotational (deg)': rpe_rot * 180 / np.pi
        })

        # Plotting (existing code)
        gt_trajectory = np.array([pose[:3, 3] for pose in gt_poses_list])
        plt.figure()
        plt.plot([x[0] for x in pred_trajectory], [z[2] for z in pred_trajectory], "b")
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 2], "r")
        plt.grid()
        plt.title(f"VO - Seq {sequence}")
        plt.xlabel("Translation in x direction [m]")
        plt.ylabel("Translation in z direction [m]")
        plt.legend(["estimated", "ground truth"])
        plt.savefig(os.path.join(args["checkpoint_path"], "plots", f"pred_traj_{sequence}.png"))
        plt.close()

    # Save metrics to CSV
    df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args['checkpoint_path'], 'metrics.csv')
    df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
