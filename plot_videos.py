import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import imageio.v2 as imageio
from PIL import Image
import math
import torch
import numpy as np
from dtw import dtw
from constants import PAD_TOKEN
from IPython.display import display, Image as IPyImage

# This is the format of the 3D data, outputted from the Inverse Kinematics model
def getSkeletalModelStructure():
    #print("Using skeleton 2")
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # head
        (0, 1, 0),

        # left shoulder
        (1, 2, 1),

        # left arm
        (2, 3, 2),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29, 3),

        # right shoulder
        (1, 5, 1),

        # right arm
        (5, 6, 2),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8, 3),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),

        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),

        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),

        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),

        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),

        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),

        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),

        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),

        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),

        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),

        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),

        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )


# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints, file_path, video_name, references=None, skip_frames=1, sequence_ID=None):
    os.makedirs(file_path, exist_ok=True)
    FPS = 25 // skip_frames
    video_file = os.path.join(file_path, f"{sequence_ID.split('.')[0]}.mp4")

    frames = []
    for j, frame_joints in enumerate(joints):
        if PAD_TOKEN in frame_joints.astype('str').tolist():
            continue

        frame_joints = frame_joints * 3
        pred_2d = np.reshape(frame_joints, (75, 3))[:, :2]

        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.set_xlim(0, 650)
        ax.set_ylim(0, 650)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        draw_frame_2D(ax, pred_2d)
        ax.text(180, 600, "Predicted Sign Pose", fontsize=12)

        if references is not None:
            ref_joints = references[j] * 3
            ref_2d = np.reshape(ref_joints, (75, 3))[:, :2]

            fig, axs = plt.subplots(1, 2, figsize=(13, 6.5))
            for axis in axs:
                axis.set_xlim(0, 650)
                axis.set_ylim(0, 650)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.invert_yaxis()

            draw_frame_2D(axs[0], pred_2d)
            axs[0].text(180, 600, "Predicted Sign Pose", fontsize=10)
            draw_frame_2D(axs[1], ref_2d)
            axs[1].text(180, 600, "Ground Truth Pose", fontsize=10)

            fig.suptitle(f"Sequence ID: {sequence_ID}", fontsize=12)
        
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = np.array(Image.open(buf).convert("RGB"))
        buf.close()
        h, w = image.shape[:2] 

        new_w = ((w + 15) // 16) * 16
        new_h = ((h + 15) // 16) * 16

        if (new_w != w or new_h != h):
            padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))  # white padding
            padded.paste(Image.fromarray(image), (0, 0))
            image = np.array(padded)
        
        frames.append(image)
        plt.close('all')
    
    os.makedirs(os.path.dirname(video_file), exist_ok=True)
    imageio.mimsave(video_file, frames, fps=FPS)
    
    with io.BytesIO() as buffer:
        imageio.mimsave(buffer, frames, format="gif", fps=FPS, loop=0)
        buffer.seek(0)
        display(IPyImage(data=buffer.read()))



# Draw a line between two points, if they are positive points
def draw_line(ax, joint1, joint2, c='black', width=2):
    if joint1[0] > -100 and joint1[1] > -100 and joint2[0] > -100 and joint2[1] > -100:
        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], color=c, linewidth=width)

# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(ax, joints):
    offset = [350, 250]
    skeleton = np.array(getSkeletalModelStructure())
    joints = joints * 10 * 12 * 2 + np.ones((75, 2)) * offset
    for j in range(skeleton.shape[0]):
        c = get_bone_colour(skeleton, j)
        joint1 = joints[skeleton[j, 0]]
        joint2 = joints[skeleton[j, 1]]
        draw_line(ax, joint1, joint2, c=c, width=2)

# get bone colour given index
def get_bone_colour(skeleton,j):

    return 'black'

# Apply DTW to the produced sequence, so it can be visually compared to the reference sequence
def alter_DTW_timing(pred_seq,ref_seq):

    # Define a cost function
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # Cut the reference down to the max count value
    #_ , ref_max_idx = torch.max(ref_seq[:, -1], 0)
    #if ref_max_idx == 0: ref_max_idx += 1
    # Cut down frames by counter
    ref_seq = ref_seq.cpu().numpy()

    # Cut the hypothesis down to the max count value
    #_, hyp_max_idx = torch.max(pred_seq[:, -1], 0)
    #if hyp_max_idx == 0: hyp_max_idx += 1
    # Cut down frames by counter
    pred_seq = pred_seq.cpu().numpy()
    #pred_seq = pred_seq[:ref_max_idx, :].cpu().numpy()
    # Run DTW on the reference and predicted sequence
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq, pred_seq, dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Initialise new sequence
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []
    for (i, pred_num) in enumerate(path[0]):

        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        # Just finished a double
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d

# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame

    avg_frame = frames_sum / len(frames)
    return avg_frame