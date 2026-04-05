import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. Simulate trajectories
# -----------------------------

T = 300
time = np.linspace(0, 1, T)

# Image 1: feedforward-like cascade
V1_img1 = np.exp(-((time - 0.2) ** 2) / 0.01)
V4_img1 = np.exp(-((time - 0.4) ** 2) / 0.02)
IT_img1 = np.exp(-((time - 0.6) ** 2) / 0.03)
traj1 = np.vstack([V1_img1, V4_img1, IT_img1]).T

# Image 2: hierarchy-breaking / oscillatory
V1_img2 = np.sin(4 * np.pi * time) * np.exp(-time)
V4_img2 = np.cos(3 * np.pi * time) * np.exp(-0.5 * time)
IT_img2 = np.sin(2 * np.pi * time + 1.0)
traj2 = np.vstack([V1_img2, V4_img2, IT_img2]).T


# -----------------------------
# 2. Trajectory metrics
# -----------------------------

def trajectory_length(traj):
    diffs = np.diff(traj, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def trajectory_curvature(traj):
    v1 = np.diff(traj, axis=0)[:-1]
    v2 = np.diff(traj, axis=0)[1:]

    numerator = np.linalg.norm(v2 - v1, axis=1)
    denominator = np.linalg.norm(v1, axis=1) ** 2

    curvature = np.zeros_like(numerator)
    valid = denominator > 1e-8
    curvature[valid] = numerator[valid] / denominator[valid]

    return np.mean(curvature)


print("Image 1 Length:", trajectory_length(traj1))
print("Image 1 Mean Curvature:", trajectory_curvature(traj1))
print("Image 2 Length:", trajectory_length(traj2))
print("Image 2 Mean Curvature:", trajectory_curvature(traj2))


# -----------------------------
# 3. 3D Plot
# -----------------------------

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectories
ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], linewidth=2)
ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], linestyle='dashed', linewidth=2)

# Large dot at beginning
ax.scatter(traj1[0, 0], traj1[0, 1], traj1[0, 2], s=120)
ax.scatter(traj2[0, 0], traj2[0, 1], traj2[0, 2], s=120)

# Arrow at end (use last segment direction)
def add_arrow(ax, traj):
    p_start = traj[-2]
    p_end = traj[-1]
    direction = p_end - p_start
    ax.quiver(
        p_start[0], p_start[1], p_start[2],
        direction[0], direction[1], direction[2],
        length=1.0,
        normalize=False
    )

add_arrow(ax, traj1)
add_arrow(ax, traj2)


# -----------------------------
# 4. Force axes to intersect at 0
# -----------------------------

# Get symmetric limits around zero
all_data = np.vstack([traj1, traj2])
max_val = np.max(np.abs(all_data))

ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, max_val)
ax.set_zlim(-max_val, max_val)

# Draw axis lines through origin
ax.plot([-max_val, max_val], [0, 0], [0, 0])
ax.plot([0, 0], [-max_val, max_val], [0, 0])
ax.plot([0, 0], [0, 0], [-max_val, max_val])

ax.set_xlabel("V1 Activity")
ax.set_ylabel("V4 Activity")
ax.set_zlabel("IT Activity")
ax.set_title("3D Neural Trajectories (Start Dot + End Arrow)")

plt.show()