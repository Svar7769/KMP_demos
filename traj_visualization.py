import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the demonstration file
demo_file = "keyboard_demo.h5"

with h5py.File(demo_file, "r") as f:
    demo_keys = list(f["data"].keys())  # Get all demo keys
    num_demos = len(demo_keys)  # Count total demonstrations
    print(f"Total demonstrations in file: {num_demos}")

# Create a 3D plot for all demos
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Iterate over each demo
for demo_idx in range(num_demos):
    with h5py.File(demo_file, "r") as f:
        states = np.array(f[f"data/demo_{demo_idx}/states"])  # Load state data

    # Extract End-Effector Positions (Assuming x, y, z are the first 3 elements)
    ee_positions = states[:, :3]  # Extract (x, y, z)

    # Plot each trajectory in 3D
    ax.plot(ee_positions[:, 0], ee_positions[:, 1],
            ee_positions[:, 2], label=f"Demo {demo_idx}")

    # Mark start and end points
    ax.scatter(ee_positions[0, 0], ee_positions[0, 1],
               ee_positions[0, 2], c="g", marker="o")  # Start
    ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1],
               ee_positions[-1, 2], c="r", marker="x")  # End

# Labels and title
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title(f"End-Effector Trajectories ({num_demos} Demos)")
ax.legend()
plt.show()
