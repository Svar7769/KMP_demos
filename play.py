import h5py
import robosuite as suite
import numpy as np
import time
import pygame

# Initialize pygame for ESC key exit
pygame.init()
screen = pygame.display.set_mode((300, 300))

# Load the demonstration file
demo_file = "keyboard_demo.h5"
with h5py.File(demo_file, "r") as f:
    num_demos = len(f["data"])
    print(f"Loaded {num_demos} demonstrations.")

# Initialize the Robosuite environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    hard_reset=False
)

# Replay each demonstration
for demo_idx in range(num_demos):
    with h5py.File(demo_file, "r") as f:
        actions = np.array(f[f"data/demo_{demo_idx}/actions"])
        states = np.array(f[f"data/demo_{demo_idx}/states"])

    print(f"Replaying demo {demo_idx + 1}...")

    # Reset environment and set the initial state
    env.reset()
    env.sim.set_state_from_flattened(states[0])  # Restore the initial state
    env.sim.forward()

    print("Press ESC to stop replay.")

    for action in actions:
        # Check for ESC key press to exit
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("Exiting replay...")
                env.close()
                pygame.quit()
                exit()

        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)  # Control speed of playback

print("Replay complete!")
env.close()
pygame.quit()
