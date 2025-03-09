import robosuite as suite
import numpy as np
import h5py
import pygame  # Pygame for keyboard input handling
import time

# Initialize pygame for keyboard input
pygame.init()
# Dummy window required for key events
screen = pygame.display.set_mode((300, 300))

# Initialize the Robosuite environment
env = suite.make(
    env_name="Lift",
    robots="UR5e",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,  # Frequency of actions
)

# Define movement and gripper control mappings
key_to_action = {
    pygame.K_w: [0.5, 0, 0, 0, 0, 0, 0],   # Move forward
    pygame.K_s: [-0.5, 0, 0, 0, 0, 0, 0],  # Move backward
    pygame.K_a: [0, -0.5, 0, 0, 0, 0, 0],  # Move left
    pygame.K_d: [0, 0.5, 0, 0, 0, 0, 0],   # Move right
    pygame.K_q: [0, 0, 0, -0.5, 0, 0, 0],  # Rotate left
    pygame.K_e: [0, 0, 0, 0.5, 0, 0, 0],   # Rotate right
    pygame.K_r: [0, 0, 0.5, 0, 0, 0, 0],   # Move up
    pygame.K_f: [0, 0, -0.5, 0, 0, 0, 0],  # Move down
    pygame.K_g: [0, 0, 0, 0, 1, 1, 0],     # Close gripper
    pygame.K_h: [0, 0, 0, 0, -1, -1, 0],   # Open gripper
}

# Storage for demo data
demo_data = {
    "actions": [],
    "states": [],
}

# Reset environment
obs = env.reset()
demo_data["states"].append(env.sim.get_state().flatten())

print("Recording demo using keyboard. Use W/A/S/D for movement, Q/E for rotation, R/F for up/down, G/H for gripper. Press ESC to stop.")

# Recording loop
running = True
while running:
    action = np.zeros(7)  # Default zero action

    # Process Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key in key_to_action:
                action = np.array(key_to_action[event.key])
            elif event.key == pygame.K_ESCAPE:  # Stop recording if ESC is pressed
                print("Stopping recording...")
                running = False

    obs, reward, done, _ = env.step(action)

    # Save actions and states
    demo_data["actions"].append(action)
    demo_data["states"].append(env.sim.get_state().flatten())

    env.render()
    time.sleep(0.05)  # Prevent excessive CPU usage

# Save demonstration
with h5py.File("keyboard_demo_wsl.h5", "w") as f:
    demo_group = f.create_group("data/demo_0")
    demo_group.create_dataset("actions", data=np.array(demo_data["actions"]))
    demo_group.create_dataset("states", data=np.array(demo_data["states"]))

print("Demonstration saved successfully as keyboard_demo_wsl.h5!")
env.close()
pygame.quit()
