import robosuite as suite
import numpy as np
import h5py
import pygame

# Initialize pygame for joystick input
pygame.init()
pygame.joystick.init()

# Check if a controller is connected
if pygame.joystick.get_count() == 0:
    raise Exception(
        "No game controller detected! Please connect a joystick or gamepad.")

# Get the first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Using controller: {joystick.get_name()}")

# Initialize the Robosuite environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

# Storage for demo data
demo_data = {
    "actions": [],
    "states": [],
}

# Reset environment
obs = env.reset()
demo_data["states"].append(env.sim.get_state().flatten())

print("Recording demonstration using joystick. Press START or ESC to stop.")

# Define a function to get joystick input


def get_joystick_action():
    pygame.event.pump()  # Process joystick events

    # Read axis values (adjust indices if needed)
    axis_0 = joystick.get_axis(0)  # Left stick X (left/right)
    axis_1 = joystick.get_axis(1)  # Left stick Y (up/down)
    axis_2 = joystick.get_axis(2)  # Right stick X (rotation)
    axis_3 = joystick.get_axis(3)  # Right stick Y (forward/backward)
    trigger = joystick.get_axis(4)  # Grip control (if applicable)

    # Convert joystick inputs to robot actions
    action = np.array([
        -axis_1,  # Move forward/backward
        axis_0,   # Move left/right
        -axis_3,  # Move up/down
        axis_2,   # Rotate
        0, 0, 0   # Placeholder for other actions
    ])

    return action


# Recording loop
running = True
while running:
    action = get_joystick_action()

    obs, reward, done, _ = env.step(action)

    # Save actions and states
    demo_data["actions"].append(action)
    demo_data["states"].append(env.sim.get_state().flatten())

    env.render()

    # Stop recording if START or ESC is pressed
    # Start button (index 7) or Select (index 6)
    if joystick.get_button(7) or joystick.get_button(6):
        print("Stopping recording...")
        running = False

# Save demonstration
with h5py.File("joystick_demo.h5", "w") as f:
    demo_group = f.create_group("data/demo_0")
    demo_group.create_dataset("actions", data=np.array(demo_data["actions"]))
    demo_group.create_dataset("states", data=np.array(demo_data["states"]))

print("Demonstration saved successfully!")
env.close()
