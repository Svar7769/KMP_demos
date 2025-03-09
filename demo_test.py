import robosuite as suite
import numpy as np
import h5py
import pygame
import time

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((300, 300))

# Key-action mapping
key_to_action = {
    pygame.K_w: [0.5, 0, 0, 0, 0, 0, 0],
    pygame.K_s: [-0.5, 0, 0, 0, 0, 0, 0],
    pygame.K_a: [0, -0.5, 0, 0, 0, 0, 0],
    pygame.K_d: [0, 0.5, 0, 0, 0, 0, 0],
    pygame.K_q: [0, 0, 0, -0.5, 0, 0, 0],
    pygame.K_e: [0, 0, 0, 0.5, 0, 0, 0],
    pygame.K_r: [0, 0, 0.5, 0, 0, 0, 0],
    pygame.K_f: [0, 0, -0.5, 0, 0, 0, 0],
    pygame.K_g: [0, 0, 0, 0, 0, 0, 1],
    pygame.K_h: [0, 0, 0, 0, 0, 0, -1],
}

# Initialize environment
env = suite.make("Lift", robots="Panda", has_renderer=True, control_freq=20)
demo_data = {"actions": [], "states": []}

# Record 5 demos
for demo_idx in range(5):
    env.reset()
    demo_data["actions"].append([])
    demo_data["states"].append([env.sim.get_state().flatten()])

    print(f"Recording demo {demo_idx + 1}. Press ESC to stop.")
    running = True
    while running:
        action = np.zeros(7)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = np.array(key_to_action[event.key])
                elif event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, done, _ = env.step(action)
        demo_data["actions"][demo_idx].append(action)
        demo_data["states"][demo_idx].append(env.sim.get_state().flatten())
        env.render()
        time.sleep(0.05)

# Save data
with h5py.File("keyboard_demo.h5", "w") as f:
    for i in range(5):
        demo_group = f.create_group(f"data/demo_{i}")
        demo_group.create_dataset(
            "actions", data=np.array(demo_data["actions"][i]))
        demo_group.create_dataset(
            "states", data=np.array(demo_data["states"][i]))

print("Demonstrations saved successfully!")
env.close()
pygame.quit()
