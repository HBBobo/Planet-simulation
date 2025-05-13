import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from myfunctions import Satellite, acceleration # Import from your file

# --- Simulation Constants ---
G_CONSTANT = 6.67430e-11 * 2   # Gravitational constant (m^3 kg^-1 s^-2)
DT = 1000.0                    # Time step (seconds)
NUM_STEPS = 10000              # Total number of simulation steps
ds = 32                        # Datapoints size

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(8, 8))
plt.ion() # Interactive mode on

ax.set_aspect('equal', adjustable='box')
ax.set_title("Simulation (Testing Movement)")
ax.set_xlabel("X position (10^8 m)")
ax.set_ylabel("Y position (10^8 m)")
ax.grid(True)

ax.set_xlim(-1e8, 1e8)
ax.set_ylim(-1e8, 1e8)

planets = [
    Satellite(name="Earth", mass=7.67e25,
              pos=np.array([0.0, 0.0]),
              vel=np.array([0.0, 1.0e3]),
              datapoints=ds),

    Satellite(name="Moon", mass=1.67e24,
              pos=np.array([0.0, -2.27e7]),
              vel=np.array([3.0e5, 0.0]),
              datapoints=ds),

    Satellite(name="Asteroid1", mass=1.1e25,
              pos=np.array([1.2e7, 5.27e7]),
              vel=np.array([8.3e4, -2.5e5]),
              datapoints=ds),

    Satellite(name="Asteroid2", mass=0.5e22,
              pos=np.array([-5.2e7, 3.27e7]),
              vel=np.array([1.5e5, -2.5e5]),
              datapoints=ds),

    Satellite(name="Asteroid3", mass=1.4e19,
              pos=np.array([3.4e7, 5.78e6]),
              vel=np.array([1.3e5, -1.5e5]),
              datapoints=ds)
]

head_markers = [ax.plot([], [], 'o', markersize=6)[0] for _ in planets]
lines = [ax.plot([], [], '-')[0] for _ in planets]

for i, marker in enumerate(head_markers):
    marker.set_label(planets[i].name)

ax.legend(loc="upper right")

print("Starting simulation (testing movement)...")
print("-" * 30)

for step in range(NUM_STEPS):
    if len(planets) >= 2:
        for i in range(len(planets)):
            for j in range(i + 1, len(planets)):
                try:
                    acceleration(planets[i], planets[j], G_CONSTANT, DT)
                except ValueError as e:
                    print(f"Step {step}, Pair ({planets[i].name}, {planets[j].name}): ValueError: {e}")
                except Exception as e:
                    print(f"Step {step}, Pair ({planets[i].name}, {planets[j].name}): An unexpected error occurred: {e}")
                    raise

    min_x_seen, max_x_seen = float('inf'), float('-inf')
    min_y_seen, max_y_seen = float('inf'), float('-inf')

    for i, p in enumerate(planets):
        p.move()

        head_markers[i].set_data([p.position[0]], [p.position[1]])
        data = p.getHistory()
        lines[i].set_data(data[:, 0], data[:, 1])

        # Update the limits based on the current position of the planet
        min_x_seen = min(min_x_seen, p.position[0])
        max_x_seen = max(max_x_seen, p.position[0])
        min_y_seen = min(min_y_seen, p.position[1])
        max_y_seen = max(max_y_seen, p.position[1])

    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # Dynamically adjust the limits based on the min/max positions seen (20% padding)
    padding_x = abs(max_x_seen - min_x_seen) * 0.20 + 1e6
    padding_y = abs(max_y_seen - min_y_seen) * 0.20 + 1e6

    new_xlim_min = min(current_xlim[0], min_x_seen - padding_x)
    new_xlim_max = max(current_xlim[1], max_x_seen + padding_x)
    new_ylim_min = min(current_ylim[0], min_y_seen - padding_y)
    new_ylim_max = max(current_ylim[1], max_y_seen + padding_y)

    new_max = max(new_xlim_max, new_ylim_max)
    new_min = min(new_xlim_min, new_ylim_min)

    if np.isfinite(new_min) and np.isfinite(new_max) and new_max > new_min:
        ax.set_xlim(new_min, new_max)
        ax.set_ylim(new_min, new_max)

    plt.pause(1e-6) # Save the CPU :D

    if step % 100 == 0 and step > 0:
        print(f"Step {step}/{NUM_STEPS} completed.\n")
        for p in planets:
             print(p)
             print("\n")

print("Simulation finished.")
plt.ioff()
plt.show()
