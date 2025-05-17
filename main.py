# main.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from Planets import *

SIMULATION_DATA_FILE = "datas/v1.1_example.json"
DEFAULT_PLOT_LIMITS = (-1e8, 1e8)  # plot limit default values
PLOT_PADDING_FACTOR = 0.20
MIN_PLOT_PADDING = 1e6             # absolute minimum padding

def setup_plot() -> tuple[Figure, Axes]:
    """
    Initializes the plot for the simulation.

    Returns:
        tuple[Figure, Axes]: The figure and axes objects for the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()

    ax.set_aspect('equal', adjustable='box')
    ax.set_title("N-Body Simulation")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(True)

    ax.set_xlim(*DEFAULT_PLOT_LIMITS)
    ax.set_ylim(*DEFAULT_PLOT_LIMITS)
    return fig, ax


def update_plot_limits(ax: Axes, planets_list: list[Planet]):
    """
    Dynamically updates the plot limits based on the positions of the planets.
    """

    if not planets_list:
        return

    min_x_seen = min(p.position[0] for p in planets_list if p.position is not None)
    max_x_seen = max(p.position[0] for p in planets_list if p.position is not None)
    min_y_seen = min(p.position[1] for p in planets_list if p.position is not None)
    max_y_seen = max(p.position[1] for p in planets_list if p.position is not None)

    if not all(np.isfinite(val) for val in [min_x_seen, max_x_seen, min_y_seen, max_y_seen]):
        return

    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    padding_x = abs(max_x_seen - min_x_seen) * PLOT_PADDING_FACTOR + MIN_PLOT_PADDING
    padding_y = abs(max_y_seen - min_y_seen) * PLOT_PADDING_FACTOR + MIN_PLOT_PADDING

    new_xlim_min = min(current_xlim[0], min_x_seen - padding_x)
    new_xlim_max = max(current_xlim[1], max_x_seen + padding_x)
    new_ylim_min = min(current_ylim[0], min_y_seen - padding_y)
    new_ylim_max = max(current_ylim[1], max_y_seen + padding_y)

    span_x = new_xlim_max - new_xlim_min
    span_y = new_ylim_max - new_ylim_min
    max_span = max(span_x, span_y)

    center_x = (new_xlim_min + new_xlim_max) / 2
    center_y = (new_ylim_min + new_ylim_max) / 2

    final_xlim_min = center_x - max_span / 2
    final_xlim_max = center_x + max_span / 2
    final_ylim_min = center_y - max_span / 2
    final_ylim_max = center_y + max_span / 2

    if np.isfinite(final_xlim_min) and np.isfinite(final_xlim_max) and \
       np.isfinite(final_ylim_min) and np.isfinite(final_ylim_max) and \
       final_xlim_max > final_xlim_min and final_ylim_max > final_ylim_min:

        ax.set_xlim(final_xlim_min, final_xlim_max)
        ax.set_ylim(final_ylim_min, final_ylim_max)


def run_simulation(planet_system: Planets, ax: Axes):
    """
    Runs the simulation step by step, updating the plot as needed.
    """

    print("Starting simulation...")
    print("-" * 30)

    running = True
    while running:
        simulation_code = planet_system.step_forward()

        if simulation_code == 1:
            update_plot_limits(ax, planet_system.planets)
            plt.pause(1e-6)

        elif simulation_code == 2:
            print("Pausing simulation")
            running = False
    
    print("-" * 30)
    print("Finishing simulation")


def create_example_planet_system(ax: Axes, G_const: float, dt_val: float, num_steps: int, steps_per_show: int, datapoints_hist: int) -> Planets:
    """
    Creates an example planetary system for demonstration purposes.
    """

    planets_list = [
        Planet(name="Sun", mass=1.989e30,
               pos=np.array([0.0, 0.0]),
               vel=np.array([0.0, 0.0]),
               datapoints=datapoints_hist,
               axis=ax),
        Planet(name="Earth", mass=5.972e24,
               pos=np.array([1.496e11, 0.0]),
               vel=np.array([0.0, 2.978e4]),
               datapoints=datapoints_hist,
               axis=ax),
        Planet(name="Moon", mass=7.342e22,
               pos=np.array([1.496e11 + 3.844e8, 0.0]),
               vel=np.array([0.0, 2.978e4 + 1.022e3]),
               datapoints=datapoints_hist,
               axis=ax)
    ]

    return Planets(planets=planets_list,
                   G=G_const,
                   dt=dt_val,
                   ms=num_steps,
                   sps=steps_per_show,
                   axis=ax)

def main():
    """
    Main function to run the simulation.
    """

    fig, ax = setup_plot()
    planet_system = None

    try:
        print(f"Loading simulation: {SIMULATION_DATA_FILE}")
        planet_system = load(SIMULATION_DATA_FILE, ax)

    except FileNotFoundError:
        print(f"Error: the '{SIMULATION_DATA_FILE}' file was not found.")

        G_CONSTANT_EX = 6.67430e-11
        DT_EX = 1000
        NUM_STEPS_EX = 50000
        STEPS_PER_SHOW_EX = 50
        DATAPOINTS_HISTORY_EX = 500
        planet_system = create_example_planet_system(ax, G_CONSTANT_EX, DT_EX, NUM_STEPS_EX, STEPS_PER_SHOW_EX, DATAPOINTS_HISTORY_EX)

    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        plt.ioff()
        plt.close(fig)
        return

    if planet_system is None or not planet_system.planets:
        print("Error: No planets found in the simulation data.")
        plt.ioff()
        plt.close(fig)
        return

    for p in planet_system.planets:
        p.show()
    ax.legend(loc="upper right")
    plt.draw()

    # planet_system.save("datas/v1.1_example.json")

    run_simulation(planet_system, ax)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()