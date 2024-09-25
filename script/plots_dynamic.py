import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from matplotlib.animation import FuncAnimation

def read_positions_from_file(file_path, agent_id) -> dict:
    """Read agent positions from a file and return a dictionary of positions."""
    positions = []
    
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split())
            positions.append((x, y))
    
    return {agent_id: positions}

def extract_task_info(path) -> dict:
    """Extract task information from a YAML file."""
    with open(path, 'r') as file:
                tasks = yaml.safe_load(file)
    return tasks

def is_inside_barrier(agent_pos, barrier_center, epsilon):
    """Check if an agent is inside a circular barrier."""
    distance = np.linalg.norm(np.array(agent_pos) - np.array(barrier_center))
    return distance <= epsilon

def is_barrier_activated(interval, phase):
    """Check if the barrier is activated based on the current frame."""
    if phase == 1:
        return interval[1] <= 80
    elif phase == 2:
        return interval[0] > 80 and interval[1] <= 170
    elif phase == 3:
        return interval[0] > 170


def check_phase(frame):
    if frame <= 800:
        return 1
    elif frame > 800 and frame <= 1700:
        return 2
    elif frame > 1700:
        return 3


def animate_agent_positions(agent_positions, tasks):
    """Animate the positions of each agent."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Start the animation in phase 1a
    phase = 1

    # Define the colors for each agent
    agent_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple', 6: 'brown', 7: 'magenta'}

    # Initialize a line and an arrow for each agent
    lines = {}
    arrows = {}
    barriers = {}
    
    for agent_id, positions in agent_positions.items():
        # Agent movement line
        line, = ax.plot([], [], marker='o', markersize=1, color=agent_colors[agent_id], label=f'Agent {agent_id}')
        lines[agent_id] = line

        # Create barriers based on tasks
        for task_name, task_info in tasks.items():
            center = task_info['CENTER']
            epsilon = task_info['EPSILON']
            involved_agents = task_info['INVOLVED_AGENTS']
            task_type = task_info['TYPE']
            interval = task_info['INTERVAL']

            if agent_id in involved_agents:
                # Arrow to barrier
                arrows[f"{task_name}_{agent_id}"] = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5))

                # Barrier
                # barrier = plt.Circle((center[0], center[1]), epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3)
                barrier = plt.Circle((center[0], center[1]), epsilon, fill=False, edgecolor=agent_colors[agent_id])
                ax.add_patch(barrier)
                barriers[f"{task_name}_{agent_id}"] = barrier

                    
    # Set the plot limits (adjust these limits based on your data)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Movements with Arrows Pointing to Goal Areas')
    ax.legend()
    ax.grid(True)

    # Find the maximum number of frames (longest movement list)
    max_frames = max(len(positions) for positions in agent_positions.values())

    def init():
        """Initialize the plot."""
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values()) + list(barriers.values()) + list(arrows.values())

    def update(frame):
        
        # Update the phase
        phase = check_phase(frame)
        
        """Update the plot for each frame."""
        for agent_id, positions in agent_positions.items():
            if frame < len(positions):
                x_coords, y_coords = zip(*positions[:frame + 1])  # Slice up to the current frame
                lines[agent_id].set_data(x_coords, y_coords)
            


            # Update barriers based on the task type
            for task_name, task_info in tasks.items():
                center = task_info['CENTER']
                epsilon = task_info['EPSILON']
                involved_agents = task_info['INVOLVED_AGENTS']
                task_type = task_info['TYPE']
                interval = task_info['INTERVAL']

                if agent_id in involved_agents:

                    if task_type == "formation_predicate":
                        # Move the rectangle relative to the neighbor

                        if agent_id == involved_agents[0]:
                            current_pos = agent_positions[involved_agents[0]][frame]
                            current_neighbor_pos = agent_positions[involved_agents[1]][frame]

                            # Update the barrier
                            if is_barrier_activated(interval, phase):
                                barriers[f"{task_name}_{agent_id}"].set_center((current_neighbor_pos[0] - center[0], current_neighbor_pos[1] - center[1]))
                                barriers[f"{task_name}_{agent_id}"].set_visible(True) 

                                # Check if the agent is inside the barrier and update the arrow
                                if not is_inside_barrier(current_pos, (current_neighbor_pos[0] - center[0], current_neighbor_pos[1] - center[1]), epsilon):
                                    # Update the arrow only if the agent is outside the barrier
                                    arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                                    arrows[f"{task_name}_{agent_id}"].xy = (current_neighbor_pos[0] - center[0], current_neighbor_pos[1] - center[1])
                                    arrows[f"{task_name}_{agent_id}"].set_visible(True)
                                else:
                                    arrows[f"{task_name}_{agent_id}"].set_visible(False)
                            else:
                                barriers[f"{task_name}_{agent_id}"].set_visible(False)
                                arrows[f"{task_name}_{agent_id}"].set_visible(False)

                            

                        else:
                            current_pos = agent_positions[involved_agents[1]][frame]
                            current_neighbor_pos = agent_positions[involved_agents[0]][frame]

                            # Update the barrier
                            if is_barrier_activated(interval, phase):
                                barriers[f"{task_name}_{agent_id}"].set_center((current_neighbor_pos[0] + center[0], current_neighbor_pos[1] + center[1]))
                                barriers[f"{task_name}_{agent_id}"].set_visible(True)

                                # Check if the agent is inside the barrier and update the arrow
                                if not is_inside_barrier(current_pos, (current_neighbor_pos[0] + center[0], current_neighbor_pos[1] + center[1]), epsilon):
                                    # Update the arrow only if the agent is outside the barrier
                                    arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                                    arrows[f"{task_name}_{agent_id}"].xy = (current_neighbor_pos[0] + center[0], current_neighbor_pos[1] + center[1])
                                    arrows[f"{task_name}_{agent_id}"].set_visible(True)
                                else:
                                    arrows[f"{task_name}_{agent_id}"].set_visible(False)
                            else:
                                barriers[f"{task_name}_{agent_id}"].set_visible(False)
                                arrows[f"{task_name}_{agent_id}"].set_visible(False)




                    elif task_type == "epsilon_position_closeness_predicate":
                        # Move the rectangle with the neighbor
                        if agent_id == involved_agents[0]:
                            neighbor_id = involved_agents[1]
                        else:  
                            neighbor_id = involved_agents[0]

                        current_pos = agent_positions[agent_id][frame]
                        current_neighbor_pos = agent_positions[neighbor_id][frame]

                        # Update the barrier
                        if is_barrier_activated(interval, phase):
                            barriers[f"{task_name}_{agent_id}"].set_center((current_neighbor_pos[0], current_neighbor_pos[1]))
                            barriers[f"{task_name}_{agent_id}"].set_visible(True)

                            # Check if the agent is inside the barrier and update the arrow
                            if not is_inside_barrier(current_pos, (current_neighbor_pos[0], current_neighbor_pos[1]), epsilon):
                                # Update the arrow only if the agent is outside the barrier
                                arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                                arrows[f"{task_name}_{agent_id}"].xy = (current_neighbor_pos[0], current_neighbor_pos[1])
                                arrows[f"{task_name}_{agent_id}"].set_visible(True)
                            else:
                                arrows[f"{task_name}_{agent_id}"].set_visible(False)   
                        else:
                            barriers[f"{task_name}_{agent_id}"].set_visible(False)
                            arrows[f"{task_name}_{agent_id}"].set_visible(False)


                    elif task_type == "go_to_goal_predicate_2d":
                        current_pos = agent_positions[agent_id][frame]

                        # Update the barrier
                        if is_barrier_activated(interval, phase):
                            barriers[f"{task_name}_{agent_id}"].set_center((center[0], center[1]))
                            barriers[f"{task_name}_{agent_id}"].set_visible(True)

                            # Check if the agent is inside the barrier and update the arrow
                            if not is_inside_barrier(current_pos, center, epsilon):
                                # Update the arrow only if the agent is outside the barrier
                                arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                                arrows[f"{task_name}_{agent_id}"].xy = (center[0], center[1])
                                arrows[f"{task_name}_{agent_id}"].set_visible(True)
                            else:
                                arrows[f"{task_name}_{agent_id}"].set_visible(False)
                        else:
                            barriers[f"{task_name}_{agent_id}"].set_visible(False)
                            arrows[f"{task_name}_{agent_id}"].set_visible(False)





        return list(lines.values()) + list(barriers.values()) + list(arrows.values())

    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=10, repeat=False)

    # Save the animation as an MP4 file
    ani.save('animation.mp4', writer='ffmpeg', fps=30)

    # # Display the animation
    # plt.show()


def main():
    # List of file paths with corresponding agent IDs
    file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2),('agent_3_positions.txt', 3),('agent_4_positions.txt', 4),('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # Initialize an empty dictionary to store all agent positions
    all_agent_positions = {}
    
    # task_dir = '/home/benjaminb/Desktop/ros2_ws/src/stl_task_decomposition'
    task_dir = '/home/benjamin/ros2_ws/src/stl_task_decomposition'
    tasks_yaml_path = os.path.join(task_dir, 'config', 'decomp_tasks.yaml')
    tasks = extract_task_info(tasks_yaml_path)
    

    for file_path, agent_id in file_paths_with_ids:
        # Read positions from each file and update the dictionary
        agent_positions = read_positions_from_file(file_path, agent_id)
        all_agent_positions.update(agent_positions)
    
    # Animate all agent positions
    animate_agent_positions(all_agent_positions, tasks)

if __name__ == '__main__':
    main()
