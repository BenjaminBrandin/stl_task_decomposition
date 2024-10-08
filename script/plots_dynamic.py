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
    
    # Start the animation in phase 1
    phase = 1

    decomp_tasks = ["TASK_1", "TASK_6", "TASK_9", "TASK_12", "TASK_15"]
    decomp_arrows = ["TASK_2", "TASK_7", "TASK_10", "TASK_13", "TASK_16"]
    # decomp_agents = {1: 5, 2: 1, 3: (1,2)}
    decomp_agents = [5, 1, 1]

    # Define the colors for each agent
    agent_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple', 6: 'brown', 7: 'magenta'}

    # Initialize a line and an arrow for each agent
    lines = {}
    arrows = {}
    barriers = {}
    agents = {}
    agent_texts = {}
    decomp_texts = {}
    barrier_positions = {}
    
    for agent_id, positions in agent_positions.items():
        # Agent movement line
        line, = ax.plot([], [], marker='o', markersize=1, color=agent_colors[agent_id], label=f'Agent {agent_id}')
        lines[agent_id] = line


        # Create circle for the agent and add it to the plot
        agent_circle = plt.Circle((0, 0), 0.05, fill=True, color=agent_colors[agent_id], alpha=1.0)
        ax.add_patch(agent_circle)
        agents[agent_id] = agent_circle

        # Create text for the agent and add it to the plot
        text = ax.text(0, 0, f'Agent {agent_id}', ha='center', va='center', color=agent_colors[agent_id])
        agent_texts[agent_id] = text


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

                if task_name in decomp_tasks:
                    barrier = plt.Circle((center[0], center[1]), epsilon, fill=False, linestyle=':', linewidth=1.5)
                    # Create text for the barrier
                    decomp_text = ax.text(center[0], center[1] + 0.1, f'collab_task_{involved_agents[0]}-{involved_agents[1]}', ha='center', va='bottom')
                    decomp_texts[f"{task_name}_{agent_id}"] = decomp_text
                    arrows[f"{task_name}_{agent_id}"] = ax.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5, linestyle=(0, (3, 10))))
                else:
                    barrier = plt.Circle((center[0], center[1]), epsilon, fill=False, edgecolor=agent_colors[agent_id])
   
                ax.add_patch(barrier)
                barriers[f"{task_name}_{agent_id}"] = barrier

     
    # Set the plot limits (adjust these limits based on your data)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Agent Movements with Arrows Pointing to Barriers')
    ax.legend()
    ax.grid(True)

    # Find the maximum number of frames (longest movement list)
    max_frames = max(len(positions) for positions in agent_positions.values())

    def init():
        """Initialize the plot."""
        for line in lines.values():
            line.set_data([], [])
        return list(lines.values()) + list(barriers.values()) + list(arrows.values()) + list(agents.values()) + list(agent_texts.values()) + list(decomp_texts.values())

    def update(frame):
        """Update the plot for each frame."""
        # Update the phase
        # frame = frame + 1700
        phase = check_phase(frame)
        # decomp_agents = {1: 5, 2: 1, 3: (1,2)}
        starting_agent = decomp_agents[phase-1]
        # print(f"starting_agent: {starting_agent}")
        
        # First, update all agent positions before tasks
        for agent_id, positions in agent_positions.items():
            if frame < len(positions):
                # Update the line data (agent's trail)
                x_coords, y_coords = zip(*positions[:frame + 1])  # Slice up to the current frame
                lines[agent_id].set_data(x_coords, y_coords)

                # Update the agent's circle position
                current_pos = agent_positions[agent_id][frame]
                agents[agent_id].set_center((current_pos[0], current_pos[1]))

                # Update the text position to follow the agent's circle
                agent_texts[agent_id].set_position((current_pos[0], current_pos[1] + 0.1))

        # Second, process all tasks in the correct order (starting from TASK_1)
        for task_name, task_info in tasks.items():
            task_number = int(task_name.split('_')[1])
            center = task_info['CENTER']
            epsilon = task_info['EPSILON']
            involved_agents = task_info['INVOLVED_AGENTS']
            task_type = task_info['TYPE']
            interval = task_info['INTERVAL']

            # Ensure barriers and arrows are only updated if the task is within the correct interval (based on frame)
            if is_barrier_activated(interval, phase):
                for agent_id in involved_agents:
                    current_pos = agent_positions[agent_id][frame]
                    barrier_positions["TASK_0"] = agent_positions[5][frame]
                    

                    if task_type == "formation_predicate":
                        if agent_id == involved_agents[1]:
                            current_neighbor_pos = agent_positions[involved_agents[0]][frame]
                            barrier_center = (current_neighbor_pos[0] + center[0], current_neighbor_pos[1] + center[1])
                            barrier_positions[f"{task_name}"] = barrier_center

                            # Update the barrier and arrow
                            barriers[f"{task_name}_{agent_id}"].set_center(barrier_center)
                            barriers[f"{task_name}_{agent_id}"].set_visible(True)

                            # if starting_agent in involved_agents and task_name in decomp_arrows:
                            if task_name in decomp_arrows:
                                arrows[f"{task_name}_{agent_id}"].set_position((current_neighbor_pos[0], current_neighbor_pos[1]))
                                arrows[f"{task_name}_{agent_id}"].xy = barrier_center
                                arrows[f"{task_name}_{agent_id}"].set_visible(True)
                            else:
                                if task_number-1 == 0:
                                    arrows[f"{task_name}_{agent_id}"].set_position((barrier_positions["TASK_0"][0], barrier_positions["TASK_0"][1]))
                                else:
                                    arrows[f"{task_name}_{agent_id}"].set_position((barrier_positions[f"TASK_{task_number-1}"][0], barrier_positions[f"TASK_{task_number-1}"][1]))
                                arrows[f"{task_name}_{agent_id}"].xy = barrier_center
                                arrows[f"{task_name}_{agent_id}"].set_visible(True)

                            # Update the text if it's a decomposed task
                            if task_name in decomp_tasks:
                                decomp_texts[f"{task_name}_{agent_id}"].set_position((barrier_center[0], barrier_center[1] + 0.1))
                                decomp_texts[f"{task_name}_{agent_id}"].set_visible(True)

                                arrows[f"{task_name}_{agent_id}"].set_position((current_neighbor_pos[0], current_neighbor_pos[1]))
                                arrows[f"{task_name}_{agent_id}"].xy = barrier_center
                                arrows[f"{task_name}_{agent_id}"].set_visible(True)
                        else:
                            barriers[f"{task_name}_{agent_id}"].set_visible(False)
                            arrows[f"{task_name}_{agent_id}"].set_visible(False)
                            if task_name in decomp_tasks:
                                decomp_texts[f"{task_name}_{agent_id}"].set_visible(False)

                    elif task_type == "epsilon_position_closeness_predicate":
                        # Move the barrier to the midpoint between the agents
                        if agent_id == involved_agents[0]:
                            neighbor_id = involved_agents[1]
                        else:
                            neighbor_id = involved_agents[0]
                        
                        current_neighbor_pos = agent_positions[neighbor_id][frame]
                        barriers[f"{task_name}_{agent_id}"].set_center((current_neighbor_pos[0], current_neighbor_pos[1]))
                        barriers[f"{task_name}_{agent_id}"].set_visible(True)
                        arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                        arrows[f"{task_name}_{agent_id}"].xy = (current_neighbor_pos[0], current_neighbor_pos[1])
                        arrows[f"{task_name}_{agent_id}"].set_visible(True)

                    elif task_type == "go_to_goal_predicate_2d":
                        # Set the barrier and arrow for go_to_goal
                        barriers[f"{task_name}_{agent_id}"].set_center((center[0], center[1]))
                        barriers[f"{task_name}_{agent_id}"].set_visible(True)
                        arrows[f"{task_name}_{agent_id}"].set_position((current_pos[0], current_pos[1]))
                        arrows[f"{task_name}_{agent_id}"].xy = (center[0], center[1])
                        arrows[f"{task_name}_{agent_id}"].set_visible(True)
            else:
                # If the task is not active in the current frame, hide barriers and arrows
                for agent_id in involved_agents:
                    barriers[f"{task_name}_{agent_id}"].set_visible(False)
                    arrows[f"{task_name}_{agent_id}"].set_visible(False)
                    if task_name in decomp_tasks:
                        decomp_texts[f"{task_name}_{agent_id}"].set_visible(False)

        return list(lines.values()) + list(barriers.values()) + list(arrows.values()) + list(agents.values()) + list(agent_texts.values()) + list(decomp_texts.values())

    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=10, repeat=False)

    # Save the animation as an MP4 file
    ani.save('animation.mp4', writer='ffmpeg', fps=40)

    # # Display the animation
    # plt.show()


def main():
    # List of file paths with corresponding agent IDs
    file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2),('agent_3_positions.txt', 3),('agent_4_positions.txt', 4),('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # Initialize an empty dictionary to store all agent positions
    all_agent_positions = {}
    
    task_dir = '/home/benjaminb/Desktop/ros2_ws/src/stl_task_decomposition'
    # task_dir = '/home/benjamin/ros2_ws/src/stl_task_decomposition'
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
