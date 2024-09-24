import matplotlib.pyplot as plt
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


def animate_agent_positions(agent_positions, tasks):
    """Animate the positions of each agent."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
        # Arrow from agent to goal
        # arrows[agent_id] = ax.annotate('', xy=(0, 0), xytext=(0, 0),
        #                                arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5))
        
        
        # Create barriers based on tasks
        for task_name, task_info in tasks.items():
            center = task_info['CENTER']
            epsilon = task_info['EPSILON']
            involved_agents = task_info['INVOLVED_AGENTS']
            task_type = task_info['TYPE']

            if agent_id in involved_agents:

                if agent_id == involved_agents[0]:
                    neighbor_id = involved_agents[1]
                else:  
                    neighbor_id = involved_agents[0]


                if task_type == 'go_to_goal_predicate_2d':
                    # Create a static rectangle
                    # barrier= plt.Rectangle((center[0] - epsilon / 2, center[1] - epsilon / 2), epsilon, epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
                    barrier = plt.Circle((center[0], center[1]), epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
                    ax.add_patch(barrier)
                    barriers[task_name] = barrier
                
                elif task_type == "formation_predicate":
                    # Create a rectangle relative to the position of the first agent
                    # barrier= plt.Rectangle((center[0] - epsilon / 2, center[1] - epsilon / 2), epsilon, epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
                    barrier = plt.Circle((center[0], center[1]), epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
                    ax.add_patch(rect)
                    barriers[f"{task_name}_{agent_id}"] = barrier

                elif task_type == "epsilon_position_closeness_predicate":
                    # Create a rectangle that moves along with the two agents
                    # barrier= plt.Rectangle((center[0] - epsilon / 2, center[1] - epsilon / 2), epsilon, epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
                    barrier = plt.Circle((center[0], center[1]), epsilon, fill=True, color=agent_colors[agent_id], alpha=0.3, label=f'Task {task_name}')
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
        # for arrow in arrows.values():
        #     arrow.set_position((0, 0))  # Reset arrows
        return list(lines.values()) + list(barriers.values())#+ list(arrows.values()) 

    def update(frame):
        """Update the plot for each frame."""
        for agent_id, positions in agent_positions.items():
            if frame < len(positions):
                x_coords, y_coords = zip(*positions[:frame + 1])  # Slice up to the current frame
                lines[agent_id].set_data(x_coords, y_coords)
                
            #     # Draw arrow from current position to the goal
            #     if agent_id in goal_areas:
            #         current_pos = positions[frame]  # Current position of the agent
            #         goal_pos = goal_areas[agent_id]  # Goal position
            #         arrows[agent_id].set_position(current_pos)
            #         arrows[agent_id].xy = goal_pos
            
            
            # Update barriers based on the task type
            for task_name, task_info in tasks.items():
                center = task_info['CENTER']
                epsilon = task_info['EPSILON']
                involved_agents = task_info['INVOLVED_AGENTS']
                task_type = task_info['TYPE']

                if agent_id in involved_agents:

                    if task_type == "formation_predicate":
                        # Move the rectangle relative to the first agent
                        if agent_id == involved_agents[0]:
                            current_pos = agent_positions[agent_id][frame]
                            current_neighbor_pos = agent_positions[involved_agents[1]][frame]
                            barriers[f"{task_name}_{agent_id}"].set_xy((current_pos[0] + center[0]- epsilon / 2, current_pos[1] + center[1]- epsilon / 2))
                        else:
                            current_pos = agent_positions[involved_agents[1]][frame]
                            current_neighbor_pos = agent_positions[involved_agents[0]][frame]
                            barriers[f"{task_name}_{agent_id}"].set_xy((current_neighbor_pos[0] + center[0]- epsilon / 2, current_neighbor_pos[1] + center[1]- epsilon / 2))


                        # barriers[task_name].set_xy((current_pos[0] - epsilon / 2, current_pos[1] - epsilon / 2))

                    elif task_type == "epsilon_position_closeness_predicate":
                        # Move the rectangle to the midpoint of the two agents
                        if agent_id == involved_agents[0]:
                            current_pos = agent_positions[agent_id][frame]
                            current_neighbor_pos = agent_positions[involved_agents[1]][frame]
                            # barriers[task_name].set_xy((current_neighbor_pos[0], current_neighbor_pos[1]))
                            barriers[f"{task_name}_{agent_id}"].set_center((current_neighbor_pos[0], current_neighbor_pos[1]))
                            
                            
                        else:
                            current_pos = agent_positions[involved_agents[1]][frame]
                            current_neighbor_pos = agent_positions[involved_agents[0]][frame]
                            # barriers[task_name].set_xy((current_pos[0], current_pos[1]))
                            barriers[f"{task_name}_{agent_id}"].set_center((current_pos[0], current_pos[1]))

        return list(lines.values()) + list(barriers.values())#+ list(arrows.values()) 


    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=50, repeat=False)

    # Display the animation
    plt.show()


def main():
    # List of file paths with corresponding agent IDs
    file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2),('agent_3_positions.txt', 3),('agent_4_positions.txt', 4),('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # Initialize an empty dictionary to store all agent positions
    all_agent_positions = {}
    
    task_dir = '/home/benjaminb/Desktop/ros2_ws/src/stl_task_decomposition'
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
