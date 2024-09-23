import matplotlib.pyplot as plt
import ast
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from matplotlib.animation import FuncAnimation

def read_positions_from_file(file_path, agent_id):
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
    fig, ax = plt.subplots(figsize=(15, 9))
    


    # Initialize a line and an arrow for each agent
    lines = {}
    arrows = {}
    moving_rects = {}
    
    for agent_id in agent_positions:
        # Agent movement line
        line, = ax.plot([], [], marker='o', markersize=1, color=agent_colors[agent_id], label=f'Agent {agent_id}')
        lines[agent_id] = line
        # Arrow from agent to goal
        arrows[agent_id] = ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                       arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5))
        # Moving rectangle with different starting position
        rect = plt.Rectangle(initial_rect_positions[agent_id], 1, 1, fill=True, color=agent_colors[agent_id], alpha=0.5)
        moving_rects[agent_id] = rect
        ax.add_patch(rect)

    # Set the plot limits (adjust these limits based on your data)
    ax.set_xlim(-5, 10)
    ax.set_ylim(-5, 10)
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
        for arrow in arrows.values():
            arrow.set_position((0, 0))  # Reset arrows
        return list(lines.values()) + list(arrows.values()) + list(moving_rects.values())

    def update(frame):
        """Update the plot for each frame."""
        for agent_id, positions in agent_positions.items():
            if frame < len(positions):
                x_coords, y_coords = zip(*positions[:frame + 1])  # Slice up to the current frame
                lines[agent_id].set_data(x_coords, y_coords)
                
                # Draw arrow from current position to the goal
                if agent_id in goal_areas:
                    current_pos = positions[frame]  # Current position of the agent
                    goal_pos = goal_areas[agent_id]  # Goal position
                    arrows[agent_id].set_position(current_pos)
                    arrows[agent_id].xy = goal_pos
            
            # Update the position of the moving rectangle
                current_pos = positions[frame]
                rect_x = current_pos[0] + offsets[agent_id][0]  # Add x-offset
                rect_y = current_pos[1] + offsets[agent_id][1]  # Add y-offset
                moving_rects[agent_id].set_xy((rect_x - 0.5, rect_y - 0.5))  # Center the rectangle

        return list(lines.values()) + list(arrows.values()) + list(moving_rects.values())


    # Create the animation
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=50, repeat=False)

    # Display the animation
    plt.show()


def main():
    # List of file paths with corresponding agent IDs
    file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2)]#,('agent_3_positions.txt', 3),('agent_4_positions.txt', 4),('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # Initialize an empty dictionary to store all agent positions
    all_agent_positions = {}
    
    task_dir = '/home/benjaminb/Desktop/ros2_ws/src/stl_task_decomposition'
    tasks_yaml_path = os.path.join(task_dir, 'config', 'tasks.yaml')
    tasks = extract_task_info(tasks_yaml_path)
    print(tasks)
    # for file_path, agent_id in file_paths_with_ids:
    #     # Read positions from each file and update the dictionary
    #     agent_positions = read_positions_from_file(file_path, agent_id)
    #     all_agent_positions.update(agent_positions)
    
    # # Animate all agent positions
    # animate_agent_positions(all_agent_positions)

if __name__ == '__main__':
    main()
