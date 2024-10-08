import matplotlib.pyplot as plt
import ast
import numpy as np
import random

def read_positions_from_file(file_path, agent_id):
    """Read agent positions from a file and return a dictionary of positions."""
    positions = []
    
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into x and y coordinates
            x, y = map(float, line.strip().split())
            positions.append((x, y))
    
    # Return a dictionary with the agent ID as the key
    return {agent_id: positions}


def plot_agent_positions(agent_positions):
    """Plot the positions of each agent."""
    plt.figure(figsize=(10, 10))

    agent_colors = {1: 'blue', 2: 'orange', 3: 'green', 4: 'red', 5: 'purple', 6: 'brown', 7: 'magenta'}
    
    for agent_id, positions in agent_positions.items():
        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, marker='o', markersize=1, color=agent_colors[agent_id], label=f'Agent {agent_id}')
        
        # Get the last position of the agent
        last_position = positions[-1]
        
        # Add a circle at the last position
        agent_circle = plt.Circle(last_position, 0.05, fill=True, color=agent_colors[agent_id], alpha=1.0)
        plt.gca().add_patch(agent_circle)

        # Add text above the circle
        plt.text(last_position[0], last_position[1] + 0.1, f'Agent {agent_id}', 
                 ha='center', va='bottom', color=agent_colors[agent_id], fontsize=12)

    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Movements')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():

    agent_1 = start_pos = (-0.008157494477927685, -1.0343097448349)
    pos1 = (-0.029219692572951317, -0.9959407448768616)
    pos2 = (-0.008157494477927685, -1.0343097448349)

    x_diff = pos1[0] - pos2[0]
    y_diff = pos1[1] - pos2[1]

    x = pos1[0] + x_diff
    y = pos1[1] + y_diff

    print(f"{x} {y}")


    # # List of file paths with corresponding agent IDs
    # file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2),('agent_3_positions.txt', 3),('agent_4_positions.txt', 4),('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # # Initialize an empty dictionary to store all agent positions
    # all_agent_positions = {}
    
    # for file_path, agent_id in file_paths_with_ids:
    #     # Read positions from each file and update the dictionary
    #     agent_positions = read_positions_from_file(file_path, agent_id)
    #     all_agent_positions.update(agent_positions)
    
    # # # Plot all agent positions
    # # plot_agent_positions(all_agent_positions)

if __name__ == '__main__':
    main()

