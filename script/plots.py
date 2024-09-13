import matplotlib.pyplot as plt
import ast

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
    plt.figure(figsize=(10, 6))
    
    for agent_id, positions in agent_positions.items():
        x_coords, y_coords = zip(*positions)
        plt.plot(x_coords, y_coords, marker='o', label=f'Agent {agent_id}')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Movements')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # List of file paths with corresponding agent IDs
    file_paths_with_ids = [('agent_1_positions.txt', 1),('agent_2_positions.txt', 2),('agent_3_positions.txt', 3),('agent_4_positions.txt', 4)]#,('agent_5_positions.txt', 5),('agent_6_positions.txt', 6),('agent_7_positions.txt', 7)]
    
    # Initialize an empty dictionary to store all agent positions
    all_agent_positions = {}
    
    for file_path, agent_id in file_paths_with_ids:
        # Read positions from each file and update the dictionary
        agent_positions = read_positions_from_file(file_path, agent_id)
        all_agent_positions.update(agent_positions)
    
    # Plot all agent positions
    plot_agent_positions(all_agent_positions)

if __name__ == '__main__':
    main()

