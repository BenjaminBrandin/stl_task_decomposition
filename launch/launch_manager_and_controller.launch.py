import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('stl_task_decomposition')

    # Path to the initial conditions YAML file
    yaml_file_path = os.path.join(package_dir, 'config', 'initial_conditions.yaml')

    # Load the YAML file and count the number of agents
    with open(yaml_file_path, 'r') as file:
        initial_conditions = yaml.safe_load(file)
    num_agents = len(initial_conditions['initial_conditions'])

    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')

    # Create the manager node
    manager_node = Node(
        package='stl_task_decomposition',
        executable='manager_node.py',
        name='manager_node',
        output='screen'
    )

    # Create the controller nodes based on the number of agents
    controller_nodes = []
    for i in range(1, num_agents + 1):
        controller_node = Node(
            package='stl_task_decomposition',
            executable='controller.py',
            name=f'controller_node_{i}',
            output='screen',
            parameters=[
                {'robot_name': f'agent{i}'},  # Adjusted robot_name parameter
                {'num_robots': num_agents}
            ]
        )
        controller_nodes.append(controller_node)

    # Create the launch description and add the nodes
    ld = LaunchDescription()
    ld.add_action(manager_node)
    for controller_node in controller_nodes:
        ld.add_action(controller_node)

    return ld
