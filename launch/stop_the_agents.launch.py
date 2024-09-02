import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    # Specify the name of the package
    pkg_name = 'stl_task_decomposition'
    config_folder_subpath = "config"
    
    # Reading initial conditions from YAML file
    yaml_file_path = os.path.join(get_package_share_directory(pkg_name), config_folder_subpath, 'initial_conditions.yaml')
    with open(yaml_file_path, 'r') as file:
        initial_conditions = yaml.safe_load(file)
    
    num_agents = len(initial_conditions['initial_conditions'])
    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the robot_name and num_agents parameters
    num_agents_arg = DeclareLaunchArgument('num_agents', default_value=str(num_agents), description='Number of agents')
    # Add the robot_name and num_agents arguments to the launch description
    ld.add_action(num_agents_arg)
    
    

    # Launch the manager node
    stop_agents = Node(
        package='stl_task_decomposition',
        executable='stop_the_agents.py',
        name='stop_agents',
        output='screen',
        emulate_tty= True,
        parameters=[{'num_robots': num_agents}],
    )
    ld.add_action(stop_agents)
    
    return ld
