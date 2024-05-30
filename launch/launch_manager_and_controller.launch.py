import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

import xacro

def generate_launch_description():
    # Specify the name of the package and path to xacro file within the package
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

    for agent_name in initial_conditions['initial_conditions'].keys():
        robot_name_arg = DeclareLaunchArgument(agent_name, default_value=agent_name, description=f'Name of {agent_name}')
        ld.add_action(robot_name_arg)

    # Add the robot_name and num_agents arguments to the launch description
    ld.add_action(num_agents_arg)
    
    # Define list to store controller nodes for later sequential launching
    controller_nodes = []
    # Spawn robots into Gazebo and start their controller nodes
    for agent_name in initial_conditions['initial_conditions'].keys():
        # Create controller nodes
        controller_node = Node(
            package='stl_task_decomposition',
            executable='controller.py',
            name=f'{agent_name}',
            output='screen',
            emulate_tty= True,
            parameters=[{'robot_name': agent_name, 'num_robots': num_agents}],
        )
        controller_nodes.append(controller_node)

    # Add all controller nodes to the launch description
    for controller_node in controller_nodes:
        ld.add_action(controller_node)

    # Start the manager node after all controller nodes have been launched
    # manager_node = Node(
    #     package='stl_task_decomposition',
    #     executable='manager_node.py',
    #     name='manager_node',
    #     output='screen',
    #     emulate_tty= True,
    # )
    # ld.add_action(manager_node)

    return ld
