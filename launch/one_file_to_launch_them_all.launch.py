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
    
    pkg_name_desc = 'sml_nexus_description'
    robot_xacro_subpath = 'description/urdf/sml_nexus.xacro'
    world_file_subpath = 'worlds/mocap_world.world'
    
    # Reading initial conditions from YAML file
    yaml_file_path = os.path.join(get_package_share_directory(pkg_name), config_folder_subpath, 'initial_conditions.yaml')
    with open(yaml_file_path, 'r') as file:
        initial_conditions = yaml.safe_load(file)
    
    num_agents = len(initial_conditions['initial_conditions'])

    # Create the launch description and populate
    ld = LaunchDescription()
    
    world_path = os.path.join(get_package_share_directory(pkg_name_desc), world_file_subpath)
    
    world = LaunchConfiguration('world') # this is a configuration (an input argument) to the launch file
    
    # This is to create a launch file input that can be used from command line to set the world file 
    declare_world_cmd = DeclareLaunchArgument(
	    name='world',
	    default_value=world_path,
	    description='Full path to the world model file to load')
    
    verbose_arg = DeclareLaunchArgument('verbose', 
                                        default_value='false', 
                                        description='Set to true to enable verbose output')
    
    ld.add_action(verbose_arg)
    ld.add_action(declare_world_cmd)
    
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments=[('verbose', 'true'), ('world', world)],
    )
    ld.add_action(gazebo)



    for agent_name, pose in initial_conditions['initial_conditions'].items():
        # Use xacro to process the file with the dynamic namespace mapping for each robot
        xacro_file = os.path.join(get_package_share_directory(pkg_name_desc), robot_xacro_subpath)
        robot_description_config = xacro.process_file(xacro_file, mappings={"namespace": agent_name})  # Set the namespace parameter inside the xacro file
        robot_desc = robot_description_config.toxml()  

        args = [
            '-topic', agent_name+'/robot_description', # this is the topic at which the robot description in published
            '-entity', agent_name, # you can decide the name to give to the entity
            '-x', str(pose['x']),
            '-y', str(pose['y']),
            '-z', str(pose['z']),
            '-Y', str(pose['yaw']),
        ]
        
        # Spawn robot into Gazebo
        node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=args,
            output='screen',
            condition=IfCondition(LaunchConfiguration('verbose'))
        )
        ld.add_action(node)

        remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
        # Publish the robot description XML file in the namespace/robot_description topic so that it can be read by the spawner node above
        node_robot_state_publisher = Node(
            package='robot_state_publisher',
            namespace=agent_name,
            executable='robot_state_publisher',
            output='screen',
            remappings=remappings,
            parameters=[{'robot_description': robot_desc}],
        )
        ld.add_action(node_robot_state_publisher)

    # Launch the manager node
    manager_node = Node(
        package='stl_task_decomposition',
        executable='manager_node.py',
        name='manager_node',
        output='screen',
        emulate_tty= True,
    )
    ld.add_action(manager_node)

    return ld
