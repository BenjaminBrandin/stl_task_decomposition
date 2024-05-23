import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():
    # Get the package directories
    stl_task_decomposition_pkg_dir = get_package_share_directory('stl_task_decomposition')
    sml_nexus_description_pkg_dir = get_package_share_directory('sml_nexus_description')

    # Read initial conditions from YAML file
    yaml_file_path = os.path.join(stl_task_decomposition_pkg_dir, 'config', 'initial_conditions.yaml')
    with open(yaml_file_path, 'r') as file:
        initial_conditions = yaml.safe_load(file)

    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    verbose_arg = DeclareLaunchArgument('verbose', default_value='false', description='Set to true to enable verbose output')
    world = LaunchConfiguration('world')

    # Specify xacro file paths
    robot_xacro_subpath = 'description/urdf/sml_nexus.xacro'
    world_file_subpath = 'worlds/mocap_world.world'

    # Create launch description
    ld = LaunchDescription()

    # Add actions for the first launch file
    ld.add_action(use_sim_time)
    ld.add_action(verbose_arg)

    manager_node = Node(
        package='stl_task_decomposition',
        executable='manager_node.py',
        name='manager_node',
        output='screen'
    )
    ld.add_action(manager_node)

    num_agents = len(initial_conditions['initial_conditions'])
    controller_nodes = []
    for i in range(1, num_agents + 1):
        controller_node = Node(
            package='stl_task_decomposition',
            executable='controller.py',
            name=f'controller_node_{i}',
            output='screen',
            parameters=[
                {'robot_name': f'agent{i}'},
                {'num_robots': num_agents}
            ]
        )
        controller_nodes.append(controller_node)
        ld.add_action(controller_node)

    # Add actions for the second launch file
    declare_world_cmd = DeclareLaunchArgument(
        name='world',
        default_value=os.path.join(sml_nexus_description_pkg_dir, world_file_subpath),
        description='Full path to the world model file to load'
    )
    ld.add_action(declare_world_cmd)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments=[('verbose', 'true'), ('world', world)],
    )
    ld.add_action(gazebo)

    for agent_name, pose in initial_conditions['initial_conditions'].items():
        xacro_file = os.path.join(sml_nexus_description_pkg_dir, robot_xacro_subpath)
        robot_description_config = xacro.process_file(xacro_file, mappings={"namespace": agent_name})
        robot_desc = robot_description_config.toxml()

        args = [
            '-topic', agent_name+'/robot_description',
            '-entity', agent_name,
            '-x', str(pose['x']),
            '-y', str(pose['y']),
            '-z', str(pose['z']),
            '-Y', str(pose['yaw']),
        ]

        node = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=args,
            output='screen',
            condition=IfCondition(LaunchConfiguration('verbose'))
        )
        ld.add_action(node)

        remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
        node_robot_state_publisher = Node(
            package='robot_state_publisher',
            namespace=agent_name,
            executable='robot_state_publisher',
            output='screen',
            remappings=remappings,
            parameters=[{'robot_description': robot_desc}],
        )
        ld.add_action(node_robot_state_publisher)

    return ld
