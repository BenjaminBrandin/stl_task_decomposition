from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import yaml


def generate_launch_description(): 
    # Path to the YAML file
    yaml_file_path = PathJoinSubstitution([FindPackageShare('yaml_files'), 'PATH/TO/YOUR/FILE.YAML']) 

    # Load the YAML file into a dictionary
    with open(yaml_file_path.perform(None), 'r') as file:
        robot_positions = yaml.safe_load(file)

    # Declare arguments
    use_sim_time = DeclareLaunchArgument('use_sim_time', default_value='true')
    gui = DeclareLaunchArgument('gui', default_value='true')
    headless = DeclareLaunchArgument('headless', default_value='false')
    num_robots = DeclareLaunchArgument('num_robots', default_value='4')

    # Nodes
    empty_world = Node(
        package='gazebo_ros', 
        executable='empty_world', 
        arguments=['-debug', '0', '-gui', LaunchConfiguration('gui'), '-use_sim_time', LaunchConfiguration('use_sim_time'), '-headless', LaunchConfiguration('headless'), '-paused', 'false'],
        output='screen'
    )

    # Robots
    robots = []
    for i in range(1, int(LaunchConfiguration('num_robots').perform(None)) + 1):
        robot_description = Node(
            package='sml_nexus_description',
            executable='sml_nexus_description',
            namespace=f'nexus{i}',
            output='screen'
        )

        urdf_spawner = Node(
            package='gazebo_ros', 
            executable='spawn_entity.py', 
            arguments=['-entity', f'nexus{i}', 
                       '-topic', 'robot_description', 
                       '-x', str(robot_positions[f'state{i}'][0]), 
                       '-y', str(robot_positions[f'state{i}'][1]), 
                       '-z', '0.5'],
            namespace=f'nexus{i}',
            output='screen'
        )

        controller = Node(
            package='sml_nexus_tutorials', 
            executable='controller.py', 
            parameters=[{'robot_name': f'nexus{i}', 'num_robots': LaunchConfiguration('num_robots')}],
            namespace=f'nexus{i}',
            output='screen'
        )

        robots.extend([robot_description, urdf_spawner, controller])

    # Manager
    manager = Node(
        package='sml_nexus_tutorials', 
        executable='manager_node.py', 
        output='screen'
    )

    # Motion capture system simulation
    mocap_simulator = Node(
        package='mocap_simulator', 
        executable='qualisys_simulator', 
        output='screen'
    )

    # Build LaunchDescription
    ld = LaunchDescription([
        use_sim_time,
        gui,
        headless,
        num_robots,
        empty_world,
        *robots,
        manager,
        mocap_simulator
    ])

    return ld