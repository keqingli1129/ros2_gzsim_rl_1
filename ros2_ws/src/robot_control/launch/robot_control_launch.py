from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    robot_description = LaunchConfiguration('robot_description')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_description',
            description='Robot description XML (URDF) as a string',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
    ])
