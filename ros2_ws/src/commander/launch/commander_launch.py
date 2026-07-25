from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='commander',
            executable='dqn_learning',
            name='commander_node',
            output='screen',
            emulate_tty=True,
            additional_env={'PYTHONUNBUFFERED': '1'},
        ),
    ])
