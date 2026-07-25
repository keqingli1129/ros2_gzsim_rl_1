import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # gz-sim resolves the URDF's `package://robot_description/meshes/...` mesh
    # URIs as `model://robot_description/meshes/...`, which it looks up relative
    # to GZ_SIM_RESOURCE_PATH. Add the share-directory parent of
    # robot_description (i.e. .../share) so "robot_description/meshes/..."
    # resolves under .../share/robot_description/meshes/....
    robot_description_share = get_package_share_directory('robot_description')
    resource_path_parent = os.path.dirname(robot_description_share)
    existing_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    os.environ['GZ_SIM_RESOURCE_PATH'] = os.pathsep.join(
        p for p in [resource_path_parent, existing_resource_path] if p
    )

    robot_description_path = PathJoinSubstitution([
        robot_description_share,
        'robot', 'cart_pole.urdf.xacro',
    ])
    robot_description = Command(['xacro ', robot_description_path])

    world_path = os.path.join(
        get_package_share_directory('robot_launch'), 'worlds', 'robomaster_rale.world')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    robot_control = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('robot_control'), 'launch',
                         'robot_control_launch.py')
        ),
        launch_arguments={'robot_description': robot_description}.items(),
    )

    commander = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('commander'), 'launch',
                         'commander_launch.py')
        )
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'robomaster_rale',
            '-topic', 'robot_description',
            '-name', 'cart_pole',
            '-x', '0', '-y', '0', '-z', '1.225',
        ],
        output='screen',
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/cart_pole/joint/cart_joint/cmd_force'
            '@std_msgs/msg/Float64]gz.msgs.Double',
            # gz-sim nests JointStatePublisher's topic under the world name
            # for a model spawned into a running world (confirmed via
            # `gz topic -l` / `gz topic -i` at runtime), unlike the plain
            # `/model/<model>/...` form documented for ApplyJointForce.
            '/world/robomaster_rale/model/cart_pole/joint_state'
            '@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/world/robomaster_rale/control@ros_gz_interfaces/srv/ControlWorld',
        ],
        remappings=[
            ('/model/cart_pole/joint/cart_joint/cmd_force', '/cart_controller/command'),
            ('/world/robomaster_rale/model/cart_pole/joint_state', '/joint_states'),
        ],
        output='screen',
    )

    robot_state_publisher_source = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_description_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )

    start_commander_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[commander],
        )
    )

    return LaunchDescription([
        gz_sim,
        robot_state_publisher_source,
        bridge,
        spawn_entity,
        robot_control,
        start_commander_after_spawn,
    ])
