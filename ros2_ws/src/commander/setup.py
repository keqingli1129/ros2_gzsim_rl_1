from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'commander'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robomania',
    maintainer_email='robomania@todo.todo',
    description='DQN cart-pole balancing agent',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dqn_learning = commander.dqn_learning:main',
        ],
    },
)
