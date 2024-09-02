from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'stl_task_decomposition'


# do not remove these links
data_files=[
 ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
 ('share/' + package_name, ['package.xml'])]

# add here extra files and directories
data_files += [
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/script', glob('script/*')),
]

setup(
    name=package_name,
    version='0.1.0', 
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benjaminb',
    maintainer_email='bbrandin@kth.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manager_node.py = script.manager_node:main',
            'controller.py = script.controller:main',
            'stop_the_agents.py = script.stop_the_agents:main',
        ],
    },
)

