
from setuptools import setup

package_name = 'human_3d_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@domain.com',
    description='Human 3D detection with YOLO and depth image',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'human_detector_node = human_3d_detector.human_detector_node:main'
        ],
    },
)
