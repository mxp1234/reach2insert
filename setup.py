from setuptools import setup, find_packages

setup(
    name='see_to_reach_feel_to_insert',
    version='0.1.0',
    description='Two-stage Peg-in-Hole: Diffusion Policy (approach) + HIL-SERL (insert)',
    author='',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'opencv-python',
        'requests',
        'pynput',
    ],
)
