from setuptools import setup

setup(    
    name="clean",
    version='0.0.1',
    install_requires=['gym', 'pybullet', 'numpy', 'stable-baselines3[extra]',
        'pyyaml']
)
