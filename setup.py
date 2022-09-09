from setuptools import setup, find_packages


setup(\
        name="eg_autotomy",\
        install_requires=["numpy>=1.18.4",\
                        "torch==1.5.1",\
                        "mpi4py==3.0.3",\
                        "gym[atari,box2d,classic_control]~=0.15.3",\
                        "pybullet==3.0.7",\
                        "scikit-image==0.19.3",\
                        "matplotlib==3.1.2"],\
        version="0.0",\
        description="Autotomy: subtractive adaptation",\
        author="Rive Sunder",\
        packages=find_packages()
        )
