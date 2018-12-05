from distutils.core import setup

setup(
    name='cs229_project',
    description='CS229 project How Real is Real?  comparison of gans and supervised-learning classifiers.',
    version='0.1dev',
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[
        'keras>=2.2.4',
        'tensorflow==1.7.1',
        'matplotlib>=3.0.1',
        'numpy>=1.13.1',
    ]
)