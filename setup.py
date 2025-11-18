from setuptools import setup

setup(
    name='ProMeteo',
    version='1.0.0',
    package_dir={"": "src"},
    py_modules=['core', 'pre_processing', 'frame'],
    install_requires=[
        'numpy',
        'pandas',
    ],
    author='Alessandro Zampella',
    description='ProMeteo (PROcessing of METEOrological Data) is a Python library for preprocessing and manipulating measurements collected by sonic anemometers mounted on meteorological towers.',
    url='https://github.com/aleszamp98/ProMeteo',
)   