import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='deeplab',
    version='1.0.0',
    author='Kai KATSUMATA',
    description='Pytorch implementation for Deeplab',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/raven38/deeplab-pytorch',
    packages=setuptools.find_packages(),
    pakcage_data={'deeplab', ['cocostuff164k.yaml']},
    classifiers=(
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ),
    entry_points={
        'console_scripts': [
            'deeplab = deeplab.eval:main',
        ],
    },
    install_requires=[
        'numpy',
        'torch>1.2.0',
        'torchvision',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'click',
        'scipy',
        'pyyaml',
        'tensorflow',
        'omegaconf',
        'addict',
        'black',
        'joblib',
        'torchnet',
    ]
)
