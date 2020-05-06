from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = '0.0.2'

requires = [
    'Click',
    'requests',
    'arrow',
    'furl',
    'python-decouple',
    'pandas',
    'pandas-ta',
    'websockets'
]
dev_requires = {}

entry_points = """
    [console_scripts]
    ionomy=cli.cli:cli
"""

setup(
    name='ionomy-python',
    version=VERSION,
    author='DistortedLogic/Memehub',
    author_email='jermeek@gmail.com',
    url='https://github.com/Distortedlogic/ionomy-python',
    license='LICENSE.txt',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
    packages=find_namespace_packages(where='src', exclude=['docs', 'tests*']),
    install_requires=requires,
    extras_require=dev_requires,
    package_dir={'': 'packages'},
    keywords=['hive', 'ionomy', 'library', 'api', 'rpc'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Developers',
    ],
    entry_points=entry_points,
    include_package_data=True
)
