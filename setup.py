from setuptools import setup, find_packages

setup(
    name='chainplot',
    packages=find_packages(),
    version='0.2.1',
    description='A(nother) matplotlib wrapper',
    author='Ben Torvaney',
    author_email='btorvaney@gmail.com',
    url='https://github.com/torvaney/chainplot',
    download_url='https://github.com/torvaney/chainplot/tarball/0.1',
    keywords=['plot', 'ggplot'],
    install_requires=[
        'matplotlib>=1.5.3',
        'numpy>=1.11.2',
        'pandas>=0.19.1',
        'scipy>=0.18.1',
        'adjustText>=0.5.3'
    ],
    classifiers=[],
)
