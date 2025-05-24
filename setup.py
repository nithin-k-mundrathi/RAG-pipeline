from setuptools import setup,find_packages

with open('requirements.txt') as f:
    requirements =  f.read().splitlines()

setup(
    name = 'rag-genai',
    version= '0.1',
    author = 'Nithin',
    packages = find_packages(), # detect other packages and import them - src, config, utils.
    install_requires = requirements,
)

