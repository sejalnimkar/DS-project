from typing import List
from setuptools import find_packages, setup

# HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''This function is going to return list of requirements'''
    requirements = []
    with open(file_path, encoding='utf-8') as f_obj:
        requirements = f_obj.readlines()
        requirements= [req.replace('\n','') for req in requirements]
        # if HYPHEN_E_DOT in requirements:
        #     requirements.remove('HYPHEN_E_DOT')
        requirements=[req for req in requirements if req and not req.startswith('-e .')]
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Sejal',
author_email='sejalnimkar@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)
