from setuptools import find_packages, setup
from typing import List

HYPER_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function will reuturn the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
    return requirements
    

setup(
name='mlproject',
version='0.0.1',
author='Izam Mohammed',
author_email='izamdeveloper1@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)