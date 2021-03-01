from setuptools import setup

setup(
    name='torcharrow',
    version='0.1.0',    
    description='Panda inspired, Arrow compatble Dataframes for PyTorch',
    url='https://github.com/...',
    author='Facebook',
    author_email='shudson@anl.gov',
    license='BSD 2-clause',
    packages=['torcharrow'],
    install_requires=['arrow',
                      'numpy',
                      'pandas'                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.5',
    ],
)