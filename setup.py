from setuptools import setup, find_packages

setup(
    name="mmhdmap",                
    version="0.1.0",                
    author="guyupan",             
    author_email="guyupan1911@gmail.com",  
    description="A simple example", 
    packages=find_packages(),        
    install_requires=[              
        "numpy>=1.20.0",
        "requests"
    ],
    python_requires=">=3.8",       
)
