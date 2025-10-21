from setuptools import setup, find_packages

setup(
    name="mmhdmap",                # 包的名字
    version="0.1.0",                 # 版本号
    author="Your Name",              # 作者
    author_email="you@example.com",  # 作者邮箱
    description="A simple example",  # 简短描述
    packages=find_packages(),        # 自动发现项目中的包
    install_requires=[               # 依赖
        "numpy>=1.20.0",
        "requests"
    ],
    python_requires=">=3.8",         # Python 版本要求
)
