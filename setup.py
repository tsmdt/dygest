from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dygest",
    author="Thomas Schmidt",
    version='0.3.1',
    packages=find_packages(),
    license="MIT",
    description="DYGEST: Document Insights Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsmdt/dygest",
    install_requires=[
        'typer==0.15.1',
        'tiktoken==0.8.0',
        'beautifulsoup4==4.12.3',
        'flair==0.14.0',
        'tqdm==4.67.1',
        'langdetect==1.0.9',
        'json_repair==0.31.0',
        'litellm==1.54.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT license",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'dygest=dygest.cli:app',
        ],
    },
)