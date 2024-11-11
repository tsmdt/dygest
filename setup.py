from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="dygest",
    author="Thomas Schmidt",
    version='0.1',
    packages=find_packages(),
    license="MIT",
    description="Digest text file with NER and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tsmdt/dygest",
    install_requires=[
        'typer==0.12.5',
        'tiktoken==0.8.0',
        'beautifulsoup4==4.12.3',
        'groq==0.11.0',
        'openai==1.54.3',
        'ollama==0.3.3',
        'flair==0.14.0',
        'tqdm==4.66.6',
        'langdetect==1.0.9',
        'json_repair==0.30.1'
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