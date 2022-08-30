from setuptools import setup, find_packages

from instruction_ner import __version__

with open("README.md", mode="r", encoding="utf-8") as fp:
    long_description = fp.read()

setup(
    name="instruction_ner",
    version=__version__,
    description="Unofficial implementation of InstructionNER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Olga Bystrova",
    author_email="bystrovaolgavl@gmail.com",
    license_files=["LICENSE"],
    url="https://github.com/ovbystrova/InstructionNER",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "instruction_ner-prepare-data = instruction_ner.prepare_data:main",
            "instruction_ner-train = instruction_ner.train:main",
            "instruction_ner-evaluate = instruction_ner.evaluate:main",
        ],
    },
    install_requires=open("./requirements/requirements.in", "r").readlines(),
    keywords=["python", "nlp", "deep learning", "ner", "t5"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers"
    ]
)