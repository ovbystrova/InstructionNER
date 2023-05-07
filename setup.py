from setuptools import find_packages, setup

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
    install_requires=[
        "dataclasses==0.6",
        "openpyxl==3.0.10",
        "pandas==1.4.3",
        "pyyaml==6.0",
        "SentencePiece==0.1.96",
        "scikit-learn==1.1.2",
        "torch==1.13.1",
        "tensorboard==2.9.1",
        "tokenizers==0.13.2",
        "transformers==4.27.1",
    ],
    keywords=["python", "nlp", "deep learning", "ner", "t5"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
    ],
)
