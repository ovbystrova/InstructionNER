from setuptools import setup


with open("README.md", mode="r", encoding="utf-8") as fp:
    long_description = fp.read()

setup(
    name="instruction_ner",
    version="0.1.1",
    description="Unofficial implementation of InstructionNER",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Olga Bystrova",
    author_email="bystrovaolgavl@gmail.com",
    license_files=["LICENSE"],
    url="https://github.com/ovbystrova/InstructionNER",
    packages=["instruction_ner"],
    entry_points={
        "console_scripts": [
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
        "torch==1.12.0",
        "tensorboard==2.9.1",
        "transformers==4.3.3"
    ],
    keywords=["python", "nlp", "deep learning", "ner", "t5"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers"
    ]
)
