from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    description = fh.read()


setup(
    name="ripple_net",
    version="0.1.0",
    author="Chibuzo Kelechi",
    py_modules=["ripple"],
    author_email="kelechichibuzo@gmail.com",
    description="Text-image search and image tagging library",
    packages=find_packages(),
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/kelechi-c/ripple_net",
    keywords=["pypi", "image search", "datasets", "CLIP", "image tagging"],
    license="Apache 2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    install_requires=[
        "sentence-transformers",
        "faiss-gpu",
        "faiss-cpu",
        "datasets",
        "matplotlib",
        "numpy",
        "transformers",
    ],
    python_requires=">=3.6",
)
