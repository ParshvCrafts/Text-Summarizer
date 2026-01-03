import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
__version__ = "0.0.1"
REPO_NAME = "Text-Summarizer"
AUTHOR_USER_NAME = "Parshv Patel"
SRC_REPO = "text_summarizer"
AUTHOR_EMAIL = "p1a2r3s4h5v6@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for text summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/ParshvCrafts/Text-Summarizer",
    project_urls={
        "Bug Tracker": f"https://github.com/ParshvCrafts/Text-Summarizer/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[],
    license="MIT"
)