import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="izza",
    version="0.1.0",
    author="Ismaël Lachheb",
    author_email="ismael.lachheb@protonmail.com",
    description="A Personnal machine learning toolbox",
    long_description="this package contain macro and functions i would like to uses in many context",
    long_description_content_type="text/markdown",
    url="https://github.com/lachhebo/izza", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
