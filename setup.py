import setuptools

setuptools.setup(
    name = "imglu",
    version = "1.0.0",
    author = "Andraž Andolšek Peklaj",
    author_email = "andraz.ap@pm.me",
    description = "Immediate mode OpenGL gui library for python.",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = ["imglu"],
    python_requires = ">=3.6",
)