import io
import os
from setuptools import setup, find_packages


# with open('requirements.txt') as f:
#   required = f.read().splitlines()

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
            os.path.join(os.path.dirname(__file__), *paths),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="player-performance-ratings",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    author="Mathias Holmstrøm",
    author_email="mathiasholmstom@gmail.com",
    description="Match Predictions based on Player Ratings",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hiderdk/player-performance-ratings",
)
