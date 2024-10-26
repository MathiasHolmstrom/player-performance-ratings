from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="player-performance-ratings",
    version="5.16.2",
    packages=find_packages(),
    install_requires=required,
    author="Mathias Holmstrøm",
    author_email="mathiasholmstom@gmail.com",
    description="Match Predictions based on Player Ratings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hiderdk/player-performance-ratings",
)
