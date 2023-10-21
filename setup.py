from setuptools import setup, find_packages

setup(
    name="player-performance-ratings",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # any dependencies your project has
    ],
    author="Mathias Holmstrøm",
    author_email="mathiasholmstom@gmail.com",
    description="Match Predictions based on Player Ratings",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Hiderdk/player-performance-ratings",
)