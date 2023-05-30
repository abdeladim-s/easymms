from pathlib import Path

from setuptools import setup, find_packages


# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')


setup(
    name="easymms",
    version="0.1.3",
    author="Abdeladim Sadiki",
    description="A simple Python package to easily use Meta's Massively Multilingual Speech (MMS) project",
    long_description=long_description,
    ext_modules=[],
    zip_safe=False,
    python_requires=">=3.8",
    packages=find_packages('.'),
    package_dir={'': '.'},
    long_description_content_type="text/markdown",
    license='MIT',
    project_urls={
        'Documentation': 'https://abdeladim-s.github.io/easymms/',
        'Source': 'https://abdeladim-s.github.io/easymms/',
        'Tracker': 'https://abdeladim-s.github.io/easymms/issues',
    },
    install_requires=["fairseq~=0.12.2", "pydub~=0.25.1", "platformdirs==3.5.1", "editdistance", "sox", "dataclasses"],
)
