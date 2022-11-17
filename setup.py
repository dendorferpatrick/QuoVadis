from setuptools import setup, find_packages


setup(
    name = "quovadis",
    version = "0.0.1",
    author = "Patrick Dendorfer",
    author_email = "patrick.dendorfer@tum.de",
    include_package_data=True,
    url = "package URL",
    project_urls = {
        "Repository": "https://github.com/dendorferpatrick/QuoVadis",
    },
    package_dir = {"": "src"},
    packages = find_packages(where="src"),
    python_requires = ">=3.6"
)