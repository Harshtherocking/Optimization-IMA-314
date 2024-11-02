from setuptools import find_packages, setup

with open("./README.md", "r") as file : 
    long_description = file.read()

setup (
        name = "ima-314",
        version = "0.0.1",
        long_description = long_description,
        package_dir={"":"./algorithms"},
        packages = find_packages(where = "./algorithms"),
        url = "https://github.com/Harshtherocking/Optimization-IMA-314/",
        author = "Harsh & Ankur",
        license= "MIT"
        )