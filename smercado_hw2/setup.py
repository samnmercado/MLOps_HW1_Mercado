from setuptools import find_packages, setup

setup(
    name="smercado_hw2",
    packages=find_packages(exclude=["smercado_hw2_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
