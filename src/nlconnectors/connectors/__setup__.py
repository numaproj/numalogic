from setuptools import find_packages


def setup_args():
    return dict(
        name="numalogic-test_connectors",
        version="0.10.a1",
        packages=find_packages(
            where="src",
            include=["test_connectors", "test_connectors.*"],
            exclude=["tests"],
        ),
        package_dir={"": "src"},
        install_requires=[
            "numalogic",
            "pydruid>=0.6",
            "boto3>=1.20",
            "PyMySQL>=1.0",
        ],
    )
