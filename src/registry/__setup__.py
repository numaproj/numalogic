from setuptools import find_packages


def setup_args():
    return dict(
        name="numalogic-test_registry",
        version="0.10.a2",
        packages=find_packages(
            where="src",
            include=["registry", "registry.*"],
            exclude=["tests"],
        ),
        package_dir={"": "src"},
        install_requires=[
            "numalogic",
            "redis[hiredis]>=4.0",
            "mlflow-skinny>2.0",
            "boto3>=1.20",
        ],
    )
