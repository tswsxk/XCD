from setuptools import setup

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

setup(
    name='XCD',
    version='0.0.1',
    extras_require={
        'test': test_deps,
    },
    install_requires=[
        "numpy",
        "EduData>=0.0.9",
        "pandas",
        "mxnet",
        "longling[ml]==1.3.16"
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
