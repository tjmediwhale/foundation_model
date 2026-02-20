from typing import List

from setuptools import setup


def load_requirements(filename: str) -> List[str]:
    with open(filename) as f:
        return f.read().splitlines()


base_requirements = load_requirements("requirements/base.txt")
kornia_requirements = load_requirements("requirements/kornia.txt")

setup(
    install_requires=base_requirements,
    extras_require={"kornia": kornia_requirements},
    use_scm_version=True,
)
