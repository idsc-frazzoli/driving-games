import distutils.text_file
from pathlib import Path
from typing import List

from setuptools import setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()


install_requires = _parse_requirements("requirements.txt")
extras_require = {"all": _parse_requirements("requirements-extra.txt")}

module = "driving_games"
package = "driving-games"
src = "src"

version = get_version(filename=f"src/{module}/__init__.py")

setup(
    name=package,
    package_dir={"": src},
    packages=[module],
    version=version,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "dg-demo = driving_games:dg_demo",
            "posets-exp = trajectory_games_tests:run_ral_exp",
        ]
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
)
