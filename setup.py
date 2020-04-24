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


install_requires = [
    # "zuper-commons-z6>=6.0.2",
    # "oyaml",
    # "pybase64",
    # "PyYAML",
    # "validate_email",
    # "mypy_extensions",
    # "typing_extensions",
    # "nose",
    # "coverage>=1.4.33",
    # "jsonschema",
    # "numpy",
    # "base58<2.0,>=1.0.2",
    # "frozendict",
    # "pytz",
    # "termcolor",
    # "numpy",
]

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
    entry_points={"console_scripts": ["dg-demo = driving_games:dg_demo", ]},
    install_requires=install_requires,
)
