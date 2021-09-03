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
    'scipy',
    'matplotlib',
    'PyGeometry-z6',
    'zuper-commons-z6>=6.0.19',
    'quickapp-z6>=6,<7',
    'compmake-z6>=6.0.8,<7',
    'reprep-z6>=6.0.3,<7',
    'networkx>=2.4',
    'zuper-typing-z6>=6.1',
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
    entry_points={"console_scripts": ["dg-demo = games_zoo:dg_demo",
                                      "crash-exp = crash:run_crashing_experiments"]},
    install_requires=install_requires,
)
