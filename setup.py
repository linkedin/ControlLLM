import setuptools


setuptools.setup(
    name="ControlLLM",
    package_dir={'': 'src'},
    packages=setuptools.find_namespace_packages(where="src"),
)