from setuptools import find_packages, setup

exec(open("abile/version.py").read())

setup(
    name="abile",
    author="Sam Duffield",
    author_email="s@mduffield.com",
    url="https://github.com/SamDuffield/bayesian-rating",
    description="state-space model perspective on rating systems (pairwise comparisons)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["jax", "jaxlib", "scipy", "ghq"],
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    platforms="any",
    version=__version__,
)
