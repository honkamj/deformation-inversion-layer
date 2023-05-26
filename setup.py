from setuptools import setup, find_packages


with open("package/DESCRIPTION.md", "r") as fh:
    long_description = fh.read()


setup(
    name='deformation_inversion_layer',
    version='0.0.1',
    license='MIT',
    author="Joel Honkamaa",
    description="Deformation inversion layer is a neural network layer for inverting deformation fields",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/honkamj/deformation_inversion_layer',
    keywords=[
        'deep learning',
        'neural networks',
        'fixed point iteration',
        'deep equilibrium networks',
        'image registration',
        'deformation',
        'coordinate mapping',
        'pytorch'
    ],
    install_requires=['torch>=1.10',],
    python_requires=">=3.6",
)