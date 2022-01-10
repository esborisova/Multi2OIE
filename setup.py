import setuptools 

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().split("\n")

setuptools.setup(
    name="relationextraction",
    version="0.0.1",
    author="Lasse Hansen",
    author_email="lasseh0310@gmail.com",
    description="A library for extracting relations within a text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HLasse/Multi2OIE",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[ ],


    keywords = ['NLP', 'knowledge graphs', 'relation extraction', 'triplets'],
    
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windoiws 10',
        'Environment :: GPU :: NVIDIA CUDA :: 10.0',   
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',

    ],
)