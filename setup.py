# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

# Metadata fields
package_name = 'langchain_google_memorystore_redis'  # Corrected package name
author = 'Google Cloud Memorystore'
description = 'Memorystore for Redis integration for LangChain'
long_description = open('README.md').read()  # Assuming you have a README.md
url = 'https://github.com/googleapis/langchain-google-memorystore-redis-python'

# Create your requirements.txt (see previous instructions)
install_requires = [line.strip() for line in open("requirements.txt")]

setup(
    name=package_name,
    version='0.1.0',  # Start with an initial version
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',  # If you have a Markdown README
    author=author,
    url=url,
    # Specify the package directory and find packages within src
    package_dir={'': 'src'},  # Specifies that the package(s) are under src
    packages=find_packages(where='src'),  # Tells setuptools to look for packages in src
    install_requires=install_requires,
    python_requires=">=3.7",  # Adjust minimum Python version as needed
    # Consider including additional package data, classifiers, etc.
)