from setuptools import setup, find_packages

setup(name='callgpt',
      version='0.2',
      description="Library for running few-shot inference on GPT models",
      packages=["gptinference"],
      install_requires=[
          'openai==1.12.0'
      ],
      license='Apache License 2.0',
      long_description=open('README.md').read(),
)
