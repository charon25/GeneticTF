import setuptools

with open("README.md", "r", encoding="utf-8") as fi:
    long_description = fi.read()

setuptools.setup(
	name="genetictf",
	version="0.0.1",
	author="Paul 'charon25' Kern",
	description="Manage an evolving population of Keras neural networks",
	long_description=long_description,
    long_description_content_type='text/markdown',
	python_requires=">=3.7",
	url="https://www.github.com/charon25/GeneticTF",
	license="MIT",
	packages=['genetictf'],
	install_requires=[
		'tensorflow>=2.7',
        'matplotlib>=3.4.3',
        'numpy>=1.21.2',
        'tqdm>=4.62.3'
	]
)
