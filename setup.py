from setuptools import setup, find_packages


packages = find_packages()


setup(
	name="MLEBE",
	version="9999",
	description = "Machine Learning Enabled Brain Segmentation",
	author = "Hendrik Klug",
	author_email = "hendrik.klug@gmail.com",
	url = "https://github.com/Jimmy2027/MLEBE",
	keywords = ["fMRI", "data analysis", "deep learning"],
	classifiers = [],
	install_requires = [],
	provides = ["mlebe"],
	packages = packages,
	include_package_data=True,
	extras_require = {},
	)
