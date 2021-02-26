# Copyright 1999-2021 Gentoo Authors
# Distributed under the terms of the GNU General Public License v2

EAPI=7

PYTHON_COMPAT=( python3_{7..8} )
DISTUTILS_SINGLE_IMPL=1
DISTUTILS_USE_SETUPTOOLS=rdepend

inherit distutils-r1

DESCRIPTION="Machine Learning Enabled Brain Extraction"
HOMEPAGE="https://github.com/Jimmy2027/MLEBE"

LICENSE="GPL-3"
SLOT="0"
#IUSE="-scanner-data -training -pretrained-models"
KEYWORDS=""
REQUIRED_USE="${PYTHON_REQUIRED_USE}"
DEPEND="
	$(python_gen_cond_dep '
	sci-libs/nibabel[${PYTHON_USEDEP}]
	media-libs/opencv[${PYTHON_USEDEP}]
	dev-python/matplotlib[${PYTHON_USEDEP}]
	dev-python/openpyxl[${PYTHON_USEDEP}]
	sci-libs/scikit-image[${PYTHON_USEDEP}]
	sci-libs/pytorch[${PYTHON_USEDEP}]
	dev-python/jsonschema[${PYTHON_USEDEP}]
	dev-python/pandas[${PYTHON_USEDEP}]
	dev-python/numpy[${PYTHON_USEDEP}]
	sci-libs/torchvision[${PYTHON_USEDEP}]
	dev-python/tqdm[${PYTHON_USEDEP}]
	sci-libs/scikit-learn[${PYTHON_USEDEP}]
	dev-python/dominate[${PYTHON_USEDEP}]
	dev-python/xlrd[${PYTHON_USEDEP}]
	')
	sci-libs/torchio[${PYTHON_SINGLE_USEDEP}]
"


RDEPEND="${DEPEND}"

src_unpack() {
	cp -r -L "$DOTGENTOO_PACKAGE_ROOT" "$S"
}

python_install_all() {
	distutils-r1_python_install_all
}
