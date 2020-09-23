# Copyright 1999-2019 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2

EAPI=7

PYTHON_COMPAT=( python{3_6, 3_7} )

DESCRIPTION="Machine Learning Enabled Brain Extraction"
HOMEPAGE="https://github.com/Jimmy2027/MLEBE"

LICENSE="GPL-3"
SLOT="0"
IUSE="-scanner-data -training -pretrained-models"
KEYWORDS=""

DEPEND=""
RDEPEND="
    sci-libs/nibabel
    media-libs/opencv
    dev-python/matplotlib
    dev-python/openpyxl
    sci-libs/scikits_image
    sci-libs/pytorch
	"
