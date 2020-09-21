# Copyright 1999-2019 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2

EAPI=7

PYTHON_COMPAT=( python{3_6, 3_7} )

DESCRIPTION="Registration Workflow and Geometric Space for Small Animal Brain Imaging"
HOMEPAGE="https://github.com/Jimmy2027/mlebe_RepSep"

LICENSE="GPL-3"
SLOT="0"
IUSE="-scanner-data"
KEYWORDS=""

DEPEND=""
RDEPEND="
	app-text/texlive[publishers,science,xetex]
	dev-python/joblib[${PYTHON_USEDEP}]
	dev-python/matplotlib[${PYTHON_USEDEP}]
	dev-python/numpy[${PYTHON_USEDEP}]
	dev-python/pandas[${PYTHON_USEDEP}]
	>=dev-python/seaborn-0.9.0[${PYTHON_USEDEP}]
	>=dev-python/statsmodels-0.9.0[${PYTHON_USEDEP}]
	>=dev-tex/pythontex-0.16[${PYTHON_USEDEP}]
	media-gfx/graphviz
	sci-biology/nilearn[${PYTHON_USEDEP}]
	>=sci-biology/samri-0.3[${PYTHON_USEDEP}]
	sci-libs/scikits_learn[${PYTHON_USEDEP}]
	sci-libs/pybids[${PYTHON_USEDEP}]
	scanner-data? ( sci-biology/samri_data )
	!scanner-data? ( sci-biology/samri_bidsdata )
"
