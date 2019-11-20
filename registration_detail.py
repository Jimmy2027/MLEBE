import itertools
import matplotlib.pyplot as plt

from samri.plotting.maps import contour_slices
from samri.utilities import bids_substitution_iterator
from joblib import Parallel, delayed
import multiprocessing as mp

# This is the autodetect function:
from samri.utilities import bids_autofind_df

num_cores = max(mp.cpu_count()-1,1)

templates = {
	'generic':'/usr/share/mouse-brain-atlases/dsurqec_40micron_masked.nii',
	}


data_dir='/home/hendrik/src/mlebe/training_bids_data/'
bids_df = bids_autofind_df(data_dir,
        path_template='sub-{{subject}}/ses-{{session}}/anat/'\
                'sub-{{subject}}_ses-{{session}}_acq-{{acquisition}}_T2w.nii.gz',
        match_regex='.+sub-(?P<sub>.+)/ses-(?P<ses>.+)/anat/'\
                '.*?_acq-(?P<acquisition>.+)_T2w\.nii\.gz',
        )
#training_bids_data/sub-3839/ses-ofM/anat/sub-3839_ses-ofM_acq-TurboRARE_T2w.nii.gz
print(bids_df)
cmap = plt.get_cmap('tab20').colors

#def func_contour_slices(substitution,file_path,data_dir,spacing):
#	contour_slices(file_path.format(**substitution),
#		alpha=[0.9],
#		colors=cmap[::2],
#		figure_title='Single-Session Fit and Distortion Control\n Subject {} | Session {}'.format(substitution['subject'],substitution['session']),
#		file_template='/usr/share/mouse-brain-atlases/dsurqec_40micron_masked.nii',
#		force_reverse_slice_order=True,
#		levels_percentile=[79],
#		ratio=[5,5],
#		slice_spacing=spacing,
#		save_as='{}/registration_detail/{}_{}.pdf'.format(data_dir,substitution['subject'],substitution['session']),
#		)
def anat_contour_slices(substitution,file_path,data_dir,spacing):
	contour_slices(file_path.format(**substitution),
		alpha=[0.9],
		colors=cmap[::2],
		figure_title='Single-Session Fit and Distortion Control\n Subject {} | Session {} | Contrast T2'.format(i[0],substitution['session']),
		file_template=templates[key],
		force_reverse_slice_order=True,
		levels_percentile=[79],
		ratio=[5,5],
		slice_spacing=spacing,
		save_as='{}/manual_overview/{}/{}_{}_T2w.pdf'.format(data_dir,key,i[0],substitution['session']),
		)

spacing = 0.5
#func_path='{data_dir}/preprocessed/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rhp_acq-geEPI_run-0_bold.nii.gz'
#func_substitutions = bids_substitution_iterator(
#	sessions=sessions,
#	subjects=subjects,
#	data_dir=data_dir,
#	validate_for_template=func_path,
#	)
#Parallel(n_jobs=num_cores,verbose=0)(map(delayed(func_contour_slices),
#	func_substitutions,
#	[func_path]*len(func_substitutions),
#	[data_dir]*len(func_substitutions),
#	[spacing]*len(func_substitutions),
#	))

anat_path='{{data_dir}}/preprocessing/{}_collapsed/sub-{{subject}}/ses-{{session}}/anat/sub-{{subject}}_ses-{{session}}_acq-TurboRARElowcov_T2w.nii.gz'.format(key,i[1],runs[i[1]])
anat_substitutions = bids_substitution_iterator(
	sessions=sessions,
	subjects=[i[0]],
	data_dir=data_dir,
	validate_for_template=anat_path,
	)
Parallel(n_jobs=num_cores,verbose=0)(map(delayed(anat_contour_slices),
	anat_substitutions,
	[anat_path]*len(anat_substitutions),
	[data_dir]*len(anat_substitutions),
	[key]*len(anat_substitutions),
	[i]*len(anat_substitutions),
	[spacing]*len(anat_substitutions),
	))

contour_slices(templates[key],
	alpha=[0.6],
	colors=cmap[::2],
	figure_title='Multi-Session Coherence Control\n Subject {} | Task {}'.format(i[0],runs[i[1]]),
	file_template=func_path,
	force_reverse_slice_order=True,
	legend_template='{session} session',
	levels_percentile=[77],
	save_as='{}/registration_detail/{}/coherence_{}_{}.pdf'.format(data_dir,key,i[0],runs[i[1]]),
	slice_spacing=spacing,
	substitutions=func_substitutions,
	)
