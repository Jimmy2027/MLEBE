import os

training_save_dir = '/mnt/data/mlebe_data/results/func_w_pretrained/dice_600_2020-02-24' #directory where the training data is stored
classifier_dir = os.path.expanduser('~/.scratch/mlebe/classifiers/cbv_bold') #directory where the classifiers are stored for later use

if not os.path.exists(classifier_dir):
    os.makedirs(classifier_dir)
    print('creating dir', os.path.expanduser(classifier_dir))
os.system('cp {a}/*.pkl {b}'.format(a= training_save_dir, b = classifier_dir))
os.system('cp {a}/1_Step/experiment_description.txt {b}'.format(a = training_save_dir, b = classifier_dir))
os.system('cp {a}/1_Step/*h5 {b}'.format(a = training_save_dir, b = classifier_dir))
os.system('cp {a}/1_Step/*_augmented.npy {b}'.format(a = training_save_dir, b = classifier_dir))