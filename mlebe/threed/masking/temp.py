import numpy as np

from dataio.transformation import get_dataset_transformation
from models import get_model
from utils.utils import json_file_to_pyobj
import json

with open('configs/mlebe_config_anat.json') as file:
    config = json.load(file)

json_opts = json_file_to_pyobj('configs/mlebe_config_anat.json')
