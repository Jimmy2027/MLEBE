def get_masking_func(in_file,
                     workflow_config_path,
                     input_type='anat'):
    from mlebe.training.utils.utils import json_file_to_pyobj

    workflow_config = json_file_to_pyobj(workflow_config_path).workflow_config
    if workflow_config.model_type == '2D':
        from mlebe.masking.predict_mask import predict_mask
        return predict_mask(in_file,
                            workflow_config_path,
                            input_type=input_type)
    elif workflow_config.model_type == '3D':
        from mlebe.masking.predict_mask import predict_mask
        return predict_mask(in_file,
                            workflow_config_path,
                            input_type=input_type)
    else:
        raise NotImplementedError('Model type [{}] is not implemented'.format(workflow_config.model_type))
