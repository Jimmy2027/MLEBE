from mlebe.training.dataio.transformation.transforms import Transformations


def get_dataset_transformation(name, opts=None, max_output_channels=None, verbose=False):
    '''
    :param opts: augmentation parameters
    '''
    # Build the transformation object and initialise the augmentation parameters
    trans_obj = Transformations(name)
    if opts: trans_obj.initialise(opts, max_output_channels)

    if verbose:
        # Print the input options
        trans_obj.print()

    # Returns a dictionary of transformations
    return trans_obj.get_transformation()
