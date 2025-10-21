"""Background sampling module entry point."""
from .field import SampleOpt, Transform, IBgField, make_plenoxel_field


def sample(field: IBgField, x_world, dir_world, opt: SampleOpt):
    """Sample the given background field.

    Parameters
    ----------
    field: IBgField
        Field instance implementing :func:`sample_rgb`.
    x_world: np.ndarray
        World-space position where the ray originates.
    dir_world: np.ndarray
        World-space direction of the ray (should be normalised).
    opt: SampleOpt
        Sampling options controlling ray marching.
    """
    return field.sample_rgb(x_world, dir_world, opt)
