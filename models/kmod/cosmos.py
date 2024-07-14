import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray, ArrayLike
import os
import pathlib
from scipy import fftpack
from sklearn.utils import shuffle
from typing import Optional, Union, Callable

import torch
from torch import nn
from torch import distributions

from . import utils


def _box_dims_adjuster(
        box_dims,
        inp_shape: Union[tuple, NDArray]
) -> NDArray:
    r""" Adjusts the different types of box_dims input params to a consistent format.
    """
    if isinstance(inp_shape, np.ndarray):
        inp_shape = inp_shape.shape
    if not isinstance(inp_shape, tuple):
        raise ValueError(f"`inp_shape` must be an array or a tuple, not {type(inp_shape)}")

    if box_dims is None:
        return np.array(inp_shape)
    if isinstance(box_dims, int):
        box_dims = float(box_dims)
    if isinstance(box_dims, float):
        return np.array([box_dims for _ in inp_shape])
    if isinstance(box_dims, np.ndarray):
        return box_dims
    raise ValueError(f"`box_dims` of type {type(box_dims)} is not valid")


def _get_k(input_array: NDArray, box_dims):
    r"""
    Get the k values for input array with given dimensions.
    Return k components and magnitudes.
    For internal use.
    """
    # Assuming 2d
    x, y = np.indices(input_array.shape, dtype='int32')
    # Centred k-space frequencies (k=0 is at the centre of the map)
    center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])
    # Scale by box length
    kx = 2. * np.pi * (x - center[0]) / box_dims[0]
    ky = 2. * np.pi * (y - center[1]) / box_dims[1]
    # Magnitudes of k vectors
    k = np.sqrt(kx ** 2 + ky ** 2)
    return [kx, ky], k


def _get_kbins(kbins: Union[int, ArrayLike], box_dims, k) -> NDArray:
    r"""
    Make a list of bin edges if kbins is an integer,
    otherwise return it as it is.
    """
    kmin = 2. * np.pi / min(box_dims)  # Minimum freq is that which fits in the box, scale of box
    kbins = np.linspace(kmin, k.max(), kbins + 1)
    return kbins


def make_power_spectrum(amp: float, b: float) -> Callable[[ArrayLike], ArrayLike]:
    r"""Create a power spectrum function

    .. math::

    y(k) = amp \cdot k^{-b}

    :param amp: Amplitude
    :param b: The power law
    :return: A power law function.
    """
    return lambda k: amp * (k ** -b)


def fft_with_k(
        input_array: NDArray,
        box_dims: Optional[Union[float, ArrayLike]] = None,
        do_adjust_vol: bool = True,
):
    r"""Generate a Fourier transform of the input array, along with the k values corresponding to each cell.

    :param input_array: the array to calculate the power spectrum of. Can be of any dimensions.
    :param box_dims: the dimensions of the box.
        If None, the current box volume is used along all dimensions.
        If it is a float, this is taken as the box length along all dimensions.
        If it is an array-like, the elements are taken as the box length along each axis.
    :param do_adjust_vol: If true, the FT values are adjusted according to the volume.
    :returns: A 4-tuple:
        - The Fourier transform of the input array
        - The $k_x$ values corresponding to each cell
        - The $k_y$ values corresponding to each cell
        - The $|k|$ values corresponding to each cell
    """
    box_dims = _box_dims_adjuster(box_dims, input_array.shape)
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))
    (kx, ky), k = _get_k(ft, box_dims)

    if do_adjust_vol:
        boxvol = np.prod(box_dims)
        pixelsize = boxvol / np.prod(ft.shape)
        amp = pixelsize / np.sqrt(boxvol)
        ft *= amp
    return ft, kx, ky, k


def grf_spectrum_to_x_space(
        ft: NDArray,
        do_real_only: bool = True,
) -> NDArray:
    r"""Transforms GRF spectrum to x-space.

    :param ft: The k-space values of the data.
    :param do_real_only: If True, only the real values are returned. If False, complex values are returned.
    """
    x_space = fftpack.ifftn(fftpack.fftshift(ft))
    if do_real_only:
        return np.real(x_space)
    return x_space


def power_spectrum_nd(
        input_array: NDArray,
        box_dims: Optional[Union[float, ArrayLike]] = None,
):
    r"""
    Calculate the power spectrum of input_array and return it as an n-dimensional array,
    where n is the number of dimensions in input_array
    box_side is the size of the box in comoving Mpc. If this is set to None (default),
    the internal box size is used

    :param input_array: the array to calculate the power spectrum of. Can be of any dimensions.
    :param box_dims: the dimensions of the box.
        If None, the current box volume is used along all dimensions.
        If it is a float, this is taken as the box length along all dimensions.
        If it is an array-like, the elements are taken as the box length along each axis.
    :returns: The power spectrum in the same dimensions as the input array.
    """
    box_dims = _box_dims_adjuster(box_dims, input_array.shape)
    box_dims = [box_dims[0] for _ in input_array.shape]  # NOTE: Why?

    # Transform to Fourier space
    ft = fftpack.fftshift(fftpack.fftn(input_array.astype('float64')))

    # Calculate power
    power_spectrum = np.abs(ft) ** 2.

    # Scale by box volume
    boxvol = np.prod(box_dims)
    pixelsize = boxvol / np.prod(input_array.shape)
    power_spectrum *= pixelsize ** 2. / boxvol

    return power_spectrum


def make_gaussian_random_field(
        n_pix,
        box_dim,
        power_spectrum: Callable[[ArrayLike], ArrayLike],
        random_seed: Optional = None,
):
    r"""Generate a Gaussian random field with the specified power spectrum.

    :param n_pix: The number of pixels along each field dimensions.

    Parameters:
        * dims (tuple): the dimensions of the field in number
            of cells. Can be 2D or 3D.
        * box_dims (float or tuple): the dimensions of the field
            in cMpc.
        * power_spectrum (callable, one parameter): the desired
            spherically-averaged power spectrum of the output.
            Given as a function of k
        * random_seed (int): the seed for the random number generation

    Returns:
        The Gaussian random field as a numpy array
    """
    dims = (n_pix, n_pix)
    box_dim = _box_dims_adjuster(box_dim, dims)
    assert len(box_dim) == len(dims) and len(dims) == 2

    rng = utils.make_random_state(seed=random_seed)
    # Generate map in Fourier space, Gaussian distributed real and imaginary parts
    # (= uniform amplitude, Gaussian phases). This field has P(k) = 1 for all k.
    map_ft_real = rng.normal(loc=0., scale=1., size=dims)
    map_ft_imag = rng.normal(loc=0., scale=1., size=dims)
    map_ft = map_ft_real + 1j * map_ft_imag

    # Get k modes for power spectrum, radially symmetric for homog. + iso. field.
    kx_ky, k = _get_k(map_ft_real, box_dim)  # Get k values given dimensions of field

    # Numerical stability
    # k[np.abs(k) < 1.e-6] = 1.e-6

    # Scale factor
    boxvol = np.prod(box_dim)  # = L^n_dims
    pixelsize = boxvol / (np.prod(map_ft_real.shape))
    scale_factor = pixelsize ** 2 / boxvol

    # Scale Fourier map by power spectrum (e.g. scale by covariance: same as reparameterization trick d_k = mu_k + noise * cov_k)
    map_ft *= np.sqrt(power_spectrum(k) / scale_factor)  # Covariance scales with volume dictated by scale factor?

    return grf_spectrum_to_x_space(
        map_ft,
        do_real_only=True,
    )


def radial_average(input_array, box_dims, kbins):
    r"""Radially average data.

    Parameters:
        * input_array (numpy array): the data array
        * box_dims = None (float or array-like): the dimensions of the
            box. If this is None, the current box volume is used along all
            dimensions. If it is a float, this is taken as the box length
            along all dimensions. If it is an array-like, the elements are
            taken as the box length along each axis.
        * kbins = 10 (integer or array-like): The number of bins,
            or a list containing the bin edges. If an integer is given, the bins
            are logarithmically spaced.

    Returns:
        A tuple with (data, bins, n_modes), where data is an array with the
        averaged data, bins is an array with the bin centers and n_modes is the
        number of modes in each bin

    """
    box_dims = _box_dims_adjuster(box_dims, input_array.shape)
    k_comp, k = _get_k(input_array, box_dims)

    kbins = _get_kbins(kbins, box_dims, k)

    # Bin the data
    dk = (kbins[1:] - kbins[:-1]) / 2.

    # Total power in each bin (weights are k-space cell densities from input array)
    outdata, _ = np.histogram(k.flatten(), bins=kbins, weights=input_array.flatten())

    # Number of modes in each bin
    n_modes = np.histogram(k.flatten(), bins=kbins)[0].astype('float')
    outdata /= n_modes

    return outdata, kbins[:-1] + dk, n_modes


def power_spectrum_1d(input_array_nd: NDArray, kbins, box_dims):
    r"""
    Calculate the spherically averaged power spectrum of an array
    and return it as a one-dimensional array.

    Parameters:
        * input_array_nd (numpy array): the data array
        * kbins = 100 (integer or array-like): The number of bins,
            or a list containing the bin edges. If an integer is given, the bins
            are logarithmically spaced.
        * box_dims = None (float or array-like): the dimensions of the
            box. If this is None, the current box volume is used along all
            dimensions. If it is a float, this is taken as the box length
            along all dimensions. If it is an array-like, the elements are
            taken as the box length along each axis.
        * return_n_modes = False (bool): if true, also return the
            number of modes in each bin

    Returns:
        A tuple with (Pk, bins), where Pk is an array with the
        power spectrum and bins is an array with the k bin centers.
    """
    box_dims = _box_dims_adjuster(box_dims, input_array_nd.shape)
    input_array = power_spectrum_nd(input_array_nd, box_dims=box_dims)

    power_spectrum, bins, n_modes = radial_average(
        input_array,
        kbins=kbins,
        box_dims=box_dims
    )
    return power_spectrum, bins, n_modes


class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, mask, prior):
        super(RealNVP, self).__init__()
        # Base distribution, a data-dimensional Gaussian
        self.prior = prior
        # Masks are not to be optimised
        self.mask = nn.Parameter(mask, requires_grad=False)
        # The s and t nets that parameterise the scale and shift
        # change of variables according to inputs, here we are
        # duplicating the networks for each layer.
        self.t = torch.nn.ModuleList(
            [net_t() for _ in range(len(self.mask))]
        )
        self.s = torch.nn.ModuleList(
            [net_s() for _ in range(len(self.mask))]
        )

    def reverse(self, z):
        # Map from Gaussian distributed z to data x
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def forward(self, x):
        # Map from data x to Gaussian distributed z
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        # Calculate log-probability of x under flow distribution
        z, log_det_J = self.forward(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, n):
        # Sample n points from flow distribution fit to data
        z = self.prior.sample((n, 1))
        x = self.reverse(z)
        return x


def make_k_grf_param_dataset(
        grf: NDArray,
        ab: NDArray,
        box_dims: Optional = None,
        seed: Optional[int] = None,
) -> NDArray:
    r"""
    :param grf: A collection of GRFs in the form of a numpy array: (batch_size, x_size, y_size)
    :param ab: A collection of power spectrum parameters corresponding to the GRFs collection (same batch size).
    :param box_dims: The dimensions of the box volume.
    :param seed: A randomness seed. If None, no data is not shuffled.
    :returns: An array of shape (batch_size, 5), such that each element is of the form (k_x, k_y, |ft|, A, B).
    """
    res = list()
    for a_grf, (a, b) in zip(grf, ab):
        ft, kx, ky, _ = fft_with_k(a_grf, box_dims=box_dims, do_adjust_vol=True)
        ft_abs = np.abs(ft)
        res.append(np.array([
            kx.reshape(-1),
            ky.reshape(-1),
            ft_abs.reshape(-1),
            a * np.ones((kx.size, )),
            b * np.ones((kx.size, )),
        ]).T)
    res = np.vstack(res)
    if seed is None:
        return res
    return shuffle(res, random_state=seed)


def analytical_grf_probability(
        grf: NDArray,
        a: float,
        b: float,
        box_dims: Optional[Union[float, ArrayLike]] = None,
        do_exact_weights: bool = False,
) -> float:
    r"""Calculates the analytical probability of measuring a certain GRF given power spectrum parameters a, b.

    .. math::

        \mathbb{P}(GRF|A, B) = \prod\limits_{i, j=1}^{N}
        \frac
            {1}
            {\sqrt{2\pi\, |A\cdot |\vec{k}_{i, j}|^{-B} |}}
        \large{e^{\frac
            {-{\delta(\vec{k}_{i, j})}^2}
            {2 A \cdot |\vec{k}_{i, j}|^{-B}}
        }}
        :label: prob

    This probability is calculated for the following power spectrum:

    .. math:: a \cdot k^{-b}
       :label: power

    :param grf: The GRF in x-space
    :param a: The amplitude of the power spectrum in :eq:`power`.
    :param b: The power-parameter of the power spectrum in :eq:`power`.
    :param box_dims: the dimensions of the box.
        If None, the current box volume is used along all dimensions.
        If it is a float, this is taken as the box length along all dimensions.
        If it is an array-like, the elements are taken as the box length along each axis.
    :param do_exact_weights: If False, removes the :math:`2\pi` normalization factor.
    Used to avoid numbers exploding to zero.
    :returns: The probability of the GRF being measured given the power spectrum.
    """
    box_dims = _box_dims_adjuster(box_dims, grf.shape)

    f_expected = make_power_spectrum(amp=a, b=b)
    ft, _, _, k = fft_with_k(grf, box_dims=box_dims, do_adjust_vol=True)

    n_k = np.unique(k).size - 2
    # n_k = int(np.log(n_k))
    # n_k = int(np.sqrt(n_k))

    calc_spec = np.abs(ft) ** 2
    # calc_spec = np.sqrt(calc_spec)

    p_spectrum = f_expected(k)
    # calc_spec, k_bins_center, mult = power_spectrum_1d(grf, kbins=n_k, box_dims=box_dims)
    # mask = mult > 0
    # p_spectrum = f_expected(k_bins_center)
    # p_spectrum = np.where(mask, f_expected(k_bins_center), 1.0)
    # p_spectrum = np.sqrt(p_spectrum)
    # p_spectrum = p_spectrum ** 2
    # calc_spec = np.where(mask, np.abs(calc_spec) ** 2, 0.0)

    res = calc_spec / p_spectrum
    res += np.log(p_spectrum)
    if do_exact_weights:
        res += np.log(2 * np.pi)
    # res *= mult
    res = -np.sum(res) / 2
    return res


if __name__ == "__main__":
    ...
