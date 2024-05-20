"""
Define a class ``RelativeBinningSplines`` that can be used to
compute relative binning summary weights.

The implementation is based on
    https://arxiv.org/pdf/2402.11439#subsubsection.3.1.1
"""
import scipy.interpolate
import scipy.sparse
import numpy as np

from cogwheel import utils


class RelativeBinningSplines(utils.JSONMixin):
    # TODO: Integrate this in cogwheel.likelihood.relative_binning
    """Class that implements relative binning compression."""
    def __init__(self, frequencies, fbin=None, pn_phase_tol=None,
                 spline_degree=3):
        """
        Parameters
        ----------
        frequencies: 1-d float array
            RFFT frequencies [Hz], only over the domain where the whitened
            waveform is nonzero.

        fbin: 1-d float array or None
            Array with edges of the frequency bins used for relative
            binning [Hz]. Alternatively, pass `pn_phase_tol`.

        pn_phase_tol: float or None
            Tolerance in the post-Newtonian phase [rad] used for
            defining frequency bins. Alternatively, pass `fbin`.

        spline_degree: int
            Degree of the spline used to interpolate the ratio between
            waveform and reference waveform for relative binning.
        """
        if (fbin is None) == (pn_phase_tol is None):
            raise ValueError('Pass exactly one of `fbin` or `pn_phase_tol`.')

        self.frequencies = frequencies
        self._coefficients = None  # Set by ``._set_splines``
        self._basis_splines = None  # Set by ``._set_splines``

        self._spline_degree = spline_degree

        if pn_phase_tol:
            self.pn_phase_tol = pn_phase_tol
        else:
            self.fbin = fbin

    @property
    def pn_phase_tol(self):
        """
        Tolerance in the post-Newtonian phase [rad] used for defining
        frequency bins.
        Setting this will recompute frequency bins such that across each
        frequency bin the change in the post-Newtonian waveform phase
        with respect to the fiducial waveform is bounded by
        `pn_phase_tol` [rad].
        """
        return self._pn_phase_tol

    @pn_phase_tol.setter
    def pn_phase_tol(self, pn_phase_tol):
        pn_exponents = np.array([-5/3, -2/3, 1])

        fbounds = self.frequencies[[0, -1]]
        pn_coeff_rng = 2*np.pi / np.abs(np.subtract(
            *fbounds[:, np.newaxis] ** pn_exponents))

        f_arr = np.linspace(*fbounds, 10000)

        diff_phase = np.sum([np.sign(exp) * rng * f_arr**exp
                             for rng, exp in zip(pn_coeff_rng, pn_exponents)],
                            axis=0)
        diff_phase -= diff_phase[0]  # Worst case scenario differential phase

        # Construct frequency bins
        nbin = np.ceil(diff_phase[-1] / pn_phase_tol).astype(int)
        diff_phase_arr = np.linspace(0, diff_phase[-1], nbin + 1)
        self.fbin = np.interp(diff_phase_arr, diff_phase, f_arr)
        self._pn_phase_tol = pn_phase_tol

    @property
    def fbin(self):
        """
        Edges of the frequency bins for relative binning [Hz].
        Setting this will automatically round them to fall in the FFT
        array, recompute the splines and summary data, and set
        ``._pn_phase_tol`` to ``None`` to keep logs clean.
        """
        return self._fbin

    @fbin.setter
    def fbin(self, fbin):
        df = self.frequencies[1] - self.frequencies[0]
        fbin_ind = np.unique(np.searchsorted(self.frequencies, fbin - df/2))
        self._fbin = self.frequencies[fbin_ind]  # Bin edges

        self._set_splines()
        self._pn_phase_tol = None  # Erase potentially outdated information

    @property
    def spline_degree(self):
        """
        Integer between 1 and 5, degree of the spline used to
        interpolate waveform ratios. Editing it will automatically
        recompute the splines and summary data.
        """
        return self._spline_degree

    @spline_degree.setter
    def spline_degree(self, spline_degree):
        self._spline_degree = spline_degree
        self._set_splines()

    def _set_splines(self):
        """
        Set attributes `_basis_splines` and `_coefficients`.
        `_basis_splines` is a sparse array of shape `(nbin, nrfft)`
        whose rows are the B-spline basis elements for `fbin` evaluated
        on the FFT grid.
        `_coefficients` is an array of shape `(nbin, nbin)` whose i-th
        row are the B-spline coefficients for a spline that interpolates
        an array of zeros with a one in the i-th place, on `fbin`.
        In other words, `_coefficients @ _basis_splines` is an array of
        shape `(nbin, nrfft)` whose i-th row is a spline that
        interpolates on `fbin` an array of zeros with a one in the i-th
        place; this spline is evaluated on the RFFT grid.
        """
        nbin = len(self.fbin)
        coefficients = np.empty((nbin, nbin))
        for i_bin, y_points in enumerate(np.eye(nbin)):
            # Note knots depend on fbin only, they're always the same
            knots, coeffs, _ = scipy.interpolate.splrep(
                self.fbin, y_points, s=0, k=self.spline_degree)
            coefficients[i_bin] = coeffs[:nbin]
        self._coefficients = coefficients

        nrfft = len(self.frequencies)
        basis_splines = scipy.sparse.lil_matrix((nbin, nrfft))

        frequencies = self.frequencies.copy()
        frequencies[-1] -= 1e-10  # Or last basis_element evaluates to 0

        for i_bin in range(nbin):
            element_knots = knots[i_bin : i_bin + self.spline_degree + 2]
            basis_element = scipy.interpolate.BSpline.basis_element(
                element_knots)
            i_start, i_end = np.searchsorted(frequencies,
                                             element_knots[[0, -1]], 'right')
            basis_splines[i_bin, i_start : i_end] = basis_element(
                frequencies[i_start : i_end])

        self._basis_splines = basis_splines.tocsr()

    def get_summary_weights(self, integrand):
        """
        Return summary data to compute efficiently integrals of the form
            4 integral g(f) r(f) df,
        where r(f) is a smooth function.
        The above integral is approximated by
            summary_weights * r(fbin)
        which is the exact result of replacing `r(f)` by a spline that
        interpolates it at `fbin`.

        Parameters
        ----------
        integrand: array of shape (..., nrfft)
            g(f) in the above notation (the oscillatory part of the
            integrand), array whose last axis corresponds to the FFT
            frequency grid.

        Return
        ------
        summary_weights: array of shape (..., nbin)
            array shaped like `integrand` except the last axis now
            correponds to the frequency bins.
        """
        # Broadcast manually
        *pre_shape, nrfft = integrand.shape
        shape = pre_shape + [len(self.fbin)]
        projected_integrand = np.zeros(shape, dtype=integrand.dtype)
        for i, arr_f in enumerate(integrand.reshape(-1, nrfft)):
            projected_integrand[np.unravel_index(i, pre_shape)] \
                = self._basis_splines @ arr_f
        df = self.frequencies[1] - self.frequencies[0]
        return 4 * df * projected_integrand.dot(self._coefficients.T)
