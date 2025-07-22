import scipy.optimize
import numpy as np

import cogwheel.gw_utils
import cogwheel.utils
from cogwheel.prior import Prior

lal = cogwheel.utils.import_lal()


class PNCoordinatesPrior2(Prior):
    """
    Implement the coordinates for intrinsic parameters of [Lee, Morisaki
    & Tagoshi 2203.05216].

    These are quite similar except:
    1. We normalize the eigenvectors of the Fisher matrix by the square
      root of their eigenvalue, so the Fisher errorbars in (mu1, mu2)
      are 1/snr
    2. Instead of sampling in s2z we use cums2z, the cumulative of the
      s2z conditional on beta, eta and the fact that |s1z| < 1.
      cums2z = ∫_a^b U(s2z) d{s2z}
      a = s2z(beta, eta, s1z=1)
      b = s2z(beta, eta, s1z=-1).
    """
    DEFAULT_EIGVECS = np.array([[-1.57616411, -0.04111396],
                                [-0.54265283,  0.08432735],
                                [-0.27537869,  0.06914793]])
    standard_params = ['m1', 'm2', 's1z', 's2z']
    range_dic = {'mu1': (-np.inf, np.inf),
                 'mu2': (-np.inf, np.inf),
                 'lnq': None,
                 'cums2z': (0, 1)}

    def __init__(self, f_ref, eigvecs=None, q_min=0.05, **kwargs):
        """
        Parameters
        ----------
        eigvecs : float array of shape (3, 2)
            Fisher matrix eigenvectors, see
            ``.eigvecs_from_reference_waveform_finder()``.

        f_ref : float
            Reference frequency (Hz).

        q_min : float
            Minimum mass ratio

        **kwargs
            Passed to super().__init__()
        """
        if eigvecs is None:
            eigvecs = self.DEFAULT_EIGVECS

        eigvecs = np.asarray(eigvecs)
        if eigvecs.shape != (3, 2):
            raise ValueError('Expecting 2 column-vectors of size 3.')

        self.eigvecs = eigvecs
        self.f_ref = f_ref
        self.range_dic = self.range_dic | {'lnq': (np.log(q_min), 0)}

        super().__init__(f_ref=f_ref, q_min=q_min, **kwargs)

    def transform(self, mu1, mu2, lnq, cums2z):
        """Sampled parameters to standard parameters."""
        q = np.exp(lnq)
        eta = cogwheel.gw_utils.q_to_eta(q)
        delta = (1-q) / (1+q)

        # Must find mass and s1z from mu1, mu2; encoded in v_ref, beta.
        # Solve for v_ref, eliminate the 2.5 PN term that contains beta:
        coeffs = np.r_[self.eigvecs[(0, 1),], ((-mu1, -mu2),)
                      ] @ (self.eigvecs[2, 1], -self.eigvecs[2, 0])

        def objective(v):
            """Function whose root is `v_ref`."""
            # v**3 smoothens the function without changing the root.
            return coeffs @ (self._0pn(v, eta), self._1pn(v, eta), 1) * v**3

        try:
            v_ref = scipy.optimize.brentq(objective, 1e-3, 1)
        except ValueError:  # Unphysical (mu1, mu2) given lnq
            return dict.fromkeys(self.standard_params, np.nan)

        mtot = v_ref**3 / (np.pi * lal.MTSUN_SI * self.f_ref)
        m1 = mtot / (1+q)

        # Now solve for beta:
        pn_1_5 = (mu1 - self.eigvecs[(0, 1), 0] @ (self._0pn(v_ref, eta),
                                                   self._1pn(v_ref, eta))
                  ) / self.eigvecs[2, 0]

        beta = 32/3 * eta * v_ref**2 * pn_1_5 + 4*np.pi
        s2z_min, s2z_max = self._s2z_bounds(beta, eta)
        s2z = s2z_min + cums2z * (s2z_max-s2z_min)

        s1z = ((24/113*beta - (1 - delta - 76/113*eta)*s2z)
               / (1 + delta - 76/113*eta))

        if np.abs(s1z) > 1:  # Unphysical (mu1, mu2) given (lnq, s2z)
            return dict.fromkeys(self.standard_params, np.nan)

        return {'m1': m1,
                'm2': q * m1,
                's1z': s1z,
                's2z': s2z}

    @staticmethod
    def _s2z_bounds(beta, eta):
        """Minimum and maximum s2z consistent with |s1z| < 1."""
        q = cogwheel.gw_utils.eta_to_q(eta)
        delta = (1-q) / (1+q)

        def s2z(s1z):
            return ((24/113*beta - (1 + delta - 76/113*eta)*s1z)
                    / (1 - delta - 76/113*eta))

        s2z_min = max(-1.0, s2z(s1z=1.0))
        s2z_max = min(1.0, s2z(s1z=-1.0))
        assert s2z_min <= s2z_max
        return s2z_min, s2z_max

    def inverse_transform(self, m1, m2, s1z, s2z):
        """Standard parameters to sampled parameters."""
        eta, beta, v_ref = self._eta_beta_vref(m1, m2, s1z, s2z)
        mu1, mu2 = self.eigvecs.T @ (self._0pn(v_ref, eta),
                                     self._1pn(v_ref, eta),
                                     self._1_5pn(v_ref, eta, beta))
        s2z_min, s2z_max = self._s2z_bounds(beta, eta)
        cums2z = (s2z - s2z_min) / (s2z_max - s2z_min)
        return {'mu1': mu1,
                'mu2': mu2,
                'lnq': np.log(m2/m1),
                'cums2z': cums2z}

    def ln_jacobian_determinant(self, m1, m2, s1z, s2z):
        """
        Natural log Jacobian determinant of the inverse transform.

        Returns
        -------
        float : log|∂{mu1, mu2, lnq, s2z} / ∂{m1, m2, s1z, s2z}|
        """
        del s2z
        mchirp = cogwheel.gw_utils.m1m2_to_mchirp(m1, m2)
        eta, beta, v_ref = self._eta_beta_vref(m1, m2, s1z, s2z)
        _, beta0, _ = self._eta_beta_vref(m1, m2, s1z, s2z=0.)

        dbeta_ds1z = beta0 / s1z  # Note s2z=0
        d0pn_dmchirp = -5/3 * self._0pn(v_ref, eta) / mchirp
        d1pn_dmchirp = - self._1pn(v_ref, eta) / mchirp
        d1_5pn_ds1z = 3/32 * v_ref**-2 / eta * dbeta_ds1z

        s2z_min, s2z_max = self._s2z_bounds(beta, eta)

        # We want |∂{mu1, mu2, lnq, cums2z} / ∂{m1, m2, s1z, s2z}|
        # = [1] * [2] * [3]

        # [1] = |∂{mu1, mu2} / ∂{mchirp, s1z}|
        detj1 = np.abs(
            (np.linalg.det(self.eigvecs[(0, 2),]) * d0pn_dmchirp
             + np.linalg.det(self.eigvecs[(1, 2),]) * d1pn_dmchirp)
            * d1_5pn_ds1z)

        # [2] = |∂{mchirp, lnq} / ∂{m1, m2}|
        detj2 = ((m1*m2)**2 * (m1 + m2)) ** -0.2

        # [3] = |∂{cums2z} / ∂{s2z}|
        detj3 = 1 / (s2z_max - s2z_min)

        return np.log(detj1 * detj2 * detj3)

    def lnprior(self, *args, **kwargs):
        """
        Logarithm of the prior in the space of sampled parameters.
        """
        standard_par_dic = self.transform(*args, **kwargs)

        if any(np.isnan(value) for value in standard_par_dic.values()):
            return -np.inf  # Unphysical sampled-parameter values

        return (self.standard_lnprior(**standard_par_dic)
                - self.ln_jacobian_determinant(**standard_par_dic))

    @staticmethod
    def eigvecs_from_reference_waveform_finder(
            reference_waveform_finder):
        """
        Return a float array of shape (3, 2) with the two main
        eigenvectors of the Fisher matrix in the space of the first 3
        coefficients of the post-Newtonian expansion.

        These are the first 2 columns of ``U.T`` in the notation of
        [2203.05216], except that we normalize each eigenvector to have
        norm ``sqrt(eigenvalue)``.
        This can be used as input to ``.__init__()``.
        """
        f_ref = reference_waveform_finder.par_dic_0['f_ref']
        fmin = reference_waveform_finder.event_data.fbounds[0]
        fmax = max(f_ref, cogwheel.gw_utils.isco_frequency(
            reference_waveform_finder.par_dic_0['m1']
            + reference_waveform_finder.par_dic_0['m2']))
        fslice = slice(*np.searchsorted(
            reference_waveform_finder.event_data.frequencies, (fmin, fmax)))
        frequencies = reference_waveform_finder.event_data.frequencies[fslice]

        h_f = reference_waveform_finder.waveform_generator \
            .get_strain_at_detectors(frequencies,
                                     reference_waveform_finder.par_dic_0)
        whitened_amplitude = np.linalg.norm(
            h_f * reference_waveform_finder.event_data.wht_filter[:, fslice],
            axis=0)  # Quadrature sum over detectors
        whitened_amplitude /= np.linalg.norm(whitened_amplitude)
        pn_exponents = 0, 1, -5/3, -3/3, -2/3  # Phase, time, 1PN, 2PN, 2.5PN
        pn_functions = np.power.outer(frequencies / f_ref, pn_exponents)
        weighted_functions = pn_functions * whitened_amplitude[:, np.newaxis]
        full_fisher_mat = weighted_functions.T @ weighted_functions

        marginalize_inds = {0, 1}  # Phase, time
        keep_inds = sorted(set(range(len(pn_exponents))) - marginalize_inds)
        keep_mat = np.eye(len(full_fisher_mat))[:, keep_inds]
        fisher_mat = np.linalg.inv(keep_mat.T
                                   @ np.linalg.inv(full_fisher_mat)
                                   @ keep_mat)
        eigvals, eigvecs = np.linalg.eig(fisher_mat)
        inds = np.argsort(-eigvals)[:2]  # Keep 2 main eigenvectors
        return eigvecs[:, inds] * np.sqrt(eigvals[inds])

    def _eta_beta_vref(self, m1, m2, s1z, s2z):
        """Return auxiliary PN quantities eta, beta and v(f_ref)."""
        mtot = m1 + m2
        eta = m1 * m2 / mtot**2
        chis = (s1z + s2z) / 2
        chia = (s1z - s2z) / 2
        delta = (m1 - m2) / mtot
        beta = 113/12 * (chis + delta*chia - 76/113*eta*chis)
        v_ref = np.cbrt(np.pi * lal.MTSUN_SI * mtot * self.f_ref)
        return eta, beta, v_ref

    @staticmethod
    def _0pn(v, eta):
        """0PN term of the phase"""
        return 3/128 / eta * v**-5

    @staticmethod
    def _1pn(v, eta):
        """1PN term of the phase."""
        return 3/128 * (55/9 + 3715/756/eta) * v**-3

    @staticmethod
    def _1_5pn(v, eta, beta):
        """1.5 PN term of the phase."""
        return 3/32 * (beta - 4*np.pi) / eta * v**-2

    def get_init_dict(self):
        """Return keyword arguments to reproduce the class instance."""
        return {'eigvecs': self.eigvecs,
                'f_ref': self.f_ref,
                'q_min': np.exp(self.range_dic['lnq'][0])}
