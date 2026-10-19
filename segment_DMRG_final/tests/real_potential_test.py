import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def V_orbital_from_real_potential(i, x0, w, U0, L_y):
    """
    Orbital-space potential V(i) = <psi_i| V |psi_i> for lowest-Landau-level orbitals
    on a cylinder of circumference L_y (lengths in ell_B), for the real-space potential

        V(x) = (1 - tanh((x - x0*2pi/L_y) / (w*2pi/L_y))) * U0 / 2 .

    Orbital i is centred at X_i = 2 pi i / L_y with |psi_i(x)|^2 = exp(-(x - X_i)^2)/sqrt(pi)
    (y integrated out), so

        V(i) = int dx V(x) exp(-(x - X_i)^2) / sqrt(pi),

    i.e. V(x) convolved with a Gaussian of variance 1/2 (rms width ell_B/sqrt(2)).
    x0 and w are in units of orbital index: without the convolution one would have
    V(i) = (1 - tanh((i - x0)/w)) * U0/2.  For w = 0 the result is U0/2 * erfc(2pi (i-x0)/L_y).

    The convolution is done exactly in Fourier space (F.T. of tanh step x Gaussian):
        V(i) = U0/2 - (U0/pi) int_0^inf dq sin(q d) exp(-q^2/4) a / sinh(a q),
        d = 2pi (i - x0)/L_y,   a = pi * w * 2pi/L_y / 2 .
    The integrand is smooth and Gaussian-decaying, so a midpoint rule is exponentially
    accurate (~1e-15) and costs the same for any w, including w -> 0.

    Parameters
    ----------
    i : int or array of int   orbital index (pass i // nlayer yourself if you fold layers)
    x0, w : step centre and width in orbital-index units (w >= 0)
    U0 : potential scale
    L_y : cylinder circumference in ell_B

    Returns an array of the same shape as i (0-d array for scalar i).
    """
    i = np.asarray(i, dtype=float)
    k = 2 * np.pi / L_y
    d = k * (i - x0)                  # orbital centre minus step centre, in ell_B
    wx = k * w                        # step width in ell_B
    a = 0.5 * np.pi * wx

    # beyond this distance the result is U0 or 0 to < 1e-15: set directly, don't integrate
    cap = 18.0 * wx + 10.0
    out = np.where(d > cap, 0.0, U0)          # saturated value; active entries overwritten below
    flat = d.ravel()
    active = np.flatnonzero(np.abs(flat) <= cap)
    if active.size == 0:
        return out

    da = flat[active]

    # q grid: midpoint rule. Aliasing images sit at 2pi/h +- d in real space and must land
    # where the (smoothed) step is already saturated: >= 18 wx (tanh tail) + 14 (Gaussian)
    h = 2 * np.pi / (np.abs(da).max() + 18.0 * wx + 14.0)
    Q = 14.0                          # exp(-Q^2/4) ~ 5e-22
    q = (np.arange(int(np.ceil(Q / h))) + 0.5) * h

    if a == 0:
        s = 1.0 / q                                          # a/sinh(aq) -> 1/q
    else:
        s = 2 * a * np.exp(-a * q) / (-np.expm1(-2 * a * q))  # = a/sinh(aq), overflow-safe
    wq = np.exp(-q**2 / 4) * s * h

    # chunk over orbitals so the (chunk x n_q) work array stays ~ 2e7 doubles (~160 MB max)
    step = max(1, int(2e7 // q.size))
    res = np.empty(da.size)
    for j in range(0, da.size, step):
        res[j:j + step] = np.sin(da[j:j + step, None] * q) @ wq
    out.flat[active] = U0 / 2 - (U0 / np.pi) * res
    return out

Vo=V_orbital_from_real_potential
U0, w, x0, L_y = 4.0, 15.0, 51.0, 15.0
i = np.arange(0, 270)
V_conv = Vo(i, x0, w, U0, L_y)
V_bare = U0 / 2 * (1 - np.tanh((i - x0) / w))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(i, V_bare, "--", lw=2, color="C3", label=r"$\frac{U_0}{2}\left(1-\tanh\frac{i-x_0}{w}\right)$")
ax[0].plot(i, V_conv, "o", ms=4, color="C0", label="V_orbital_from_real_potenital")
ax[0].set_xlabel("orbital index i")
ax[0].set_ylabel("V(i)")
ax[0].set_title(rf"$U_0={U0:g},\ w={w:g},\ x_0={x0:g},\ L_y={L_y:g}\ell_B$")
ax[0].legend()

ax[1].plot(i, V_conv - V_bare, "o-", ms=3, color="C0")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel("orbital index i")
ax[1].set_ylabel("difference")
ax[1].set_title(rf"convolved $-$ bare (max $|\Delta|={np.abs(V_conv-V_bare).max():.1e}$)")
plt.tight_layout()
plt.savefig("V_orbital.png", dpi=150)
print(np.abs(V_conv - V_bare).max())