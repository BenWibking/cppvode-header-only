#!/usr/bin/env python3
"""Plot scalar Rosenbrock transfer functions on the negative real axis."""

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# Local ROS2S coefficients from include/integrators/rodas.hpp.
GAMMA = 0.292893218813452
A21 = 2.0000000000000036
A31 = 6.828427124746214
A32 = 3.4142135623731007
C21 = -6.828427124746214
C31 = -10.949747468305889
C32 = -7.535533905932761
B1 = 6.828427124746214
B2 = 3.414213562373101
B3 = 1.0


def code_ros2s_r(z):
    d = 1.0 - GAMMA * z
    k1 = GAMMA * z / d
    k2 = GAMMA * (z + (C21 + A21 * z) * k1) / d
    k3 = GAMMA * (z + (C31 + A31 * z) * k1 + (C32 + A32 * z) * k2) / d
    return 1.0 + B1 * k1 + B2 * k2 + B3 * k3


def paper_method_ab_r(z):
    sqrt3 = math.sqrt(3.0)
    a = (1.0 + sqrt3) / 2.0
    gamma = (3.0 + sqrt3) / 6.0
    return (1.0 - a * z) / (1.0 - gamma * z) ** 3


def paper_method_d_r(z):
    return (1.0 - z) / (1.0 - 0.5 * z) ** 4


def paper_method_c_r(z):
    sqrt3 = math.sqrt(3.0)
    gamma = (3.0 + sqrt3) / 6.0
    a31 = (1272.0 - 823.0 * sqrt3) / 354.0
    a32 = 3.0 * (-51.0 + 49.0 * sqrt3) / 59.0
    c21 = -1.0 + 7.0 * sqrt3 / 18.0
    c31 = (25.0 - 13.0 * sqrt3) / 6.0
    c32 = 12.0 - 6.0 * sqrt3
    m1 = (-12089.0 + 5037.0 * sqrt3) / 472.0
    m2 = 9.0 * (344.0 - 135.0 * sqrt3) / 118.0
    m3 = 3.0 * (3.0 - sqrt3) / 4.0
    d = 1.0 - gamma * z
    k1 = gamma * z / d
    k2 = gamma * (z + c21 * k1) / d
    y3 = 1.0 + a31 * k1 + a32 * k2
    k3 = gamma * (z * y3 + c31 * k1 + c32 * k2) / d
    return 1.0 + m1 * k1 + m2 * k2 + m3 * k3


def draw(z_min, out, title):
    z = np.linspace(z_min, 0.0, 2001)
    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    ax.plot(z, code_ros2s_r(z), lw=2.2, label="Code ROS2S")
    ax.plot(z, paper_method_ab_r(z), lw=2.0, ls="--", label="Paper Method A/B")
    ax.plot(z, paper_method_c_r(z), lw=2.0, ls=":", label="Paper Method C")
    ax.plot(z, paper_method_d_r(z), lw=2.0, ls="-.", label="Paper Method D")
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axvline(0.0, color="0.2", lw=0.8)
    ax.set_xlim(z_min, 0.0)
    ax.set_ylim(-0.24, 1.05)
    ax.set_xlabel("z = h lambda")
    ax.set_ylabel("R(z)")
    ax.set_title(title)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    draw(-100.0, out_dir / "rosenbrock_transfer_functions.png",
         "Scalar Transfer Functions on the Negative Real Axis")
    draw(-12.0, out_dir / "rosenbrock_transfer_functions_zoom.png",
         "Scalar Transfer Functions, Near-Origin View")

    roots = np.roots([-0.1213203435596425, 0.12132034355964394, 1.0])
    negative_roots = [root.real for root in roots if abs(root.imag) < 1.0e-12 and root.real < 0.0]
    print("wrote logs/rosenbrock_transfer_functions.png")
    print("wrote logs/rosenbrock_transfer_functions_zoom.png")
    print("code ROS2S R(z) = (1 + 0.12132034355964394 z - "
          "0.1213203435596425 z^2) / (1 - 0.292893218813452 z)^3")
    if negative_roots:
        print(f"code ROS2S negative-real zero near z = {negative_roots[0]:.12g}")


if __name__ == "__main__":
    main()
