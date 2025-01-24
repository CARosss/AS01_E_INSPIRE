#!/usr/bin/env python

from math import *
import sys

class NedCalculator:
    def __init__(self, z, H0=75, WM=0.3, WV=None, verbose=0):
        self.z = z
        self.H0 = H0
        self.WM = WM
        self.WV = WV if WV is not None else 1.0 - WM - 0.4165/(H0*H0)
        self.verbose = verbose
        self.constants()
        self.calculate()

    def constants(self):
        # initialize constants
        self.WR = 0.0        # Omega(radiation)
        self.WK = 0.0        # Omega curvature = 1-Omega(total)
        self.c = 299792.458  # velocity of light in km/sec
        self.Tyr = 977.8     # coefficient for converting 1/H into Gyr
        self.DTT = 0.5       # time from z to now in units of 1/H0
        self.DTT_Gyr = 0.0   # value of DTT in Gyr
        self.age = 0.5       # age of Universe in units of 1/H0
        self.age_Gyr = 0.0   # value of age in Gyr
        self.zage = 0.1      # age of Universe at redshift z in units of 1/H0
        self.zage_Gyr = 0.0  # value of zage in Gyr
        self.DCMR = 0.0      # comoving radial distance in units of c/H0
        self.DCMR_Mpc = 0.0
        self.DCMR_Gyr = 0.0
        self.DA = 0.0        # angular size distance
        self.DA_Mpc = 0.0
        self.DA_Gyr = 0.0
        self.kpc_DA = 0.0
        self.DL = 0.0        # luminosity distance
        self.DL_Mpc = 0.0
        self.DL_Gyr = 0.0    # DL in units of billions of light years
        self.V_Gpc = 0.0
        self.a = 1.0         # 1/(1+z), the scale factor of the Universe
        self.az = 0.5        # 1/(1+z(object))

    def calculate(self):
        h = self.H0 / 100.0
        self.WR = 4.165E-5 / (h * h)   # includes 3 massless neutrino species, T0 = 2.72528
        self.WK = 1 - self.WM - self.WR - self.WV
        self.az = 1.0 / (1 + 1.0 * self.z)
        self.age = 0.0
        n = 1000         # number of points in integrals

        for i in range(n):
            self.a = self.az * (i + 0.5) / n
            adot = sqrt(self.WK + (self.WM / self.a) + (self.WR / (self.a * self.a)) + (self.WV * self.a * self.a))
            self.age += 1.0 / adot

        self.zage = self.az * self.age / n
        self.zage_Gyr = (self.Tyr / self.H0) * self.zage
        self.DTT = 0.0
        self.DCMR = 0.0

        # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
        for i in range(n):
            self.a = self.az + (1 - self.az) * (i + 0.5) / n
            adot = sqrt(self.WK + (self.WM / self.a) + (self.WR / (self.a * self.a)) + (self.WV * self.a * self.a))
            self.DTT += 1.0 / adot
            self.DCMR += 1.0 / (self.a * adot)

        self.DTT = (1.0 - self.az) * self.DTT / n
        self.DCMR = (1.0 - self.az) * self.DCMR / n
        self.age = self.DTT + self.zage
        self.age_Gyr = self.age * (self.Tyr / self.H0)
        self.DTT_Gyr = (self.Tyr / self.H0) * self.DTT
        self.DCMR_Gyr = (self.Tyr / self.H0) * self.DCMR
        self.DCMR_Mpc = (self.c / self.H0) * self.DCMR

        # tangential comoving distance
        ratio = 1.00
        x = sqrt(abs(self.WK)) * self.DCMR
        if x > 0.1:
            if self.WK > 0:
                ratio = 0.5 * (exp(x) - exp(-x)) / x
            else:
                ratio = sin(x) / x
        else:
            y = x * x
            if self.WK < 0: y = -y
            ratio = 1.0 + y / 6.0 + y * y / 120.0

        DCMT = ratio * self.DCMR
        self.DA = self.az * DCMT
        self.DA_Mpc = (self.c / self.H0) * self.DA
        self.kpc_DA = self.DA_Mpc / 206.264806
        self.DA_Gyr = (self.Tyr / self.H0) * self.DA
        self.DL = self.DA / (self.az * self.az)
        self.DL_Mpc = (self.c / self.H0) * self.DL
        self.DL_Gyr = (self.Tyr / self.H0) * self.DL

        # comoving volume computation
        ratio = 1.00
        x = sqrt(abs(self.WK)) * self.DCMR
        if x > 0.1:
            if self.WK > 0:
                ratio = (0.125 * (exp(2.0 * x) - exp(-2.0 * x)) - x / 2.0) / (x * x * x / 3.0)
            else:
                ratio = (x / 2.0 - sin(2.0 * x) / 4.0) / (x * x * x / 3.0)
        else:
            y = x * x
            if self.WK < 0: y = -y
            ratio = 1.0 + y / 5.0 + (2.0 / 105.0) * y * y

        VCM = ratio * self.DCMR * self.DCMR * self.DCMR / 3.0
        self.V_Gpc = 4.0 * pi * ((0.001 * self.c / self.H0) ** 3) * VCM

    def get_age_Gyr(self):
        return self.age_Gyr

    def get_zage_Gyr(self):
        return self.zage_Gyr

    def get_DTT_Gyr(self):
        return self.DTT_Gyr

    def get_DCMR_Mpc(self):
        return self.DCMR_Mpc

    def get_kpc_DA(self):
        return self.kpc_DA

    def get_distance_modulus(self):
        return 5 * log10(self.DL_Mpc * 1e6) - 5

    def get_verbose_output(self):
        if self.verbose == 1:
            return (
                f"For H_o = {self.H0:.1f}, Omega_M = {self.WM:.2f}, Omega_vac = {self.WV:.2f}, z = {self.z:.3f}\n"
                f"It is now {self.age_Gyr:.1f} Gyr since the Big Bang.\n"
                f"The age at redshift z was {self.zage_Gyr:.1f} Gyr.\n"
                f"The light travel time was {self.DTT_Gyr:.1f} Gyr.\n"
                f"The comoving radial distance, which goes into Hubble's law, is {self.DCMR_Mpc:.1f} Mpc or {self.DCMR_Gyr:.1f} Gly.\n"
                f"The comoving volume within redshift z is {self.V_Gpc:.1f} Gpc^3.\n"
                f"The angular size distance D_A is {self.DA_Mpc:.1f} Mpc or {self.DA_Gyr:.1f} Gly.\n"
                f"This gives a scale of {self.kpc_DA:.2f} kpc/arcsec.\n"
                f"The luminosity distance D_L is {self.DL_Mpc:.1f} Mpc or {self.DL_Gyr:.1f} Gly.\n"
                f"The distance modulus, m-M, is {self.get_distance_modulus():.2f}"
            )
        else:
            return f"{self.zage_Gyr:.2f} {self.DCMR_Mpc:.2f} {self.kpc_DA:.2f} {self.get_distance_modulus():.2f}"

# Example usage:
"""ned_calculator = NedCalculator(z=1.0, H0=70, WM=0.3, WV=0.7, verbose=1)
print(ned_calculator.get_verbose_output())
print(ned_calculator.get_age_Gyr())"""
