plancks_constant = 4.13566733e-15  #(10) eV s
""" Planck's constant 4.13566733(10)e-15 eV s """

electron_volt = 1.602176487e-19 #(40) J / eV
""" Electron volt 1.602176487(40)e-19 (J/eV) """

neutron_mass = 1.00866491597 #(43) u
"""
From NIST Reference on Constants, Units, and Uncertainty
   http://physics.nist.gov/cuu/index.html
neutron mass = 1.008 664 915 97(43) u
"""

atomic_mass_constant = 1.660538782e-27 #(83) kg / u
"""
From NIST Reference on Constants, Units, and Uncertainty
   http://physics.nist.gov/cuu/index.html
atomic mass constant m_u = 1.660 538 782(83) x 10-27 kg
"""

VELOCITY_FACTOR = (plancks_constant*electron_volt
                   / (neutron_mass * atomic_mass_constant)) * 1e10
"""
(plancks_constant*electron_volt
                   / (neutron_mass * atomic_mass_constant)) * 1e10
"""

def neutron_velocity(wavelength):
    """
    Velocity (m/s) <=> wavelength (A)
    lambda = h / p = h (eV) (J/eV) / ( m_n (kg) v (m/s) ) (10^10 A/m)
    Since plancks constant is in eV,
    lambda = (1e10 * h*electron_volt/(neutron_mass/N_A)) / velocity
    """

    return VELOCITY_FACTOR / wavelength

def travel_time(distance, wavelength):
    return 1e8 * distance / neutron_velocity(wavelength) # cm / (m/s) * 1e8 = ns

def get_partition(field_id: str):
    return field_id.split("_", 1)[1]