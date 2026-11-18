# -*- coding: utf-8 -*-
"""Two-compartment CA1 pyramidal cell (type 1) with a lamotrigine (LTG) effect on sodium inactivation.

Model
    A soma + dendrite pair (coupling conductance ``g_c``, soma fraction ``p``) with
    voltage- and calcium-dependent channels:
      * soma     : Na, KDR, CaL, AHP (SK-like), AHP_BK, M-current, KNa
      * dendrite : Na, KDR, KA, CaT, CaR, h, AHP, AHP_BK, M-current
      * synapses : AMPA, GABA and NMDA (computed from the presynaptic potentials).
    Time is in ms and potentials in mV (``dt`` = 1/2048 ms by default).

Lamotrigine
    The drug concentration ``C_LTG`` shifts the Na inactivation curve ``h_inf`` towards
    more negative potentials by ``theta_shift = V_max * C_LTG / (V50 + C_LTG)`` (mV),
    a Michaelis-Menten relation. At ``C_LTG = 0`` the shift is 0.

Slow Na inactivation gate ``s``
    This version also has a second inactivation variable ``s`` (soma ``y[25]``, dendrite
    ``y[26]``) multiplying the Na conductance. It is ACTIVE BY DEFAULT and, unlike the
    LTG shift, it is not zero at ``C_LTG = 0``: it lowers the Na conductance at rest
    (s is about 0.86 at -70 mV). The model without LTG did not have it. Set
    ``use_s_gate = 0`` to force s = 1 and recover that earlier Na current.

State vector ``y`` (``dydx`` holds the derivatives; each entry has a property alias below)
     0 Vs        1 Vd         2 h_s        3 m_KDR_s    4 m_s        5 m_d
     6 h_d       7 m_KDR_d    8 m_KA_d     9 h_KA_d    10 m_CaT_d   11 h_CaT_d
    12 Ca_s     13 m_CaL_s   14 Ca_d      15 m_h_d     16 m_CaR_d   17 h_CaR_d
    18 m_AHP_s  19 m_AHP_d   20 m_AHP_BK_s 21 m_AHP_BK_d 22 m_m_s    23 m_m_d
    24 Na_intra 25 s_s       26 s_d

Items to check (flagged in the code with NOTE, behaviour left unchanged)
    1. Calcium currents appear cross-wired between soma and dendrite (see ``updateParameters``).
    2. NMDA has a single state shared by all presynaptic cells (see ``computeI_NMDA``).
    3. ``rk4`` keeps currents and gate rates fixed during its four stages.
    4. ``tau_s`` is only 1.5-2.5 ms, not slow compared with ``h`` (0.5 ms).
    5. ``h_CaR_inf`` has no slope divisor; two Ca rate functions are singular at one voltage.
    6. BK gate: ``Q`` (1/V) is multiplied by ``V`` in mV, which overflows below about -11 mV.
"""

import numpy as np
from numba import float64, int32
from numba.experimental import jitclass


def get_Variable_Names():
    """Names of the parameters that the GUI lets the user edit for this cell model."""
    return [
        'g_leak', 'g_Na_s', 'g_KDR_s', 'g_CaL_s', 'g_AHP_s', 'g_AHP_BK_s', 'g_m_s', 'g_KNa',
        'g_Na_d', 'g_KDR_d', 'g_KA_d', 'g_CaT_d', 'g_h_d', 'g_CaR_d', 'g_AHP_d', 'g_AHP_BK_d', 'g_m_d',
        'E_leak', 'E_Na', 'E_K', 'E_Ca', 'E_h',
        'g_AMPA', 'E_AMPA', 'g_GABA', 'E_GABA', 'g_NMDA', 'E_NMDA',
        'Cm', 'p', 'g_c', 'PERCENT', 't_start', 't_end', 'istim',
        'C_LTG', 'V_max', 'V50', 'use_s_gate',
    ]  # fmt: skip


def get_ODE_solver():
    """Name of the method that advances the state by one time step."""
    return ["rk4"]


def get_ODE_solver_Time():
    """Name of the method that runs a whole simulation over a time vector."""
    return ["rk4_Time"]


# Numba jitclass field types: every attribute of the class must be declared here.
spec = [
    ('Type', float64),
    ('Layer', float64),
    ('E_leak', float64),
    ('E_Na', float64),
    ('E_K', float64),
    ('E_Ca', float64),
    ('E_h', float64),
    ('g_KDR_s', float64),
    ('g_Na_s', float64),
    ('g_CaL_s', float64),
    ('g_AHP_s', float64),
    ('g_AHP_BK_s', float64),
    ('g_m_s', float64),
    ('g_KNa', float64),
    ('g_Na_d', float64),
    ('g_KDR_d', float64),
    ('g_KA_d', float64),
    ('g_CaT_d', float64),
    ('g_CaR_d', float64),
    ('g_h_d', float64),
    ('g_AHP_d', float64),
    ('g_AHP_BK_d', float64),
    ('g_m_d', float64),
    ('g_AMPA', float64),
    ('g_NMDA', float64),
    ('g_GABA', float64),
    ('E_AMPA', float64),
    ('E_NMDA', float64),
    ('E_GABA', float64),
    ('g_c', float64),
    ('g_leak', float64),
    ('Cm', float64),
    ('p', float64),
    ('Noise', float64),
    ('I_soma_stim', float64),
    ('I_dend_stim', float64),
    ('PERCENT', float64),
    ('I_synSoma', float64),
    ('I_synDend', float64),
    ('I_AMPA_CA1', float64),
    ('I_GABA_CA1', float64),
    ('s_GABA_CA1', float64),
    ('F', float64),
    ('R', float64),
    ('T', float64),
    ('Q', float64),
    ('alpha_Na', float64),
    ('Rpump', float64),
    ('Na_eq', float64),
    ('A_s', float64),
    ('A_d', float64),
    ('Cdur', float64),
    ('deadTime', float64),
    ('Cmax', float64),
    ('preThresh', float64),
    ('alpha', float64),
    ('beta', float64),
    ('lastRelease', float64),
    ('Rinf', float64),
    ('Rtau', float64),
    ('Ra', float64),
    ('R0', float64),
    ('R1', float64),
    ('C', float64),
    ('mg', float64),
    ('dt', float64),
    ('NbODEs', int32),
    ('NbODEs_s_AMPA', int32),
    ('NbODEs_s_GABA', int32),
    ('dydx', float64[:]),
    ('dydx1', float64[:]),
    ('dydx2', float64[:]),
    ('dydx3', float64[:]),
    ('y', float64[:]),
    ('yt', float64[:]),
    ('ds_AMPA1', float64[:]),
    ('ds_AMPA2', float64[:]),
    ('ds_AMPA3', float64[:]),
    ('s_AMPA', float64[:]),
    ('s_AMPAo', float64[:]),
    ('ds_GABA1', float64[:]),
    ('ds_GABA2', float64[:]),
    ('ds_GABA3', float64[:]),
    ('s_GABA', float64[:]),
    ('s_GABAo', float64[:]),
    ('fonc_Ca_s', float64),
    ('fonc_Ca_d', float64),
    ('fonc_ghk_s', float64),
    ('fonc_ghk_d', float64),
    ('h_inf_s', float64),
    ('m_inf_s', float64),
    ('m_KDR_inf_s', float64),
    ('h_inf_d', float64),
    ('m_inf_d', float64),
    ('m_KDR_inf_d', float64),
    ('m_KA_inf_d', float64),
    ('h_KA_inf_d', float64),
    ('tau_m_KA', float64),
    ('tau_h_KA', float64),
    ('m_CaT_inf_d', float64),
    ('h_CaT_inf_d', float64),
    ('m_CaR_inf_d', float64),
    ('h_CaR_inf_d', float64),
    ('tau_m_CaT', float64),
    ('tau_h_CaT', float64),
    ('m_CaL_inf_s', float64),
    ('tau_m_CaL', float64),
    ('m_h_inf_d', float64),
    ('tau_m_h', float64),
    ('m_AHP_inf_s', float64),
    ('tau_m_AHP_s', float64),
    ('m_AHP_inf_d', float64),
    ('tau_m_AHP_d', float64),
    ('m_AHP_BK_inf_s', float64),
    ('tau_m_AHP_BK_s', float64),
    ('m_AHP_BK_inf_d', float64),
    ('tau_m_AHP_BK_d', float64),
    ('m_m_inf_s', float64),
    ('tau_m_m_s', float64),
    ('m_m_inf_d', float64),
    ('tau_m_m_d', float64),
    ('I_Na_s', float64),
    ('I_KDR_s', float64),
    ('I_Na_d', float64),
    ('I_KDR_d', float64),
    ('I_KA_d', float64),
    ('I_CaT_d', float64),
    ('I_CaR_d', float64),
    ('I_CaL_s', float64),
    ('I_h_d', float64),
    ('I_AHP_s', float64),
    ('I_AHP_d', float64),
    ('I_AHP_BK_s', float64),
    ('I_AHP_BK_d', float64),
    ('I_m_s', float64),
    ('I_m_d', float64),
    ('I_leak_s', float64),
    ('I_leak_d', float64),
    ('drive_channel_d', float64),
    ('drive_channel_s', float64),
    ('I_KNa', float64),
    ('T_fonc', float64[:]),
    ('F_fonc', float64[:]),
    ('fifteenpoxer3', float64),
    ('xghk', float64),
    ('qtm_KA', float64),
    ('t_start', float64),
    ('t_end', float64),
    ('istim', float64),
    ('C_LTG', float64),
    ('V_max', float64),
    ('V50', float64),
    ('theta_shift', float64),
    ('s_inf_s', float64),
    ('s_inf_d', float64),
    ('tau_s_s', float64),
    ('tau_s_d', float64),
    ('use_s_gate', float64),
]


@jitclass(spec)
class pyrCellCA1:
    """CA1 pyramidal cell (soma + dendrite) with lamotrigine-dependent Na inactivation (see module docstring)."""

    def __init__(self):
        """Set default parameters, allocate the state vectors and start from the resting state."""
        self.Type = 1
        self.Layer = 1

        # Reversal potentials (mV)
        self.E_leak = -70
        self.E_Na = 60
        self.E_K = -80
        self.E_Ca = 140
        self.E_h = -10

        # Soma conductances
        self.g_KDR_s = 6.0
        self.g_Na_s = 70.0
        self.g_CaL_s = 0.5
        self.g_AHP_s = 0.1
        self.g_AHP_BK_s = 2
        self.g_m_s = 3.1
        self.g_KNa = 0

        # Dendrite conductances
        self.g_Na_d = 10
        self.g_KDR_d = 2
        self.g_KA_d = 55
        self.g_CaT_d = 1
        self.g_CaR_d = 0.75
        self.g_h_d = 0.4
        self.g_AHP_d = 0.1
        self.g_AHP_BK_d = 0.1
        self.g_m_d = 0.1

        # Synapses (previous defaults: g_AMPA 18, g_NMDA 0.35, g_GABA 18, E_GABA -55)
        self.g_AMPA = 8
        self.g_NMDA = 0.15
        self.g_GABA = 25
        self.E_AMPA = 0.0
        self.E_NMDA = 0.0
        self.E_GABA = -75

        # Coupling, leak, capacitance, soma fraction and relative leak noise
        self.g_c = 1
        self.g_leak = 0.18
        self.Cm = 1
        self.p = 0.15
        self.Noise = 0.05

        self.I_soma_stim = 0.0
        self.I_dend_stim = 0.0

        # Lamotrigine: Michaelis-Menten shift of the Na inactivation curve
        self.C_LTG = 0
        self.V_max = 26.20
        self.V50 = 102.29
        self.theta_shift = self.michaelis_menten(self.C_LTG, self.V_max, self.V50)
        # 1: the slow Na inactivation gate s is active, 0: s is forced to 1 (see module docstring)
        self.use_s_gate = 1

        self.PERCENT = 0.05
        self.I_synSoma = 0.0
        self.I_synDend = 0.0
        self.I_AMPA_CA1 = 0.0
        self.I_GABA_CA1 = 0.0
        self.s_GABA_CA1 = 0.0

        # Physical constants and pump parameters
        self.F = 96480
        self.R = 8.315
        self.T = 273.16
        self.Q = self.F / (self.R * self.T)
        self.alpha_Na = 0.01
        self.Rpump = 0.060  # published value: 0.018
        self.Na_eq = 9.5
        self.A_s = 0.015
        self.A_d = 0.035

        # ODE vectors (init_vector also initialises the NMDA variables)
        self.dt = 1.0 / 2048.0
        self.NbODEs = 27
        self.NbODEs_s_AMPA = 1
        self.NbODEs_s_GABA = 1
        self.init_vector()

        # Constants precomputed once
        self.fifteenpoxer3 = np.power(15.0, 3.0)  # 15^3, Na pump half-saturation term
        self.xghk = (0.0853 * (34.0 + 273.16)) / 2.0  # Goldman-Hodgkin-Katz scale for the Ca current
        self.qtm_KA = np.exp(((37.0 - 24.0) / 10.0) * np.log(5.0))  # temperature factor of the KA time constant

        # Default current step used by rk4_Time
        self.t_start = 100
        self.t_end = 1100
        self.istim = 40

        self.setParameters()
        self.updateParameters()

        np.random.seed(1)  # seeds numba's generator so that the leak noise is reproducible

    def init_vector(self):
        """Allocate all state and work vectors (zeros), reset the stimulation and the NMDA variables."""
        self.dydx = np.zeros(self.NbODEs)
        self.dydx1 = np.zeros(self.NbODEs)
        self.dydx2 = np.zeros(self.NbODEs)
        self.dydx3 = np.zeros(self.NbODEs)
        self.y = np.zeros(self.NbODEs)
        self.yt = np.zeros(self.NbODEs)

        self.ds_AMPA1 = np.zeros(self.NbODEs_s_AMPA)
        self.ds_AMPA2 = np.zeros(self.NbODEs_s_AMPA)
        self.ds_AMPA3 = np.zeros(self.NbODEs_s_AMPA)
        self.s_AMPA = np.zeros(self.NbODEs_s_AMPA)
        self.s_AMPAo = np.zeros(self.NbODEs_s_AMPA)

        self.ds_GABA1 = np.zeros(self.NbODEs_s_GABA)
        self.ds_GABA2 = np.zeros(self.NbODEs_s_GABA)
        self.ds_GABA3 = np.zeros(self.NbODEs_s_GABA)
        self.s_GABA = np.zeros(self.NbODEs_s_GABA)
        self.s_GABAo = np.zeros(self.NbODEs_s_GABA)

        self.I_soma_stim = 0.0
        self.I_dend_stim = 0.0

        # NMDA receptor kinetics (a single set of variables, see the note in computeI_NMDA)
        self.Cdur = 1.0
        self.deadTime = 1.0
        self.Cmax = 1.0
        self.preThresh = 0.0
        self.alpha = 0.072
        self.beta = 0.0066
        self.lastRelease = -1.0e99
        self.Rinf = 0.0
        self.Rtau = 0.0
        self.Ra = 0.0
        self.R0 = 0.0
        self.R1 = 0.0
        self.C = 0.0
        self.mg = 1.0

    # ------------------------------------------------------------------
    # Property aliases: readable names for the entries of y (state) and dydx (derivative)
    # ------------------------------------------------------------------
    @property
    def Vs_(self):
        return self.y[0]

    @Vs_.setter
    def Vs_(self, value):
        self.y[0] = value

    @property
    def dVs_dt(self):
        return self.dydx[0]

    @dVs_dt.setter
    def dVs_dt(self, value):
        self.dydx[0] = value

    @property
    def Vd_(self):
        return self.y[1]

    @Vd_.setter
    def Vd_(self, value):
        self.y[1] = value

    @property
    def dVd_dt(self):
        return self.dydx[1]

    @dVd_dt.setter
    def dVd_dt(self, value):
        self.dydx[1] = value

    @property
    def m_s_(self):
        return self.y[4]

    @m_s_.setter
    def m_s_(self, value):
        self.y[4] = value

    @property
    def dm_s_dt(self):
        return self.dydx[4]

    @dm_s_dt.setter
    def dm_s_dt(self, value):
        self.dydx[4] = value

    @property
    def h_s_(self):
        return self.y[2]

    @h_s_.setter
    def h_s_(self, value):
        self.y[2] = value

    @property
    def dh_s_dt(self):
        return self.dydx[2]

    @dh_s_dt.setter
    def dh_s_dt(self, value):
        self.dydx[2] = value

    @property
    def m_KDR_s_(self):
        return self.y[3]

    @m_KDR_s_.setter
    def m_KDR_s_(self, value):
        self.y[3] = value

    @property
    def dm_KDR_s_dt(self):
        return self.dydx[3]

    @dm_KDR_s_dt.setter
    def dm_KDR_s_dt(self, value):
        self.dydx[3] = value

    @property
    def m_d_(self):
        return self.y[5]

    @m_d_.setter
    def m_d_(self, value):
        self.y[5] = value

    @property
    def dm_d_dt(self):
        return self.dydx[5]

    @dm_d_dt.setter
    def dm_d_dt(self, value):
        self.dydx[5] = value

    @property
    def h_d_(self):
        return self.y[6]

    @h_d_.setter
    def h_d_(self, value):
        self.y[6] = value

    @property
    def dh_d_dt(self):
        return self.dydx[6]

    @dh_d_dt.setter
    def dh_d_dt(self, value):
        self.dydx[6] = value

    @property
    def m_KDR_d_(self):
        return self.y[7]

    @m_KDR_d_.setter
    def m_KDR_d_(self, value):
        self.y[7] = value

    @property
    def dm_KDR_d_dt(self):
        return self.dydx[7]

    @dm_KDR_d_dt.setter
    def dm_KDR_d_dt(self, value):
        self.dydx[7] = value

    @property
    def m_KA_d_(self):
        return self.y[8]

    @m_KA_d_.setter
    def m_KA_d_(self, value):
        self.y[8] = value

    @property
    def dm_KA_d_dt(self):
        return self.dydx[8]

    @dm_KA_d_dt.setter
    def dm_KA_d_dt(self, value):
        self.dydx[8] = value

    @property
    def h_KA_d_(self):
        return self.y[9]

    @h_KA_d_.setter
    def h_KA_d_(self, value):
        self.y[9] = value

    @property
    def dh_KA_d_dt(self):
        return self.dydx[9]

    @dh_KA_d_dt.setter
    def dh_KA_d_dt(self, value):
        self.dydx[9] = value

    @property
    def m_CaT_d_(self):
        return self.y[10]

    @m_CaT_d_.setter
    def m_CaT_d_(self, value):
        self.y[10] = value

    @property
    def dm_CaT_d_dt(self):
        return self.dydx[10]

    @dm_CaT_d_dt.setter
    def dm_CaT_d_dt(self, value):
        self.dydx[10] = value

    @property
    def h_CaT_d_(self):
        return self.y[11]

    @h_CaT_d_.setter
    def h_CaT_d_(self, value):
        self.y[11] = value

    @property
    def dh_CaT_d_dt(self):
        return self.dydx[11]

    @dh_CaT_d_dt.setter
    def dh_CaT_d_dt(self, value):
        self.dydx[11] = value

    @property
    def Ca_s_(self):
        return self.y[12]

    @Ca_s_.setter
    def Ca_s_(self, value):
        self.y[12] = value

    @property
    def dCa_s_dt(self):
        return self.dydx[12]

    @dCa_s_dt.setter
    def dCa_s_dt(self, value):
        self.dydx[12] = value

    @property
    def m_CaL_s_(self):
        return self.y[13]

    @m_CaL_s_.setter
    def m_CaL_s_(self, value):
        self.y[13] = value

    @property
    def dm_CaL_s_dt(self):
        return self.dydx[13]

    @dm_CaL_s_dt.setter
    def dm_CaL_s_dt(self, value):
        self.dydx[13] = value

    @property
    def Ca_d_(self):
        return self.y[14]

    @Ca_d_.setter
    def Ca_d_(self, value):
        self.y[14] = value

    @property
    def dCa_d_dt(self):
        return self.dydx[14]

    @dCa_d_dt.setter
    def dCa_d_dt(self, value):
        self.dydx[14] = value

    @property
    def m_h_d_(self):
        return self.y[15]

    @m_h_d_.setter
    def m_h_d_(self, value):
        self.y[15] = value

    @property
    def dm_h_d_dt(self):
        return self.dydx[15]

    @dm_h_d_dt.setter
    def dm_h_d_dt(self, value):
        self.dydx[15] = value

    @property
    def m_CaR_d_(self):
        return self.y[16]

    @m_CaR_d_.setter
    def m_CaR_d_(self, value):
        self.y[16] = value

    @property
    def dm_CaR_d_dt(self):
        return self.dydx[16]

    @dm_CaR_d_dt.setter
    def dm_CaR_d_dt(self, value):
        self.dydx[16] = value

    @property
    def h_CaR_d_(self):
        return self.y[17]

    @h_CaR_d_.setter
    def h_CaR_d_(self, value):
        self.y[17] = value

    @property
    def dh_CaR_d_dt(self):
        return self.dydx[17]

    @dh_CaR_d_dt.setter
    def dh_CaR_d_dt(self, value):
        self.dydx[17] = value

    @property
    def m_AHP_s_(self):
        return self.y[18]

    @m_AHP_s_.setter
    def m_AHP_s_(self, value):
        self.y[18] = value

    @property
    def dm_AHP_s_dt(self):
        return self.dydx[18]

    @dm_AHP_s_dt.setter
    def dm_AHP_s_dt(self, value):
        self.dydx[18] = value

    @property
    def m_AHP_d_(self):
        return self.y[19]

    @m_AHP_d_.setter
    def m_AHP_d_(self, value):
        self.y[19] = value

    @property
    def dm_AHP_d_dt(self):
        return self.dydx[19]

    @dm_AHP_d_dt.setter
    def dm_AHP_d_dt(self, value):
        self.dydx[19] = value

    @property
    def m_AHP_BK_s_(self):
        return self.y[20]

    @m_AHP_BK_s_.setter
    def m_AHP_BK_s_(self, value):
        self.y[20] = value

    @property
    def dm_AHP_BK_s_dt(self):
        return self.dydx[20]

    @dm_AHP_BK_s_dt.setter
    def dm_AHP_BK_s_dt(self, value):
        self.dydx[20] = value

    @property
    def m_AHP_BK_d_(self):
        return self.y[21]

    @m_AHP_BK_d_.setter
    def m_AHP_BK_d_(self, value):
        self.y[21] = value

    @property
    def dm_AHP_BK_d_dt(self):
        return self.dydx[21]

    @dm_AHP_BK_d_dt.setter
    def dm_AHP_BK_d_dt(self, value):
        self.dydx[21] = value

    @property
    def m_m_s_(self):
        return self.y[22]

    @m_m_s_.setter
    def m_m_s_(self, value):
        self.y[22] = value

    @property
    def dm_m_s_dt(self):
        return self.dydx[22]

    @dm_m_s_dt.setter
    def dm_m_s_dt(self, value):
        self.dydx[22] = value

    @property
    def m_m_d_(self):
        return self.y[23]

    @m_m_d_.setter
    def m_m_d_(self, value):
        self.y[23] = value

    @property
    def dm_m_d_dt(self):
        return self.dydx[23]

    @dm_m_d_dt.setter
    def dm_m_d_dt(self, value):
        self.dydx[23] = value

    @property
    def Na_intra_(self):
        return self.y[24]

    @Na_intra_.setter
    def Na_intra_(self, value):
        self.y[24] = value

    @property
    def dNa_intra_dt(self):
        return self.dydx[24]

    @dNa_intra_dt.setter
    def dNa_intra_dt(self, value):
        self.dydx[24] = value

    @property
    def s_s_(self):
        return self.y[25]

    @s_s_.setter
    def s_s_(self, value):
        self.y[25] = value

    @property
    def ds_s_dt(self):
        return self.dydx[25]

    @ds_s_dt.setter
    def ds_s_dt(self, value):
        self.dydx[25] = value

    @property
    def s_d_(self):
        return self.y[26]

    @s_d_.setter
    def s_d_(self, value):
        self.y[26] = value

    @property
    def ds_d_dt(self):
        return self.dydx[26]

    @ds_d_dt.setter
    def ds_d_dt(self, value):
        self.dydx[26] = value

    # ------------------------------------------------------------------
    # Initial state and noise
    # ------------------------------------------------------------------
    def setParameters(self):
        """Set the state vector ``y`` to the resting state (steady-state gates at the leak potential)."""
        self.y[0] = self.E_leak
        self.y[1] = self.E_leak
        self.y[2] = self.h_inf_CA1(self.y[0])
        self.y[3] = self.m_KDR_inf_CA1(self.y[0])
        self.y[4] = self.m_inf_CA1(self.y[0])
        self.y[5] = self.m_inf_CA1(self.y[1])
        self.y[6] = self.h_inf_CA1(self.y[1])
        self.y[7] = self.m_KDR_inf_CA1(self.y[1])
        self.y[8] = self.m_KA_inf_CA1(self.y[1])
        self.y[9] = self.h_KA_inf_CA1(self.y[1])
        self.y[12] = 0.00005
        self.y[14] = 0.00005
        self.y[10] = self.m_CaT_inf_CA1(self.y[1])
        self.y[11] = self.h_CaT_inf_CA1(self.y[1])
        self.y[13] = self.m_CaL_inf_CA1(self.y[1])
        self.y[15] = self.m_h_inf_CA1(self.y[1])
        self.y[16] = self.m_CaR_inf_CA1(self.y[1])
        self.y[17] = self.h_CaR_inf_CA1(self.y[1])
        self.y[18] = self.m_AHP_inf_CA1(self.y[12])
        self.y[19] = self.m_AHP_inf_CA1(self.y[14])
        self.y[20] = self.m_AHP_BK_inf_CA1(self.y[0], self.y[12])
        self.y[21] = self.m_AHP_BK_inf_CA1(self.y[1], self.y[14])
        self.y[22] = self.m_m_inf_CA1(self.y[0])
        self.y[23] = self.m_m_inf_CA1(self.y[1])
        self.y[24] = 9.5
        self.y[25] = self.s_inf_CA1(self.y[0])  # soma
        self.y[26] = self.s_inf_CA1(self.y[1])  # dendrite
        # NOTE: h and s start from values computed with the current theta_shift. After changing
        # C_LTG, call setParameters() again (state reset) so that they start from the new curves.

    def bruitGaussien(self, s, m):
        """One Gaussian random number with standard deviation ``s`` and mean ``m``."""
        return np.random.normal(m, s)

    def I_leak_CA1(self, g_leak, PERCENT, V, E_leak):
        """Leak current with a noisy conductance (standard deviation = PERCENT * g_leak)."""
        return self.bruitGaussien(PERCENT * g_leak, g_leak) * (V - E_leak)

    # ------------------------------------------------------------------
    # Rates and currents at the current state
    # ------------------------------------------------------------------
    def updateParameters(self):
        """Recompute, from the current ``y``, every steady state, time constant and ionic current.

        Called once per time step by ``rk4_Time``; the values are then kept fixed during the
        four stages of ``rk4`` (see the note there).
        """
        # Lamotrigine shift of the Na inactivation curve (used by h_inf_CA1)
        self.theta_shift = self.michaelis_menten(self.C_LTG, self.V_max, self.V50)

        self.s_inf_s = self.s_inf_CA1(self.y[0])
        self.s_inf_d = self.s_inf_CA1(self.y[1])
        self.tau_s_s = self.tau_s_CA1(self.y[0])
        self.tau_s_d = self.tau_s_CA1(self.y[1])
        self.fonc_Ca_s = self.fonc_Ca_CA1(self.y[12])
        self.fonc_Ca_d = self.fonc_Ca_CA1(self.y[14])
        self.fonc_ghk_s = self.fonc_ghk_CA1(self.y[0], self.y[12])
        self.fonc_ghk_d = self.fonc_ghk_CA1(self.y[1], self.y[14])
        self.h_inf_s = self.h_inf_CA1(self.y[0])
        self.m_inf_s = self.m_inf_CA1(self.y[0])
        self.m_KDR_inf_s = self.m_KDR_inf_CA1(self.y[0])
        self.h_inf_d = self.h_inf_CA1(self.y[1])
        self.m_inf_d = self.m_inf_CA1(self.y[1])
        self.m_KDR_inf_d = self.m_KDR_inf_CA1(self.y[1])
        self.m_KA_inf_d = self.m_KA_inf_CA1(self.y[1])
        self.h_KA_inf_d = self.h_KA_inf_CA1(self.y[1])
        self.tau_m_KA = self.tau_m_KA_CA1(self.y[1])
        self.tau_h_KA = self.tau_h_KA_CA1(self.y[1])
        self.m_CaT_inf_d = self.m_CaT_inf_CA1(self.y[1])
        self.h_CaT_inf_d = self.h_CaT_inf_CA1(self.y[1])
        self.m_CaR_inf_d = self.m_CaR_inf_CA1(self.y[1])
        self.h_CaR_inf_d = self.h_CaR_inf_CA1(self.y[1])
        self.tau_m_CaT = self.tau_m_CaT_CA1(self.y[1])
        self.tau_h_CaT = self.tau_h_CaT_CA1(self.y[1])
        # NOTE: the "soma" CaL gate is computed from the dendritic potential y[1]
        # (compare with the AHP and M gates, which use y[0] for the soma).
        self.m_CaL_inf_s = self.m_CaL_inf_CA1(self.y[1])
        self.tau_m_CaL = self.tau_m_CaL_CA1(self.y[1])
        self.m_h_inf_d = self.m_h_inf_CA1(self.y[1])
        self.tau_m_h = self.tau_m_h_CA1(self.y[1])
        self.m_AHP_inf_s = self.m_AHP_inf_CA1(self.y[12])
        self.tau_m_AHP_s = self.tau_m_AHP_CA1(self.y[12])
        self.m_AHP_inf_d = self.m_AHP_inf_CA1(self.y[14])
        self.tau_m_AHP_d = self.tau_m_AHP_CA1(self.y[14])
        self.m_AHP_BK_inf_s = self.m_AHP_BK_inf_CA1(self.y[0], self.y[12])
        self.tau_m_AHP_BK_s = self.tau_m_AHP_BK_CA1(self.y[0], self.y[12])
        self.m_AHP_BK_inf_d = self.m_AHP_BK_inf_CA1(self.y[1], self.y[14])
        self.tau_m_AHP_BK_d = self.tau_m_AHP_BK_CA1(self.y[1], self.y[14])
        self.m_m_inf_s = self.m_m_inf_CA1(self.y[0])
        self.tau_m_m_s = self.tau_m_m_CA1(self.y[0])
        self.m_m_inf_d = self.m_m_inf_CA1(self.y[1])
        self.tau_m_m_d = self.tau_m_m_CA1(self.y[1])

        # Slow Na inactivation gate: use the state variable, or 1 when the gate is switched off
        if self.use_s_gate > 0.5:
            s_gate_s = self.y[25]
            s_gate_d = self.y[26]
        else:
            s_gate_s = 1.0
            s_gate_d = 1.0

        # Ionic currents. Suffix _s = soma, _d = dendrite.
        self.I_Na_s = self.I_Na_CA1(self.g_Na_s, self.y[4], self.y[2], s_gate_s, self.y[0], self.E_Na)
        self.I_KDR_s = self.I_KDR_CA1(self.g_KDR_s, self.y[3], self.y[0], self.E_K)
        self.I_Na_d = self.I_Na_CA1(self.g_Na_d, self.y[5], self.y[6], s_gate_d, self.y[1], self.E_Na)
        self.I_KDR_d = self.I_KDR_CA1(self.g_KDR_d, self.y[7], self.y[1], self.E_K)
        self.I_KA_d = self.I_KA_CA1(self.g_KA_d, self.y[8], self.y[9], self.y[1], self.E_K)
        # NOTE: the dendritic CaT current uses the SOMATIC Ca factors (fonc_Ca_s, fonc_ghk_s), while
        # the somatic CaL current below uses the DENDRITIC ones and y[1]. Check that this is intended.
        self.I_CaT_d = self.I_CaT_CA1(
            self.g_CaT_d, self.y[10], self.y[11], self.fonc_Ca_s, self.fonc_ghk_s, self.y[1], self.E_Ca
        )
        self.I_CaR_d = self.I_CaR_CA1(self.g_CaR_d, self.y[16], self.y[17], self.y[1], self.E_Ca)
        self.I_CaL_s = self.I_CaL_CA1(self.g_CaL_s, self.y[13], self.fonc_Ca_d, self.fonc_ghk_d, self.y[1], self.E_Ca)
        self.I_h_d = self.I_h_CA1(self.g_h_d, self.y[15], self.y[1], self.E_h)
        self.I_AHP_s = self.I_AHP_CA1(self.g_AHP_s, self.y[18], self.y[0], self.E_K)
        self.I_AHP_d = self.I_AHP_CA1(self.g_AHP_d, self.y[19], self.y[1], self.E_K)
        self.I_AHP_BK_s = self.I_AHP_BK_CA1(self.g_AHP_BK_s, self.y[20], self.y[0], self.E_K)
        self.I_AHP_BK_d = self.I_AHP_BK_CA1(self.g_AHP_BK_d, self.y[21], self.y[1], self.E_K)
        self.I_m_s = self.I_m_CA1(self.g_m_s, self.y[22], self.y[0], self.E_K)
        self.I_m_d = self.I_m_CA1(self.g_m_d, self.y[23], self.y[1], self.E_K)
        self.I_leak_s = self.I_leak_CA1(self.g_leak, self.PERCENT, self.y[0], self.E_leak)
        self.I_leak_d = self.I_leak_CA1(self.g_leak, self.PERCENT, self.y[1], self.E_leak)
        # Calcium entry that drives the Ca concentrations (dendrite: CaT + CaR, soma: CaL)
        self.drive_channel_d = self.drive_channel_CA1((self.I_CaT_d) + (self.I_CaR_d))
        self.drive_channel_s = self.drive_channel_CA1(self.I_CaL_s)

        self.I_KNa = self.I_KNa_CA1(self.g_KNa, self.y[24], self.y[0], self.E_K)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def rk4(self):
        """Advance ``y`` by one time step ``dt`` with a fourth-order Runge-Kutta scheme.

        NOTE: the currents and gate rates come from the last ``updateParameters`` call and are not
        recomputed at the intermediate stages, so only the explicit terms of ``derivT`` follow the
        stage values (accurate enough for the small ``dt``, but not a full RK4 for the currents).
        """
        self.yt = self.y + 0.0  # state at time t
        self.dydx1 = self.derivT()  # K1
        self.y = self.yt + self.dydx1 * self.dt / 2
        self.dydx2 = self.derivT()  # K2
        self.y = self.yt + self.dydx2 * self.dt / 2
        self.dydx3 = self.derivT()  # K3
        self.y = self.yt + self.dydx3 * self.dt
        self.derivT()  # K4 (stored in self.dydx)
        self.y = self.yt + self.dt / 6.0 * (self.dydx1 + 2 * self.dydx2 + 2 * self.dydx3 + self.dydx)  # state at t+dt

    def rk4_Time(self, t):
        """Simulate the isolated cell over the time vector ``t`` (ms) and return the somatic potential.

        A somatic current step of amplitude ``istim`` is applied for ``t_start <= t <= t_end``.
        """
        lfp = np.zeros(len(t))
        for k in range(len(t)):
            if t[k] >= self.t_start and t[k] <= self.t_end:
                self.I_soma_stim = self.istim
            else:
                self.I_soma_stim = 0.0
            self.updateParameters()
            self.rk4()
            lfp[k] = self.y[0]
        return lfp

    def derivT(self):
        """Right-hand side of the ODEs: fill ``dydx`` and return a copy of it."""
        # Membrane equations of the two compartments, coupled through g_c
        self.dVs_dt = (
            1.0 / self.Cm
            * (
                -self.I_KNa - self.I_Na_s - self.I_KDR_s - self.I_CaL_s - self.I_AHP_s - self.I_AHP_BK_s
                - self.I_m_s - self.I_leak_s - self.I_synSoma
                - (self.g_c / self.p * (self.Vs_ - self.Vd_))
                + self.I_soma_stim
            )
        )  # fmt: skip
        self.dVd_dt = (
            1.0 / self.Cm
            * (
                -self.I_Na_d - self.I_KDR_d - self.I_KA_d - self.I_CaT_d - self.I_CaR_d - self.I_h_d
                - self.I_AHP_d - self.I_AHP_BK_d - self.I_m_d - self.I_leak_d - self.I_synDend
                - (self.g_c / (1 - self.p) * (-self.Vs_ + self.Vd_))
                + self.I_dend_stim
            )
        )  # fmt: skip
        # Gating variables: dx/dt = (x_inf - x) / tau
        self.ds_s_dt = (self.s_inf_s - self.s_s_) / self.tau_s_s
        self.ds_d_dt = (self.s_inf_d - self.s_d_) / self.tau_s_d
        self.dh_s_dt = (self.h_inf_s - self.h_s_) / (0.5)
        self.dm_s_dt = (self.m_inf_s - self.m_s_) / (0.05)
        self.dm_KDR_s_dt = (self.m_KDR_inf_s - self.m_KDR_s_) / (2.2)
        self.dh_d_dt = (self.h_inf_d - self.h_d_) / (0.5)
        self.dm_d_dt = (self.m_inf_d - self.m_d_) / (0.05)
        self.dm_KDR_d_dt = (self.m_KDR_inf_d - self.m_KDR_d_) / (2.2)
        self.dm_KA_d_dt = (self.m_KA_inf_d - self.m_KA_d_) / (self.tau_m_KA)
        self.dh_KA_d_dt = (self.h_KA_inf_d - self.h_KA_d_) / (self.tau_h_KA)
        # Intracellular calcium: calcium entry + relaxation to 0.0001 with a time constant of 200
        self.dCa_s_dt = self.drive_channel_s + ((0.0001 - self.Ca_s_) / 200.0) * 1.0
        self.dCa_d_dt = self.drive_channel_d + ((0.0001 - self.Ca_d_) / 200.0) * 1.0
        self.dm_CaT_d_dt = (self.m_CaT_inf_d - self.m_CaT_d_) / (self.tau_m_CaT)
        self.dh_CaT_d_dt = (self.h_CaT_inf_d - self.h_CaT_d_) / (self.tau_h_CaT)
        self.dm_CaR_d_dt = (self.m_CaR_inf_d - self.m_CaR_d_) / (50.0)
        self.dh_CaR_d_dt = (self.h_CaR_inf_d - self.h_CaR_d_) / (5.0)
        self.dm_CaL_s_dt = (self.m_CaL_inf_s - self.m_CaL_s_) / (self.tau_m_CaL)
        self.dm_h_d_dt = (self.m_h_inf_d - self.m_h_d_) / (self.tau_m_h)
        self.dm_AHP_s_dt = (self.m_AHP_inf_s - self.m_AHP_s_) / (self.tau_m_AHP_s)
        self.dm_AHP_d_dt = (self.m_AHP_inf_d - self.m_AHP_d_) / (self.tau_m_AHP_d)
        self.dm_AHP_BK_s_dt = (self.m_AHP_BK_inf_s - self.m_AHP_BK_s_) / (self.tau_m_AHP_BK_s)
        self.dm_AHP_BK_d_dt = (self.m_AHP_BK_inf_d - self.m_AHP_BK_d_) / (self.tau_m_AHP_BK_d)
        self.dm_m_s_dt = (self.m_m_inf_s - self.m_m_s_) / (self.tau_m_m_s)
        self.dm_m_d_dt = (self.m_m_inf_d - self.m_m_d_) / (self.tau_m_m_d)
        # Intracellular sodium: Na entry (somatic Na current) minus the Na/K pump
        Na_intra_power3 = self.Na_intra_ * self.Na_intra_ * self.Na_intra_
        Na_eq_power3 = self.Na_eq * self.Na_eq * self.Na_eq
        self.dNa_intra_dt = -self.alpha_Na * (self.A_s * (self.I_Na_s)) - self.Rpump * (
            Na_intra_power3 / (Na_intra_power3 + self.fifteenpoxer3)
            - (Na_eq_power3 / (Na_eq_power3 + self.fifteenpoxer3))
        )

        return self.dydx + 0.0

    # ------------------------------------------------------------------
    # Gate steady states and time constants (V in mV, Ca in the units of y[12], y[14])
    # ------------------------------------------------------------------
    def m_inf_CA1(self, V):
        """Na activation, steady state."""
        return 1.0 / (1.0 + np.exp(-(V + 40.0) / 3.0))

    def h_inf_CA1(self, V):
        """Na inactivation, steady state; shifted by ``theta_shift`` (lamotrigine)."""
        return 1.0 / (1.0 + np.exp((V + 45.0 + self.theta_shift) / 3.0))

    def s_inf_CA1(self, V):
        """Slow Na inactivation gate s, steady state (half-point -44 mV, slope 14.4 mV)."""
        return 1.0 / (1.0 + np.exp((V + 44) / 14.4))

    def tau_s_CA1(self, V):
        """Time constant of the gate s: 1.5 to 2.5 (model time units, ms).

        NOTE: this is barely slower than h (0.5 ms), so s is not a slow gate as written.
        """
        return 1.5 + 1.0 / (1.0 + np.exp((V + 50.0) / 10.0))

    def m_KDR_inf_CA1(self, V):
        """Delayed-rectifier K activation, steady state."""
        return 1.0 / (1.0 + np.exp(-(V + 42.0) / 2.0))

    def foncMax_CA1(self, v1, v2):
        """Maximum of two numbers (used to bound time constants from below)."""
        if v1 < v2:
            return v2
        else:
            return v1

    def zeta(self, V):
        """Voltage-dependent gating charge of the KA activation."""
        return -1.8 - (1.0 / (1.0 + np.exp((V + 40.0) / 5.0)))

    def alpha_m_KA_CA1(self, V):
        """KA activation rate term (at 37 C)."""
        return np.exp(0.001 * self.zeta(V) * (V + 1.0) * self.F / (self.R * (self.T + 37.0)))

    def m_KA_inf_CA1(self, V):
        """KA activation, steady state."""
        return 1.0 / (1.0 + self.alpha_m_KA_CA1(V))

    def h_KA_inf_CA1(self, V):
        """KA inactivation, steady state."""
        alpha = np.exp(0.003 * (V + 56.0) * self.F / (self.R * (self.T + 37.0)))
        return 1.0 / (1.0 + alpha)

    def tau_m_KA_CA1(self, V):
        """KA activation time constant (at least 0.1)."""
        beta = np.exp(0.00039 * self.zeta(V) * (V + 1.0) * self.F / (self.R * (self.T + 37.0)))
        v1 = beta / (0.1 * self.qtm_KA * (1 + self.alpha_m_KA_CA1(V)))
        return self.foncMax_CA1(v1, 0.1)

    def tau_h_KA_CA1(self, V):
        """KA inactivation time constant (at least 2)."""
        return self.foncMax_CA1((0.26 * (V + 50)), 2.0)

    def alpha_m_CaT_CA1(self, V):
        """CaT activation, opening rate. NOTE: the denominator is 0 at V = 19.88 mV."""
        n = -0.196 * (V - 19.88)
        d = np.exp(-(V - 19.88) / 10.0) - 1.0
        return n / d

    def beta_m_CaT_CA1(self, V):
        """CaT activation, closing rate."""
        return 0.046 * np.exp(-V / 22.73)

    def alpha_h_CaT_CA1(self, V):
        """CaT inactivation, opening rate."""
        return 0.00016 * np.exp(-(V + 57.0) / 19.0)

    def beta_h_CaT_CA1(self, V):
        """CaT inactivation, closing rate."""
        n = 1.0
        d = np.exp(-(V - 15.0) / 10.0) + 1.0
        return n / d

    def m_CaT_inf_CA1(self, V):
        """CaT activation, steady state."""
        return self.alpha_m_CaT_CA1(V) / (self.alpha_m_CaT_CA1(V) + self.beta_m_CaT_CA1(V))

    def h_CaT_inf_CA1(self, V):
        """CaT inactivation, steady state."""
        return self.alpha_h_CaT_CA1(V) / (self.alpha_h_CaT_CA1(V) + self.beta_h_CaT_CA1(V))

    def m_CaR_inf_CA1(self, V):
        """CaR activation, steady state."""
        return 1.0 / (1.0 + np.exp(-(V + 48.5) / 3.0))

    def h_CaR_inf_CA1(self, V):
        """CaR inactivation, steady state. NOTE: no slope divisor (1 mV slope), unlike the other gates."""
        return 1.0 / (1.0 + np.exp(V + 53.0))

    def tau_m_CaT_CA1(self, V):
        """CaT activation time constant."""
        return 1.0 / (self.alpha_m_CaT_CA1(V) + self.beta_m_CaT_CA1(V))

    def tau_h_CaT_CA1(self, V):
        """CaT inactivation time constant."""
        return 1.0 / (0.68 * (self.alpha_h_CaT_CA1(V) + self.beta_h_CaT_CA1(V)))

    def alpha_m_CaL_CA1(self, V):
        """CaL activation, opening rate. NOTE: the denominator is 0 at V = -27.01 mV."""
        n = -0.055 * (V + 27.01)
        d = np.exp(-(V + 27.01) / 3.8) - 1.0
        return n / d

    def beta_m_CaL_CA1(self, V):
        """CaL activation, closing rate."""
        return 0.94 * np.exp(-(V + 63.01) / 17.0)

    def m_CaL_inf_CA1(self, V):
        """CaL activation, steady state."""
        alpha_m_CaL_CA1 = self.alpha_m_CaL_CA1(V)
        return alpha_m_CaL_CA1 / (alpha_m_CaL_CA1 + self.beta_m_CaL_CA1(V))

    def tau_m_CaL_CA1(self, V):
        """CaL activation time constant."""
        return 1.0 / (5 * (self.alpha_m_CaL_CA1(V) + self.beta_m_CaL_CA1(V)))

    def drive_channel_CA1(self, ICa):
        """Rate of rise of intracellular Ca produced by an inward Ca current ``ICa`` (0 if outward)."""
        result = -10000.0 * ICa / (0.2 * 96480.0)
        if result > 0.0:
            return result * 1.0e-3
        else:
            return 0.0

    def fonc_Ca_CA1(self, Ca):
        """Ca-dependent factor of the Ca currents (decreases as Ca increases)."""
        return 0.001 / (0.001 + Ca)

    def fonc_f_CA1(self, z):
        """z / (exp(z) - 1), with a series expansion near z = 0 (used by the GHK flux)."""
        if np.abs(z) < 0.0001:
            return 1.0 - (z / 2.0)
        else:
            return z / (np.exp(z) - 1.0)

    def fonc_ghk_CA1(self, V, Cai):
        """Goldman-Hodgkin-Katz driving-force term for calcium."""
        nu = V / self.xghk
        return -self.xghk * (1.0 - (Cai / 2.0) * np.exp(nu)) * self.fonc_f_CA1(nu)

    def m_h_inf_CA1(self, V):
        """h-current (HCN) activation, steady state."""
        return 1.0 - (1.0 / (1.0 + np.exp(-(V + 90.0) / 8.5)))

    def tau_m_h_CA1(self, V):
        """h-current activation time constant."""
        if V > (-30.0):
            return 1.0
        else:
            n = 2.0
            d = np.exp(-((V + 145.0) / 17.5)) + np.exp((V + 16.8) / 16.5)
            return (n / d) + 10.0

    def fonc_Cac_CA1(self, Ca_in):
        """Squared Ca concentration relative to 0.025 (activation of the AHP current)."""
        return (Ca_in / 0.025) * (Ca_in / 0.025)

    def m_AHP_inf_CA1(self, Ca_in):
        """AHP activation, steady state (depends on Ca only)."""
        return self.fonc_Cac_CA1(Ca_in) / (1.0 + self.fonc_Cac_CA1(Ca_in))

    def tau_m_AHP_CA1(self, Ca_in):
        """AHP activation time constant (at least 0.5)."""
        v1 = 1.0 / (0.003 * (1.0 + self.fonc_Cac_CA1(Ca_in)) * np.power(3.0, (34.0 - 22.0) / 10.0))
        return self.foncMax_CA1(v1, 0.5)

    def m_AHP_BK_inf_CA1(self, V, Ca_in):
        """BK-type AHP activation, steady state (depends on V and Ca). Also updates ``Q`` for 34 C.

        NOTE: ``Q`` is in 1/V but ``V`` is in mV, so ``exp(-1.68 * V * Q)`` overflows for V below about
        -11 mV (alpha then evaluates to 0): the BK gate is practically closed at sub-threshold potentials.
        A factor 0.001 (as in the KA functions) may be missing. Behaviour left unchanged.
        """
        self.Q = self.F / (self.R * (self.T + 34.0))
        if Ca_in == 0:
            alpha = 0
        elif (1.0 + (0.18 / Ca_in) * np.exp(-1.68 * V * self.Q)) == 0.0:
            alpha = 1e20
        else:
            alpha = 0.48 / (1.0 + (0.18 / Ca_in) * np.exp(-1.68 * V * self.Q))
        if (0.011 * np.exp(-2.0 * V * self.Q)) == 0.0:
            beta = 0
        elif (1.0 + (Ca_in / (0.011 * np.exp(-2.0 * V * self.Q)))) == 0.0:
            beta = 0
        else:
            beta = 0.28 / (1.0 + (Ca_in / (0.011 * np.exp(-2.0 * V * self.Q))))
        tau = 1.0 / (alpha + beta)
        return alpha * tau

    def tau_m_AHP_BK_CA1(self, V, Ca_in):
        """BK-type AHP activation time constant. Also updates ``Q`` for 34 C (same units note as m_AHP_BK_inf_CA1)."""
        self.Q = self.F / (self.R * (self.T + 34.0))
        if Ca_in == 0:
            alpha = 0
        elif (1.0 + (0.18 / Ca_in) * np.exp(-1.68 * V * self.Q)) == 0:
            alpha = 1e20
        else:
            alpha = 0.48 / (1.0 + (0.18 / Ca_in) * np.exp(-1.68 * V * self.Q))
        if (0.011 * np.exp(-2.0 * V * self.Q)) == 0.0:
            beta = 0
        elif (1.0 + (Ca_in / (0.011 * np.exp(-2.0 * V * self.Q)))) == 0.0:
            beta = 0
        else:
            beta = 0.28 / (1.0 + (Ca_in / (0.011 * np.exp(-2.0 * V * self.Q))))
        return 1.0 / (alpha + beta)

    def m_m_inf_CA1(self, V):
        """M-current activation, steady state."""
        alpha = 0.016 * (np.exp((V + 52.7) / 23.0))
        beta = 0.016 * (np.exp(-(V + 52.7) / 18.8))
        tau = 1.0 / (alpha + beta)
        return alpha * tau

    def tau_m_m_CA1(self, V):
        """M-current activation time constant."""
        alpha = 0.016 * (np.exp((V + 52.7) / 23.0))
        beta = 0.016 * (np.exp(-(V + 52.7) / 18.8))
        return 1.0 / (alpha + beta)

    # ------------------------------------------------------------------
    # Ionic currents (g * gates * driving force)
    # ------------------------------------------------------------------
    def I_Na_CA1(self, g_Na, m_inf, h, s, Vs, E_Na):
        """Na current: g * m^2 * h * s * (V - E_Na)."""
        return g_Na * m_inf * m_inf * h * s * (Vs - E_Na)

    def I_KNa_CA1(self, g_KNa, Na_intra, Vs, E_K):
        """Na-activated K current."""
        w = 0.37 / (1.0 + (38.7 / np.power(Na_intra, 3.5)))
        return g_KNa * w * (Vs - E_K)

    def I_KDR_CA1(self, g_KDR, m, V, E_K):
        """Delayed-rectifier K current."""
        return g_KDR * m * m * (V - E_K)

    def I_KA_CA1(self, g_KA, m, h, V, E_K):
        """A-type K current."""
        return g_KA * m * h * (V - E_K)

    def I_CaT_CA1(self, g_CaT, m_CaT, h_CaT, fonc_Ca, fonc_ghk, V, E_Ca):
        """T-type Ca current (GHK form; ``V`` and ``E_Ca`` are not used)."""
        return g_CaT * m_CaT * m_CaT * h_CaT * fonc_Ca * fonc_ghk

    def I_CaR_CA1(self, g_CaR, m_CaR, h_CaR, V, E_Ca):
        """R-type Ca current."""
        return g_CaR * m_CaR * m_CaR * m_CaR * h_CaR * (V - E_Ca)

    def I_CaL_CA1(self, g_CaL, m_CaL, fonc_Ca, fonc_ghk, V, E_Ca):
        """L-type Ca current (GHK form; ``V`` and ``E_Ca`` are not used)."""
        return g_CaL * m_CaL * fonc_Ca * fonc_ghk

    def I_h_CA1(self, g_h, m, V, E_h):
        """h current (HCN)."""
        return g_h * m * (V - E_h)

    def I_AHP_CA1(self, g_AHP, m_AHP, V, E_K):
        """Ca-activated K (AHP) current."""
        return g_AHP * m_AHP * m_AHP * m_AHP * (V - E_K)

    def I_AHP_BK_CA1(self, g_AHP, m_AHP, V, E_K):
        """BK-type Ca- and voltage-activated K current."""
        return g_AHP * m_AHP * (V - E_K)

    def I_m_CA1(self, g_m, m_m, V, E_K):
        """M-type K current."""
        return g_m * m_m * m_m * (V - E_K)

    # ------------------------------------------------------------------
    # Synapses (presynaptic potentials ``Vpre`` are given by the network class)
    # ------------------------------------------------------------------
    def I_AMPA(self, Vd):
        """AMPA current from the current gating state ``s_AMPA`` at potential ``Vd``."""
        return ((self.g_AMPA * 0.1) / 3.32) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_AMPA2(self, Vpre, PreSynaptic_Soma_Dend_AMPA):
        """AMPA currents for presynaptic potentials ``Vpre``; the flag is 1 for a somatic, else dendritic, synapse."""
        self.computeI_AMPA(Vpre)
        Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_AMPA])
        return ((self.g_AMPA * 0.1) / 3.32) * self.s_AMPA * (Vd - self.E_AMPA)

    def I_GABA(self, Vd):
        """GABA current from the current gating state ``s_GABA`` at potential ``Vd``."""
        return ((self.g_GABA * 0.1) / 3.32) * self.s_GABA * (Vd - self.E_GABA)

    def I_GABA2(self, Vpre, PreSynaptic_Soma_Dend_GABA):
        """GABA currents for presynaptic potentials ``Vpre``; the flag is 1 for a somatic, else dendritic, synapse."""
        self.computeI_GABA(Vpre)
        Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_GABA])
        return ((self.g_GABA * 0.1) / 3.32) * self.s_GABA * (Vd - self.E_GABA)

    def I_NMDA2(self, Vpre, PreSynaptic_Soma_Dend_AMPA, t):
        """NMDA currents for presynaptic potentials ``Vpre`` at time ``t`` (see the note in computeI_NMDA)."""
        Vd = np.array([self.VSoma() if s_d == 1 else self.VDend() for s_d in PreSynaptic_Soma_Dend_AMPA])
        return np.array([self.computeI_NMDA(Vpre[k], t, Vd[k]) for k in range(len(Vpre))])

    def T_fonc_CA1(self, Vpre):
        """AMPA transmitter concentration as a sigmoid of the presynaptic potential."""
        return 2.84 / (1 + np.exp(-(Vpre - 2.0) / 5.0))

    def F_fonc_CA1(self, Vpre):
        """GABA transmitter release as a sigmoid of the presynaptic potential."""
        return 1.0 / (1.0 + np.exp(-(Vpre - 0.0) / 2.0))

    def computeI_AMPA(self, Vpre):
        """Advance the AMPA gating variable ``s_AMPA`` by one time step (RK4) and return it."""
        self.T_fonc = 1.1 * self.T_fonc_CA1(Vpre)
        self.s_AMPAo = self.s_AMPA + 0.0  # state at time t
        self.ds_AMPA1 = self.derivs_AMPA_CA1()  # K1
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA1 * self.dt / 2
        self.ds_AMPA2 = self.derivs_AMPA_CA1()  # K2
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA2 * self.dt / 2
        self.ds_AMPA3 = self.derivs_AMPA_CA1()  # K3
        self.s_AMPA = self.s_AMPAo + self.ds_AMPA3 * self.dt
        # K4 and state at t+dt
        self.s_AMPA = self.s_AMPAo + self.dt / 6.0 * (
            self.ds_AMPA1 + 2 * self.ds_AMPA2 + 2 * self.ds_AMPA3 + self.derivs_AMPA_CA1()
        )
        return self.s_AMPA

    def derivs_AMPA_CA1(self):
        """Derivative of the AMPA gating variable (rise 1.1*T, decay 0.19)."""
        return self.T_fonc * (1 - self.s_AMPA) - 0.19 * self.s_AMPA

    def computeI_GABA(self, Vpre):
        """Advance the GABA gating variable ``s_GABA`` by one time step (RK4) and return it."""
        self.F_fonc = 10.0 * self.F_fonc_CA1(Vpre)
        self.s_GABAo = self.s_GABA + 0.0  # state at time t
        self.ds_GABA1 = self.derivs_GABA_CA1()  # K1
        self.s_GABA = self.s_GABAo + self.ds_GABA1 * self.dt / 2
        self.ds_GABA2 = self.derivs_GABA_CA1()  # K2
        self.s_GABA = self.s_GABAo + self.ds_GABA2 * self.dt / 2
        self.ds_GABA3 = self.derivs_GABA_CA1()  # K3
        self.s_GABA = self.s_GABAo + self.ds_GABA3 * self.dt
        # K4 and state at t+dt
        self.s_GABA = self.s_GABAo + self.dt / 6.0 * (
            self.ds_GABA1 + 2 * self.ds_GABA2 + 2 * self.ds_GABA3 + self.derivs_GABA_CA1()
        )
        return self.s_GABA

    def derivs_GABA_CA1(self):
        """Derivative of the GABA gating variable (rise 10*F, decay 0.07)."""
        return self.F_fonc * (1.0 - self.s_GABA) - 0.07 * self.s_GABA

    def computeI_NMDA(self, Vpre, t, Vs_d):
        """NMDA current for one presynaptic potential ``Vpre`` at time ``t`` (transmitter pulse model).

        A pulse of transmitter of duration ``Cdur`` is triggered when ``Vpre`` exceeds ``preThresh``
        (after a dead time), the receptor open fraction ``Ra`` rises and decays accordingly, and the
        current is gated by the voltage-dependent magnesium block ``B``.

        NOTE: ``C``, ``Ra``, ``R0``, ``R1`` and ``lastRelease`` are single attributes, but
        ``I_NMDA2`` calls this method once per presynaptic cell, so all synapses of the cell share
        one kinetic state. NMDA needs one state per synapse to be exact.
        """
        q = t - self.lastRelease - self.Cdur

        if q > self.deadTime:
            if Vpre > self.preThresh:
                self.C = self.Cmax
                self.R0 = self.Ra
                self.lastRelease = t
        elif q < 0.0:
            pass
        elif self.C == self.Cmax:
            self.R1 = self.Ra
            self.C = 0.0
        if self.C > 0:
            self.Rinf = self.Cmax * self.alpha / (self.Cmax * self.alpha + self.beta)
            self.Rtau = 1.0 / (self.Cmax * self.alpha + self.beta)
            self.Ra = self.Rinf + (self.R0 - self.Rinf) * np.exp(-(t - self.lastRelease) / self.Rtau)
        else:
            self.Ra = self.R1 * np.exp(-1.0 * self.beta * (t - (self.lastRelease + self.Cdur)))

        B = 1.0 / (1 + np.exp(0.062 * (-Vs_d)) * self.mg / 10.0)  # magnesium block

        return self.Ra * B * self.g_NMDA * (Vs_d - self.E_NMDA)

    # ------------------------------------------------------------------
    # Interface used by the network class
    # ------------------------------------------------------------------
    def VsOutput(self):
        """Potential of the compartment recorded by the network (soma)."""
        return self.y[0]

    def VSoma(self):
        """Somatic membrane potential."""
        return self.y[0]

    def VDend(self):
        """Dendritic membrane potential."""
        return self.y[1]

    def init_I_syn(self):
        """Reset the total synaptic currents (called before the synaptic inputs are summed)."""
        self.I_synSoma = 0.0
        self.I_synDend = 0.0

    def add_I_synSoma(self, I, vect):
        """Add to the somatic synaptic current the sum of ``I[i] * vect[i]`` (``vect`` selects the synapses)."""
        for i in range(len(I)):
            self.I_synSoma += I[i] * vect[i]

    def add_I_synDend(self, I, vect):
        """Add to the dendritic synaptic current the sum of ``I[i] * vect[i]``."""
        for i in range(len(I)):
            self.I_synDend += I[i] * vect[i]

    def michaelis_menten(self, c, Vmax, V50):
        """Michaelis-Menten relation Vmax * c / (V50 + c); gives the lamotrigine shift (mV)."""
        return (Vmax * c) / (V50 + c)
