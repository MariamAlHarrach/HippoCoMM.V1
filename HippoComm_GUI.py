# -*- coding: utf-8 -*-
"""HippoCoMM -- main graphical user interface.

PyQt6 application used to build, simulate and analyse a 3D microscale network
model of the hippocampal CA1 subfield, and to compute the resulting local field
potential (LFP) recorded by a virtual electrode.

Typical workflow (left control panel, top to bottom)
    1. Tissue size, neuron density and number of cells.
    2. Percentage of each cell type (and fraction of CA3 cells that are stimulated).
    3. Number of connections and connection distances between cell types.
    4. Cell placement -> "Place cells" (computes positions *and* connectivity).
    5. Stimulation (CA3 input) and simulation parameters.
    6. "Update Model" (instantiates the cells), then "Run".
    7. Electrode position/shape -> "Comp" to compute and display the LFP.

Conventions used throughout this file
    * Cell-type codes stored in ``List_Neurone_type``:
      1 = CA1 pyramidal cell (PYR), 2 = CA3 pyramidal cell that receives the
      stimulation (PYR_STIM), 3 = basket cell (BAS), 4 = OLM / somatostatin cell
      (SOM), 5 = bistratified cell (BIS).
    * Cells are stored in this order: the stimulated CA3 cells first, then the
      CA1 PYR cells, then BAS, SOM and BIS (``Nb_of_PYR`` counts both PYR groups).
    * Times are in ms, the sampling frequency ``Fs`` is in kHz (so ``dt = 1/Fs`` ms)
      and lengths (tissue size, cell positions, electrode position) are in micrometres.
    * ``ConnectivityMatrix[i]`` is the array of presynaptic (source) cell indices of
      cell ``i``. In the connectivity tables of the GUI, rows are the target cell
      type and columns the source cell type.

Related code
    * ``Core/``            : EEG/3D viewers, parameter editors, LFP and electrode models.
    * ``Modeles/Wendling/``: cell models and the network class ``Colonne_cortical``.

Run with ``python HippoComm_GUI.py``."""

__author__ = 'Maxime'

# Qt bindings (star imports kept first, as in the original, so the names they define keep the same precedence)
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *

# Standard library
import copy
import csv
import datetime
import os
import pickle
import platform
import random
import struct
import sys
import time

# Scientific stack
import numpy as np
import pyedflib
import scipy.io
from numba import guvectorize, njit
from scipy import signal
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.spatial import distance

# Matplotlib: the Qt backend must be selected before pyplot is imported
import matplotlib
import matplotlib.patches as patches

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# HippoCoMM modules
from Core import Electrode
from Core import RecordedPotential
from Core.EEGViewer import lfpViewer_EEG
from Core.Graph_viewer3D_VTK5 import Graph_viewer3D_VTK
from Core.Modify_Each_NMM_param import Modify_1_NMM
from Core.Modify_X_NMM_VTK import Modify_X_NMM

# Platform-specific tweaks
if platform.system() == "Darwin":
    matplotlib.rcParams.update({'font.size': 8})


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def peakdet(v, delta, x=None):
    """Detect local maxima and minima of a 1-D signal.

    Hysteresis-based peak detection: a point is a peak if the signal then falls by
    more than ``delta`` below it (and conversely for a trough).

    Args:
        v: 1-D signal.
        delta: minimum peak-to-trough amplitude (must be a positive scalar).
        x: optional abscissa for ``v`` (defaults to the sample index).

    Returns:
        (maxtab, mintab): two arrays of ``(x position, value)`` pairs.

    Note: invalid inputs terminate the program through ``sys.exit``.
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def iir_band_filter(ite_data, fs, order, lowcut, highcut, zerophase, btype, ftype, rps=None):
    """Apply an IIR filter to ``ite_data`` (second-order sections).

    Args:
        ite_data: 1-D signal.
        fs: sampling frequency (Hz).
        order: filter order.
        lowcut, highcut: cut-off frequencies (Hz); clipped to [0, Nyquist].
        zerophase: if truthy, filter forward then backward (zero phase shift).
        btype: scipy filter type ('bandpass', ...).
        ftype: scipy design family ('bessel', 'butter', 'cheby1', 'cheby2', 'ellip', ...).
        rps: ripple (cheby1: rp, cheby2: rs, ellip: (rp, rs)); unused otherwise.

    Returns:
        The filtered signal.
    """
    fe = fs / 2.0
    low = lowcut / fe
    high = highcut / fe
    if low < 0:
        low = 0
    if high > 1:
        high = 1
    if ftype == "cheby1":
        rp = rps
        z, p, k = signal.iirfilter(order, [low, high], btype=btype, ftype=ftype, output="zpk", rp=rp)
    elif ftype == "cheby2":
        rs = rps
        z, p, k = signal.iirfilter(order, [low, high], btype=btype, ftype=ftype, output="zpk", rs=rs)
    elif ftype == "ellip":
        rp = rps[0]
        rs = rps[1]
        z, p, k = signal.iirfilter(order, [low, high], btype=btype, ftype=ftype, output="zpk", rp=rp, rs=rs)
    else:
        z, p, k = signal.iirfilter(order, [low, high], btype=btype, ftype=ftype, output="zpk")
    sos = signal.zpk2sos(z, p, k)
    ite_data = signal.sosfilt(sos, ite_data)
    if zerophase:
        ite_data = signal.sosfilt(sos, ite_data[::-1])[::-1]
    return ite_data


def signalfilterbandpass(LFPs, Fs, lowcut, highcut, order):
    """Band-pass filter one or several LFP signals with a Bessel filter.

    ``LFPs`` is promoted to 2-D (signals x samples) and each row is filtered
    (non zero-phase). The order is clipped to [0, 10] and the cut-offs are kept
    inside (0, Fs/2). Note that the rows of ``LFPs`` are modified in place.

    Args:
        LFPs: signal(s), shape (n,) or (n_signals, n).
        Fs: sampling frequency (Hz).
        lowcut, highcut: band edges (Hz).
        order: filter order.

    Returns:
        2-D array (n_signals, n) with the filtered signals.
    """
    if len(LFPs.shape) == 1:
        LFPs = LFPs[np.newaxis, :]

    if order > 10:
        order = 10
    if order < 0:
        order = 0
    if lowcut <= 0:
        lowcut = 1 / Fs
    if highcut > Fs / 2:
        highcut = Fs / 2 - 1
    for idx_lfp, lfp in enumerate(LFPs):

        LFPs[idx_lfp] = iir_band_filter(lfp, Fs, order, lowcut, highcut, 0, 'bandpass', 'bessel', rps=None)
    return LFPs


@guvectorize(
    ["float64[:,:],int64, int64, float64, float64, int64, float64, int64, int64, float64, int64, float64[:,:]"],
    '(n,m),(),(),(),(),(),(),(),(),(),()->(n,m)',
)
def Generate_Stim_Signals(
    Stim_Signals_in,
    seed,
    nbOfSamplesStim,
    i_inj,
    tau,
    nbStim,
    varianceStim,
    nb_Stim_Signals,
    nbEch,
    dt,
    stimtype,
    Stim_Signals_out,
):
    """Build the current waveforms injected in the stimulated CA3 cells (numba gufunc).

    For each of the ``nb_Stim_Signals`` cells, ``nbStim`` pulses are drawn and summed.
    Each pulse has a duration of ``nbOfSamplesStim`` samples and amplitude ``i_inj``,
    with an exponential rise/decay of time constant ``tau`` (RC smoothing). The
    pulse onsets are jittered around the middle of the simulation window:
        * ``stimtype`` = 1 ('Burst')     : Gaussian jitter, standard deviation ``varianceStim`` (samples);
        * ``stimtype`` = 0 ('Continuous'): uniform jitter over the whole window.
    Pulses that would fall outside the window are skipped.

    Args (gufunc signature, see decorator):
        Stim_Signals_in: array only used to give the gufunc its (n, m) shape.
        seed: seed of the random generator (0 is passed to numpy as is).
        nbOfSamplesStim, i_inj, tau, nbStim, varianceStim, nb_Stim_Signals,
        nbEch, dt, stimtype: stimulation parameters (dt in ms).
        Stim_Signals_out: output array, shape (nb_Stim_Signals, nbEch).
    """
    n = int(1.0 * nbEch / 2.0)
    t = np.arange(nbEch) * dt

    np.random.seed(seed)
    for St in range(nb_Stim_Signals):
        y2 = np.zeros(nbEch)
        for i in range(nbStim):

            y = np.zeros(t.shape)

            if stimtype:  # gaussian
                intervalle_dt = int(np.round((np.random.normal(0, varianceStim))))
            else:
                intervalle_dt = int(np.round((np.random.randint(-n, n))))

            if (n + intervalle_dt) < 0:
                continue
            if (n + intervalle_dt + nbOfSamplesStim) > nbEch:
                continue

            for tt, tp in enumerate(t):
                if tt > (n + intervalle_dt):
                    y[tt] = (1.0 - np.exp(-(tp - (n + intervalle_dt) * dt) / tau)) * i_inj
                if tt > (n + intervalle_dt + nbOfSamplesStim):
                    y[tt] = (np.exp(-(tp - ((n + intervalle_dt) + nbOfSamplesStim) * dt) / tau)) * y[
                        n + intervalle_dt + nbOfSamplesStim - 1
                    ]

                y2[tt] += y[tt]

        Stim_Signals_out[St, :] = y2
    # return Stim_Signals


# Background input added to the stimulation signals of the stimulated CA3 cells in ``simulate``.
# Same parameters as ``Colonne_cortical.Generate_input`` in Colonne_cortical_Wendling2.py.
#   I_inj : pulse amplitude (uA/cm^2)   tau : RC time constant (ms)   stimDur : pulse duration (ms)
#   nbstim: number of pulses per cell   StimStart: first sample at which a pulse may start
BACKGROUND_INPUT = {'I_inj': 60.0, 'tau': 4.0, 'stimDur': 3.0, 'nbstim': 10, 'StimStart': 0}


def Generate_Background_Input(
    nb_Stim_Signals, nbEch, dt, seed=0, I_inj=60.0, tau=4.0, stimDur=3.0, nbstim=10, StimStart=0
):
    """Background input current for the stimulated CA3 cells: ``nbstim`` pulses at random times.

    For each of the ``nb_Stim_Signals`` cells, ``nbstim`` identical pulses (amplitude ``I_inj``,
    duration ``stimDur`` ms, exponential rise and decay with time constant ``tau`` ms) are started at
    instants drawn uniformly between sample ``StimStart`` and the end of the simulation, then summed.
    Unlike the main stimulation, the pulses are not centred on a common time, so each cell receives
    its own random input spread over the whole simulation. The result is meant to be added to the
    signals made by ``Generate_Stim_Signals``.

    Args:
        nb_Stim_Signals: number of stimulated cells.
        nbEch: number of samples of the simulation.
        dt: sampling period (ms).
        seed: seed of the random generator; 0 means a different input at each run.
        I_inj, tau, stimDur, nbstim, StimStart: see above (``StimStart`` is in samples).

    Returns:
        Array of shape (nb_Stim_Signals, nbEch).
    """
    NLim = 2500  # a pulse is truncated after this many samples
    rng = np.random.RandomState(seed) if seed != 0 else np.random.RandomState()
    nbOfSamplesStim = int(stimDur / dt)

    # One pulse: rises during stimDur, then decays
    t = np.arange(nbEch) * dt
    tt = np.arange(nbEch)
    y = np.zeros(nbEch)
    rise = tt <= nbOfSamplesStim
    y[rise] = (1.0 - np.exp(-t[rise] / tau)) * I_inj
    decay = tt > nbOfSamplesStim
    y[decay] = np.exp(-(t[decay] - nbOfSamplesStim * dt) / tau) * y[nbOfSamplesStim - 1]

    Stim_Signals_out = np.zeros((nb_Stim_Signals, nbEch))
    for St in range(nb_Stim_Signals):
        for i in range(nbstim):
            t0 = int(int(rng.random_sample() * (nbEch - StimStart)) + StimStart)
            t0 = min(max(t0, 0), nbEch - 1)
            n = min(NLim, nbEch - t0)
            Stim_Signals_out[St, t0 : t0 + n] += y[:n]
    return Stim_Signals_out


def Generate_Periodic_Stim_Signals(
    nb_Stim_Signals, nbEch, dt, seed, stimDur, i_inj, tau, nbStim, varstim, StimStart, periode
):
    """Periodic stimulation of the CA3 cells: ``nbStim`` jittered pulses at every period.

    Pulse onsets are placed every ``periode`` ms from ``StimStart`` ms to the end of the simulation. All
    cells share the same slightly shifted grid (each onset is moved by up to +-periode/4), and every pulse of
    every cell is then jittered by a Gaussian of standard deviation ``varstim`` ms. Each pulse has amplitude
    ``i_inj``, duration ``stimDur`` ms and an exponential rise/decay with time constant ``tau`` ms.
    Same algorithm as ``Colonne_cortical.Generate_Stims`` ('Periodic') in Colonne_cortical_Wendling2.py.

    Args:
        nb_Stim_Signals: number of stimulated cells.
        nbEch, dt: number of samples and sampling period (ms).
        seed: seed of the random generator; 0 means a different stimulation at each run.
        stimDur, i_inj, tau, nbStim, varstim, StimStart, periode: see above (all times in ms).

    Returns:
        Array of shape (nb_Stim_Signals, nbEch).
    """
    NLim = 2500  # a pulse is truncated after this many samples
    rng = np.random.RandomState(seed) if seed != 0 else np.random.RandomState()
    nbOfSamplesStim = int(stimDur / dt)

    # One pulse: rises during stimDur, then decays
    t = np.arange(nbEch) * dt
    tt = np.arange(nbEch)
    y = np.zeros(nbEch)
    rise = tt <= nbOfSamplesStim
    y[rise] = (1.0 - np.exp(-t[rise] / tau)) * i_inj
    decay = tt > nbOfSamplesStim
    y[decay] = np.exp(-(t[decay] - nbOfSamplesStim * dt) / tau) * y[nbOfSamplesStim - 1]

    # Onsets of the periods, shifted by up to +-periode/4 (the same for all cells)
    npos = np.arange(int(StimStart / dt), nbEch, int(periode / dt), dtype=float)
    npos += (rng.uniform(size=len(npos)) - 0.5) * (int(periode / dt) / 2)

    varianceStim = varstim / dt  # jitter standard deviation in samples
    Stim_Signals_out = np.zeros((nb_Stim_Signals, nbEch))
    for St in range(nb_Stim_Signals):
        for j in range(len(npos)):
            for i in range(nbStim):
                t0 = int(npos[j]) + int(np.round(rng.normal(0, varianceStim)))
                if t0 > 0 and t0 < nbEch:
                    n = min(NLim, nbEch - t0)
                    Stim_Signals_out[St, t0 : t0 + n] += y[:n]
    return Stim_Signals_out


@guvectorize(["int64[:], int64, int64, int64, int64[:]"], '(n),(),(),()->(n)')
def pickcell_simple_fun(size, loop, low, high, pickedcell):
    """Draw ``loop`` distinct cell indices uniformly in [low, high) (numba gufunc).

    The result is written in ``pickedcell`` (initialised with -1). ``size`` is only
    used by numba to define the output shape.
    """
    while loop > 0:
        cell = np.random.randint(low, high)
        if cell in pickedcell:
            pass
        else:
            pickedcell[loop - 1] = cell
            loop -= 1


@guvectorize(["int64[:], int64, float64, float64, float64[:], int64[:], int64[:]"], '(n),(),(),(),(m),(k)->(n)')
def pickcell_gauss_fun(size, loop, mean, var, CellDistances, sortedIndex, pickedcell):
    """Draw ``loop`` distinct cells with a distance-dependent probability (numba gufunc).

    For each connection, a distance ``l = |N(mean, var)|`` is drawn and the closest
    candidate whose distance is >= ``l`` is chosen (candidates are ordered by
    increasing distance in ``sortedIndex``). A chosen cell is removed from the
    candidates, so cells are picked without replacement.

    Args:
        size: only used by numba to define the output shape.
        loop: number of cells to pick.
        mean, var: parameters of the Gaussian used to draw distances.
        CellDistances: distances (um) from the target cell to all cells.
        sortedIndex: candidate cell indices sorted by increasing distance.
        pickedcell: output array of picked cell indices.
    """
    while loop > 0:
        l = abs(np.random.normal(mean, var))
        cellInd = np.where(CellDistances[sortedIndex] >= l)[0]
        if cellInd.size == 0:
            continue
        else:
            cellInd = cellInd[0]
        cell = sortedIndex[cellInd]
        if not cell in pickedcell:
            pickedcell[loop - 1] = cell
            sortedIndex = np.delete(sortedIndex, cellInd)
            loop -= 1


@guvectorize(["float64[:], int64, int64[:]"], '(n),()->(n)')
def argsort_list_fun(CellDistances, plus, sortedIndex):
    """Sort cell indices by distance (numba gufunc).

    NOTE: currently unused (its only call is commented out in the connectivity code).
    It also rebinds the local ``sortedIndex`` instead of filling the output array,
    so it would return nothing useful as written; ``np.argsort`` is used instead.
    """
    sortedIndex = np.argsort(CellDistances) + plus


class LineEdit(QLineEdit):
    """QLineEdit that only accepts a decimal number (optional sign and fractional part)."""

    KEY = Qt.Key.Key_Return

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegularExpressionValidator(QRegularExpression("[+-]?\\d*[\\.]?\\d+"))
        QREV.setLocale(QLocale(QLocale.Language.English))
        self.setValidator(QREV)
        self.setValidator(QREV)


class LineEdit_Int(QLineEdit):
    """QLineEdit that only accepts an integer (optional sign)."""

    KEY = Qt.Key.Key_Return

    def __init__(self, *args, **kwargs):
        QLineEdit.__init__(self, *args, **kwargs)
        QREV = QRegularExpressionValidator(QRegularExpression("[+-]?\\d+"))
        QREV.setLocale(QLocale(QLocale.Language.English))
        self.setValidator(QREV)


def set_QPushButton_background_color(button=None, color=None):
    """Set the background colour of ``button`` to the QColor ``color`` (no-op if either is None)."""
    if color == None or button == None:
        return
    else:
        button.setAutoFillBackground(True)
        values = "{r}, {g}, {b} ".format(r=color.red(), g=color.green(), b=color.blue())
        button.setStyleSheet("QPushButton { background-color: rgb(" + values + "); }")


def label_color_clicked(event, button):
    """Open a colour dialog starting from the button colour and apply the chosen colour to ``button``."""
    color = QColor(button.palette().button().color())
    colordial = QColorDialog(color)
    colordial.exec()
    selectedcolor = colordial.currentColor()
    colordial.close()
    set_QPushButton_background_color(button, selectedcolor)
    pass


def Layout_grid_Label_Edit(label=['None'], edit=['None']):
    """Build a widget made of rows "label | edit field".

    Args:
        label: list of label texts.
        edit: list of initial texts of the edit fields (same length as ``label``).

    Returns:
        (widget, Edit_List): the container widget and the list of ``LineEdit`` objects.
    """
    widget = QWidget()
    layout_range = QVBoxLayout()

    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(0)
    widget.setLayout(grid)
    layout_range.addLayout(grid)
    Edit_List = []
    for idx in range(len(label)):
        Label = QLabel(label[idx])
        Edit = LineEdit(edit[idx])
        grid.addWidget(Label, idx, 0)
        grid.addWidget(Edit, idx, 1)
        Edit_List.append(Edit)
    return widget, Edit_List


class Colonne_cortical_Thread(QThread):
    """QThread wrapper around the network model (``Colonne_cortical``).

    It instantiates the model class in ``self.CC`` (the object the GUI actually uses)
    and is meant to run ``CC.Simulate`` in a background thread.

    NOTE: in the current GUI the simulation is called directly from
    ``ModelMicro_GUI.simulate`` (GUI thread), so ``run`` is not used. If it were used,
    ``run`` unpacks 6 outputs while ``CC.Simulate`` returns 11, and ``arret`` refers
    to a non-existent attribute ``self.C``; both would need fixing first.
    """

    finished = pyqtSignal()
    updateTime = pyqtSignal(float)

    def __init__(self, CortexClass):
        """Instantiate the model class ``CortexClass`` and initialise the simulation parameters."""
        QThread.__init__(self)
        self.CC = CortexClass()

        self.percent = 0.0
        self.T = 0
        self.dt = 0
        self.Stim_Signals = []

    @pyqtSlot(float)
    def updatePercent(self, pourcent):
        """Slot receiving the simulation progress (fraction 0-1) and re-emitting it as ``updateTime``."""
        self.percent = pourcent
        self.updateTime.emit(self.percent)

    def __del__(self):
        """Wait for the thread to finish before destruction."""
        self.wait()

    def get_percentage(self):
        """Return the last progress value received."""
        return self.percent

    def run(self):
        """Run the simulation (QThread entry point; see the class note) and emit ``finished``."""
        self.t, self.pyrVs, self.pyrVd, self.olmVs, self.basketVs, self.BisVs = self.CC.Simulate(
            self.T, self.dt, self.Stim_Signals
        )

        self.finished.emit()

    def arret(self):
        """Intended to stop the simulation (see the class note: ``self.C`` does not exist)."""
        self.C.Stop = False


class ModelMicro_GUI(QMainWindow):
    """Main window of HippoCoMM.

    Layout: parameter panel (left, scrollable) | EEG viewer + LFP viewer (middle) |
    3D VTK view, connectivity-matrix view and stimulation view (right).

    Main attributes
        CC                  : network model object (``Colonne_cortical``) that holds the cells and runs the simulation.
        CellPosition        : (N, 3) array of cell coordinates (um).
        ConnectivityMatrix  : object array; element ``i`` = indices of the presynaptic cells of cell ``i``.
        List_C / List_Var   : number of connections / connection-distance spread (um) per (target, source) cell-type pair.
        List_Neurone_type   : cell-type code of each cell (see the module docstring).
        Sigs_dict           : dict of simulated membrane potentials ("t" + one entry per cell).
        LFP                 : last computed LFP signal.
    """

    def __init__(self, parent=None):
        """Build the window: model object, viewers, splitters, parameter panel and menu.

        ``parent`` is the QApplication (used to call ``processEvents`` during long computations).
        """
        super(ModelMicro_GUI, self).__init__()
        self.parent = parent

        if getattr(sys, 'frozen', False):
            self.application_path = os.path.dirname(sys.executable)
        elif __file__:
            self.application_path = os.path.dirname(__file__)
        sys.path.append(self.application_path)

        from Modeles.Wendling.Colonne_cortical_Wendling import Colonne_cortical

        self.Colonne = Colonne_cortical
        self.Colonne_cortical_Thread = Colonne_cortical_Thread(self.Colonne)
        self.CC = self.Colonne_cortical_Thread.CC
        self.CC.updateTime.something_happened.connect(self.updateTime)

        self.dt = 1 / 25
        self.T = 400
        self.nbEch = int(self.T / self.dt)

        self.Nb_of_PYR = 0
        self.Nb_of_BAS = 0
        self.Nb_of_SOM = 0
        self.Nb_of_BIS = 0
        self.List_PYR_Stimulated = []

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.electrode_pos = [0.0, 0.0, 0.0]

        self.createCells = True

        self.list_neurone_color = [
            QColor('darkRed').name(),  # pyramid
            QColor('darkGreen').name(),  # BAS
            QColor('darkBlue').name(),  # OLM
            QColor('darkYellow').name(),  # BIS
            QColor("#ffa500").name(),  # stim
            QColor('white').name(),  # electrode
        ]

        # some variable for widget's length
        self.Qbox_H = 30
        self.Qbox_W = 60
        self.Qedit_H = 30
        self.Qedit_W = 60

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        #### global layout
        self.mainHBOX = QHBoxLayout()
        self.setmarginandspacing(self.mainHBOX)
        ###

        wid_Param_VBOX = QWidget()
        self.Param_VBOX = QVBoxLayout()
        self.Param_VBOX.setAlignment(Qt.AlignmentFlag.AlignTop)
        wid_Param_VBOX.setLayout(self.Param_VBOX)
        wid_Param_VBOX.setMaximumWidth(350)
        self.setmarginandspacing(self.Param_VBOX)

        # call column layout
        self.set_Param_VBOX()

        self.Vsplitter_middle = QSplitter(Qt.Orientation.Vertical)
        self.mascene_EEGViewer = lfpViewer_EEG()
        self.mascene_LFPViewer = LFPViewer(self)
        self.Vsplitter_middle.addWidget(self.mascene_EEGViewer)
        self.Vsplitter_middle.addWidget(self.mascene_LFPViewer)
        self.Vsplitter_middle.setStretchFactor(4, 0)
        self.Vsplitter_middle.setStretchFactor(1, 0)
        self.Vsplitter_middle.setSizes([1500, 1500 // 4])

        self.Vsplitter = QSplitter(Qt.Orientation.Vertical)

        self.GraphWidget = QWidget()
        self.GraphLayout = QVBoxLayout()
        self.Graph_viewer = Graph_viewer3D_VTK(self)
        self.GraphinfoLayout = QHBoxLayout()
        labelr, r_e = Layout_grid_Label_Edit(label=['r'], edit=['50'])
        labell, l_e = Layout_grid_Label_Edit(label=['line'], edit=['5'])
        labels, s_e = Layout_grid_Label_Edit(label=['scale'], edit=['200'])
        self.r_e = r_e[0]
        self.l_e = l_e[0]
        self.s_e = s_e[0]
        self.RedrawVTK_PB = QPushButton('Redraw')
        self.GraphinfoLayout.addWidget(labelr)
        self.GraphinfoLayout.addWidget(self.r_e)
        self.GraphinfoLayout.addWidget(labell)
        self.GraphinfoLayout.addWidget(self.l_e)
        self.GraphinfoLayout.addWidget(labels)
        self.GraphinfoLayout.addWidget(self.s_e)
        self.GraphinfoLayout.addWidget(self.RedrawVTK_PB)
        self.GraphLayout.addLayout(self.GraphinfoLayout)
        self.GraphLayout.addWidget(self.Graph_viewer)
        self.GraphWidget.setLayout(self.GraphLayout)
        self.r_e.editingFinished.connect(self.updateVTKinfo)
        self.l_e.editingFinished.connect(self.updateVTKinfo)
        self.s_e.editingFinished.connect(self.updateVTKinfo)
        self.RedrawVTK_PB.clicked.connect(self.Graph_viewer.draw_Graph)

        self.masceneCM = CMViewer(self)
        self.VStimWidget = QWidget()
        self.VStimLayout = QVBoxLayout()
        self.masceneStim = StimViewer(self)
        self.DisplayStim_PB = QPushButton('DisplayStim')
        self.DisplayStim_PB.clicked.connect(self.DisplayStim_func)
        self.VStimLayout.addWidget(self.DisplayStim_PB)
        self.VStimLayout.addWidget(self.masceneStim)
        self.VStimWidget.setLayout(self.VStimLayout)
        self.Vsplitter.addWidget(self.GraphWidget)
        self.Vsplitter.addWidget(self.masceneCM)
        self.Vsplitter.addWidget(self.VStimWidget)
        self.Vsplitter.setStretchFactor(0, 0)
        self.Vsplitter.setStretchFactor(1, 0)
        self.Vsplitter.setStretchFactor(2, 0)
        self.Vsplitter.setStretchFactor(3, 0)
        self.Vsplitter.setSizes([1500, 1500, 1500, 1500])

        ### add  vectical global splitters
        self.mainsplitter = QSplitter(Qt.Orientation.Horizontal)

        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(wid_Param_VBOX)
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350 + 24)

        scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.mainsplitter.addWidget(self.Vsplitter_middle)
        self.mainsplitter.addWidget(self.Vsplitter)
        self.mainsplitter.setStretchFactor(0, 3)
        self.mainsplitter.setStretchFactor(1, 1)
        self.mainsplitter.setSizes([1500 * 3, 1500])
        layend = QHBoxLayout()
        layend.addWidget(scroll)
        layend.addWidget(self.mainsplitter)
        self.mainHBOX.addLayout(layend)

        self.centralWidget.setLayout(self.mainHBOX)

        # Menu
        # set actions
        extractLoad_Model = QAction("Load Model", self)
        extractLoad_Model.triggered.connect(self.LoadModel)

        extractLoad_Simul = QAction("Load Simulation", self)
        extractSave_Simul = QAction("Save Simulation", self)
        extractLoad_Simul.triggered.connect(self.LoadSimul)
        extractSave_Simul.triggered.connect(self.SaveSimul)

        extractLoad_Res = QAction("Load Results", self)
        extractSave_Res = QAction("Save Results", self)
        extractSave_Res.triggered.connect(self.SaveRes)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileLoad = menubar.addMenu('&Load/Save')
        fileLoad.addAction(extractLoad_Model)
        fileLoad.addSeparator()
        fileLoad.addAction(extractLoad_Simul)
        fileLoad.addAction(extractSave_Simul)
        fileLoad.addSeparator()
        fileLoad.addAction(extractSave_Res)

        self.NewModify1NMM = None
        self.NewModifyXNMM = None
        self.selected_cells = []

        self.update_cellNnumber()

    # ----------------------------------------------------------------------
    # Display helpers and progress feedback
    # ----------------------------------------------------------------------
    def updateVTKinfo(self):
        """Apply the "r" (sphere radius), "line" (line width) and "scale" fields to the 3D viewer."""
        self.Graph_viewer.LinewidthfromGUI = float(self.l_e.text())
        self.Graph_viewer.radiuswidthfromGUI = float(self.r_e.text())
        self.Graph_viewer.setScales(float(self.s_e.text()))

    @pyqtSlot(float)
    def updateTime(self, pourcent):
        """Slot updating the "Computation in progress" box with elapsed and estimated remaining time.

        ``pourcent`` is the fraction (0-1) of the simulation already computed.
        """
        ts = time.time() - self.t0
        tf = ts / (pourcent + 0.00000001)
        tr = tf - ts
        self.msg.setText(
            "Computation in progress\nPlease wait.\nTime spend : "
            + str(datetime.timedelta(seconds=int(ts)))
            + "\nTime remaining : "
            + str(datetime.timedelta(seconds=int(tr)))
        )
        self.parent.processEvents()

    def DisplayStim_func(self):
        """Generate the CA3 stimulation signals from the current parameters and plot them."""
        self.Generate_Stim_Signals()
        self.masceneStim.update()

    # ----------------------------------------------------------------------
    # Parameter panel (left column)
    # ----------------------------------------------------------------------
    def set_Param_VBOX(self):
        # Tissue size
        """Create all the group boxes of the left parameter panel and wire their signals.

        Sections: tissue size, cell-type percentages, connection tables (number and
        distance), cell placement, stimulation, simulation, electrode / LFP and the
        (currently hidden) synchronisation panel.
        """
        self.tissue_size_GB = QGroupBox(r'tissue size (µm)')
        labelX, x_e = Layout_grid_Label_Edit(label=['x'], edit=['300'])
        labelY, y_e = Layout_grid_Label_Edit(label=['y'], edit=['300'])
        labelZ, z_e = Layout_grid_Label_Edit(label=['z'], edit=['50'])
        self.x_e = x_e[0]
        self.y_e = y_e[0]
        self.z_e = z_e[0]
        labelDensity1 = QLabel('Density')
        labelDensity2 = QLabel(r'Neuron/mm<sup>3</sup>')
        self.Density_e = LineEdit('318400')
        labelnbcell = QLabel('Nb cells')
        self.nbcell_e = LineEdit_Int('1000')

        self.get_XYZ_PB = QPushButton('Get XYZ')
        self.get_Density_PB = QPushButton('Get Density')
        self.get_nbcell_PB = QPushButton('Get Nb cells')
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        grid.addWidget(self.get_XYZ_PB, 3, 0)
        grid.addWidget(labelDensity1, 0, 1)
        grid.addWidget(labelDensity2, 1, 1)
        grid.addWidget(self.Density_e, 2, 1)
        grid.addWidget(self.get_Density_PB, 3, 1)
        grid.addWidget(labelnbcell, 1, 2)
        grid.addWidget(self.nbcell_e, 2, 2)
        grid.addWidget(self.get_nbcell_PB, 3, 2)
        self.get_XYZ_PB.clicked.connect(lambda state, s='get_XYZ_PB': self.get_dimension_func(s))
        self.get_Density_PB.clicked.connect(lambda state, s='get_Density_PB': self.get_dimension_func(s))
        self.get_nbcell_PB.clicked.connect(lambda state, s='get_nbcell_PB': self.get_dimension_func(s))
        self.tissue_size_GB.setLayout(grid)

        # Cell-type percentages
        self.pourcentageCell_GB = QGroupBox(r'% de cell')
        labelPYR = QLabel('PYR')
        labelBAS = QLabel('BAS')
        labelOLM = QLabel('OLM')
        labelBIS = QLabel('BIS')
        labelSUM = QLabel('Sum')
        labelPYRSTIM = QLabel('PYRSTIM')
        self.Per_PYR = LineEdit('82')
        self.Per_BAS = LineEdit('6')
        self.Per_SOM = LineEdit('6')
        self.Per_BIS = LineEdit('6')
        self.labelSUM2 = QLabel('100')
        self.Per_PYR_stimulated = LineEdit('22')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(labelPYR, 0, 0)
        grid.addWidget(labelBAS, 0, 1)
        grid.addWidget(labelOLM, 0, 2)
        grid.addWidget(labelBIS, 0, 3)
        grid.addWidget(labelSUM, 0, 4)
        grid.addWidget(labelPYRSTIM, 0, 5)
        grid.addWidget(self.Per_PYR, 1, 0)
        grid.addWidget(self.Per_BAS, 1, 1)
        grid.addWidget(self.Per_SOM, 1, 2)
        grid.addWidget(self.Per_BIS, 1, 3)
        grid.addWidget(self.labelSUM2, 1, 4)
        grid.addWidget(self.Per_PYR_stimulated, 1, 5)
        self.pourcentageCell_GB.setLayout(grid)

        [
            s.textChanged.connect(self.update_percent_sum)
            for s in [self.Per_PYR, self.Per_BAS, self.Per_SOM, self.Per_BIS]
        ]

        # Number of cells per type
        self.Apply_percentage_PB = QPushButton('Apply Percentage')
        self.Nb_of_PYR_l = LineEdit_Int('0')
        self.Nb_of_BAS_l = LineEdit_Int('0')
        self.Nb_of_SOM_l = LineEdit_Int('0')
        self.Nb_of_BIS_l = LineEdit_Int('0')
        self.Nb_of_PYR_Stimulated_l = LineEdit_Int('0')
        emptylabel = QLabel('    ')
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(self.Apply_percentage_PB, 2, 1, 1, 3)
        grid.addWidget(self.Nb_of_PYR_l, 3, 0)
        grid.addWidget(self.Nb_of_BAS_l, 3, 1)
        grid.addWidget(self.Nb_of_SOM_l, 3, 2)
        grid.addWidget(self.Nb_of_BIS_l, 3, 3)
        grid.addWidget(emptylabel, 3, 4)
        grid.addWidget(self.Nb_of_PYR_Stimulated_l, 3, 5)
        self.Apply_percentage_PB.clicked.connect(self.update_cellNnumber)

        # Connection-number table
        self.Nb_VAr_connexion_TW = QTabWidget()
        self.Nb_VAr_connexion_TW.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.Nb_connexion_GB = QWidget()
        self.Nb_VAr_connexion_TW.addTab(self.Nb_connexion_GB, r'Connection number')
        label_source = QLabel('Source')
        label_Target = QLabel('T\na\nr\ng\ne\nt')
        label_PYR1 = QLabel('PYR')
        label_PYR2 = QLabel('PYR')
        label_PYR_STIM = QLabel('PYR_STIM')
        label_BAS1 = QLabel('BAS')
        label_BAS2 = QLabel('BAS')
        label_OLM1 = QLabel('OLM')
        label_OLM2 = QLabel('OLM')
        label_BIS1 = QLabel('BIS')
        label_BIS2 = QLabel('BIS')
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label_PYR1, 2, 1)
        grid.addWidget(label_BAS1, 3, 1)
        grid.addWidget(label_OLM1, 4, 1)
        grid.addWidget(label_BIS1, 5, 1)
        grid.addWidget(label_PYR2, 1, 2)
        grid.addWidget(label_BAS2, 1, 3)
        grid.addWidget(label_OLM2, 1, 4)
        grid.addWidget(label_BIS2, 1, 5)
        grid.addWidget(label_PYR_STIM, 1, 6)
        # Default number of presynaptic partners of a cell, per (target type, source type) pair.
        # Rows    = target cell type: PYR, BAS, OLM(SOM), BIS.
        # Columns = source cell type: PYR, BAS, OLM(SOM), BIS, PYR_STIM (CA3 input).
        # N >= 1 : N partners; 0 < N < 1 : probability of one extra partner (see Create_Connectivity_Matrices_sparse).
        self.List_C = np.array(
            [
                [2, 5, 5, 5, 1],  # target PYR
                [2, 5, 5, 0, 0],  # target BAS
                [3, 0, 0, 0, 0],  # target SOM
                [0, 0, 0, 0, 1],
            ],
            dtype=float,
        )  # target BIS
        self.List_C_e = []
        for l in range(len(self.List_C)):
            for c in range(len(self.List_C[0])):
                edit = LineEdit(str(self.List_C[l, c]))
                self.List_C_e.append(edit)
                grid.addWidget(edit, l + 2, c + 2)
        self.Nb_connexion_GB.setLayout(grid)

        # Connection-distance table
        self.List_Var_GB = QWidget()
        self.Nb_VAr_connexion_TW.addTab(self.List_Var_GB, r"intercellular connection distance (µm)")
        label_source = QLabel('Source')
        label_Target = QLabel('T\na\nr\ng\ne\nt')
        label_PYR1 = QLabel('PYR')
        label_PYR2 = QLabel('PYR')
        label_BAS1 = QLabel('BAS')
        label_BAS2 = QLabel('BAS')
        label_OLM1 = QLabel('OLM')
        label_OLM2 = QLabel('OLM')
        label_BIS1 = QLabel('BIS')
        label_BIS2 = QLabel('BIS')
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid.setContentsMargins(
            5,
            5,
            5,
            5,
        )
        grid.addWidget(label_source, 0, 1, 1, 5, Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(label_Target, 1, 0, 5, 1, Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label_PYR1, 2, 1)
        grid.addWidget(label_BAS1, 3, 1)
        grid.addWidget(label_OLM1, 4, 1)
        grid.addWidget(label_BIS1, 5, 1)
        grid.addWidget(label_PYR2, 1, 2)
        grid.addWidget(label_BAS2, 1, 3)
        grid.addWidget(label_OLM2, 1, 4)
        grid.addWidget(label_BIS2, 1, 5)
        # Spread (um) of the Gaussian that favours close partners, per (target type, source type) pair
        # (rows = target PYR, BAS, OLM(SOM), BIS; columns = source PYR, BAS, OLM(SOM), BIS).
        self.List_Var = np.array(
            [
                [20, 500, 500, 500],  # target PYR
                [700, 700, 700, 700],  # target BAS
                [840, 840, 840, 840],  # target SOM
                [700, 700, 700, 700],
            ]
        )
        self.List_Var_e = []
        for l in range(len(self.List_Var)):
            for c in range(len(self.List_Var)):
                edit = LineEdit_Int(str(self.List_Var[l, c]))
                self.List_Var_e.append(edit)
                grid.addWidget(edit, l + 2, c + 2)
        self.List_Var_GB.setLayout(grid)

        # Cell placement
        self.cell_placement_GB = QGroupBox(r"cell placement")
        self.cell_placement_CB = QComboBox()
        list = [
            'grille 2D',
            'grille 2D isolated CA3',
            'grille 3D',
            'grille 3D isolated CA3',
            'random2D / Square',
            'random2D / Square isolated CA3',
            'random3D / Square',
            'random3D / Square isolated CA3',
            'BestCandidate / Square',
            'BestCandidate / Square isolated CA3',
            'BestCandidate / cubic',
            'BestCandidate / cubic isolated CA3',
        ]
        self.cell_placement_CB.addItems(list)
        self.cell_placement_PB = QPushButton('Place cells')
        self.cell_scalesize_PB = QPushButton('scale size')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(self.cell_placement_CB, 0, 0, 1, 1)
        grid.addWidget(
            self.cell_placement_PB,
            0,
            1,
            1,
            2,
        )
        grid.addWidget(
            self.cell_scalesize_PB,
            0,
            3,
            1,
            2,
        )
        self.cell_placement_GB.setLayout(grid)
        self.cell_placement_PB.clicked.connect(self.PlaceCell_func)
        self.cell_scalesize_PB.clicked.connect(self.scalesize_func)

        # Stimulation parameters
        self.Stimulation_parameters_GB = QGroupBox(r"Stimulation parameters")
        label1 = QLabel('Stim duration (ms)')
        label2 = QLabel(r"Injected current density (µA/cm<sup>2</sup>)")
        label3 = QLabel("Time constant RC (ms)")
        label4 = QLabel("Stim number per cell")
        label5 = QLabel("Jitter variance (ms)")
        label6 = QLabel("seed")
        label7 = QLabel("Stim type")
        self.StimDuration_e = LineEdit('3')
        self.i_inj_e = LineEdit('30')
        self.tau_e = LineEdit('4')
        self.nbStim_e = LineEdit_Int('5')
        self.varianceStim_e = LineEdit('12')
        self.seed_e = LineEdit('0')
        self.stimtype_CB = QComboBox()
        self.stimtype_CB.addItems(['Burst', 'Continuous', 'Periodic'])
        label8 = QLabel("Period (ms, Periodic)")
        label9 = QLabel("Stim start (ms, Periodic)")
        self.StimPeriode_e = LineEdit('200')
        self.StimStart_e = LineEdit('30')
        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0)
        grid.addWidget(self.StimDuration_e, 0, 1)
        grid.addWidget(label2, 1, 0)
        grid.addWidget(self.i_inj_e, 1, 1)
        grid.addWidget(label3, 2, 0)
        grid.addWidget(self.tau_e, 2, 1)
        grid.addWidget(label4, 3, 0)
        grid.addWidget(self.nbStim_e, 3, 1)
        grid.addWidget(label5, 4, 0)
        grid.addWidget(self.varianceStim_e, 4, 1)
        grid.addWidget(label6, 6, 0)
        grid.addWidget(self.seed_e, 6, 1)
        grid.addWidget(label7, 7, 0)
        grid.addWidget(self.stimtype_CB, 7, 1)
        grid.addWidget(label8, 8, 0)
        grid.addWidget(self.StimPeriode_e, 8, 1)
        grid.addWidget(label9, 9, 0)
        grid.addWidget(self.StimStart_e, 9, 1)
        self.Stimulation_parameters_GB.setLayout(grid)

        # Simulation parameters
        self.Simulation_parameters_GB = QGroupBox(r"Simulation parameters")
        label1 = QLabel('Simulation duration (ms)')
        label2 = QLabel(r"Sampling frequency (kHz)")
        self.SimDuration_e = LineEdit('300')
        self.Fs_e = LineEdit('25')
        self.UpdateModel_PB = QPushButton('Update Model')
        self.ModifyModel_PB = QPushButton('All Model Param')
        self.ModifyXModel_PB = QPushButton('Modify X Models')
        self.Reset_states_PB = QPushButton('Reset states')
        self.Run_PB = QPushButton('Run')
        self.displaycurves_CB = QCheckBox('Display Curves')
        self.displayVTK_CB = QCheckBox('Display VTK')
        self.displaycurve_per_e = LineEdit('100')
        displaycurve_per_l = QLabel(r'% to plot')
        self.displaycurve_per_e = LineEdit('30')

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label1, 0, 0, 1, 2)
        grid.addWidget(self.SimDuration_e, 0, 2)
        grid.addWidget(label2, 1, 0, 1, 2)
        grid.addWidget(self.Fs_e, 1, 2)
        grid.addWidget(self.UpdateModel_PB, 0, 3)
        grid.addWidget(self.ModifyModel_PB, 1, 3)
        grid.addWidget(self.ModifyXModel_PB, 2, 3)
        grid.addWidget(self.Run_PB, 3, 3)
        grid.addWidget(self.Reset_states_PB, 3, 2)
        grid.addWidget(self.displaycurves_CB, 3, 0)
        grid.addWidget(self.displayVTK_CB, 2, 0)
        grid.addWidget(displaycurve_per_l, 4, 0)
        grid.addWidget(self.displaycurve_per_e, 4, 1)
        self.Simulation_parameters_GB.setLayout(grid)
        self.Run_PB.clicked.connect(self.simulate)
        self.UpdateModel_PB.clicked.connect(self.update_Model)
        self.ModifyModel_PB.clicked.connect(self.modify_Model)
        self.ModifyXModel_PB.clicked.connect(self.ModXNMMclicked)
        self.Reset_states_PB.clicked.connect(self.Reset_states_clicked)
        self.displaycurves_CB.stateChanged.connect(self.displaycurves_CB_fonc)
        self.displaycurves_CB.setChecked(True)
        self.displayVTK_CB.stateChanged.connect(self.displayVTK_CB_fonc)
        self.displayVTK_CB.setChecked(True)

        # Electrode placement and LFP
        self.electrode_placement_GB = QGroupBox(r"electrode position")
        label_x = QLabel('x y z')
        label_y = QLabel('y')
        label_z = QLabel('z')
        self.electrode_x_e = LineEdit('320')
        self.electrode_y_e = LineEdit('320')
        self.electrode_z_e = LineEdit('-225')
        self.Compute_LFP_PB = QPushButton('Comp')
        self.Compute_LFP2_type_CB = QComboBox()
        self.Compute_LFP2_type_CB.addItems(["CSD", "Dipole"])
        self.electrod_disk_CB = QComboBox()
        self.electrod_disk_CB.addItems(['Point', 'Disk', 'Cylinder'])
        self.electrode_radius_e = LineEdit('50')
        self.electrode_angle1_e = LineEdit('0')
        self.electrode_angle2_e = LineEdit('0')
        self.electrode_h_e = LineEdit('1')

        self.lfpplot_filtre_Active = QCheckBox('Active')
        self.lfpplot_filtre_order = LineEdit_Int('1')
        self.lfpplot_filtre_FCL = LineEdit('1')
        self.lfpplot_filtre_FCH = LineEdit('1000')

        grid = QGridLayout()
        grid.setContentsMargins(5, 5, 5, 5)
        grid.addWidget(label_x, 0, 0)
        grid.addWidget(self.electrode_x_e, 0, 1)
        grid.addWidget(self.electrode_y_e, 0, 2)
        grid.addWidget(self.electrode_z_e, 0, 3)
        grid.addWidget(self.Compute_LFP2_type_CB, 0, 4)
        grid.addWidget(self.Compute_LFP_PB, 0, 5)

        grid.addWidget(self.electrod_disk_CB, 1, 0)
        grid.addWidget(QLabel('r \u03b81 \u03b82 h'), 1, 1)
        grid.addWidget(self.electrode_radius_e, 1, 2)
        grid.addWidget(self.electrode_angle1_e, 1, 3)
        grid.addWidget(self.electrode_angle2_e, 1, 4)
        grid.addWidget(self.electrode_h_e, 1, 5)

        grid.addWidget(self.lfpplot_filtre_Active, 2, 0)
        grid.addWidget(QLabel("Order"), 2, 1)
        grid.addWidget(QLabel("L(Hz)"), 2, 2)
        grid.addWidget(QLabel("H(Hz)"), 2, 3)
        grid.addWidget(self.lfpplot_filtre_order, 3, 1)
        grid.addWidget(self.lfpplot_filtre_FCL, 3, 2)
        grid.addWidget(self.lfpplot_filtre_FCH, 3, 3)

        self.electrode_placement_GB.setLayout(grid)
        [
            x.editingFinished.connect(self.electrode_placement_func)
            for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]
        ]
        [
            x.editingFinished.connect(self.electrode_placement_func)
            for x in [self.electrode_radius_e, self.electrode_angle1_e, self.electrode_angle2_e, self.electrode_h_e]
        ]
        self.electrod_disk_CB.currentIndexChanged.connect(self.electrode_placement_func)
        self.Compute_LFP_PB.clicked.connect(self.Compute_LFP_fonc)

        # Constants of the dipole LFP model (see Compute_LFP_fonc): soma and dendrite sizes, extracellular
        # conductivity ``sigma``, coupling conductance ``gc`` and soma/total surface ratio ``p``.
        # NOTE: 10e-3 equals 0.01, so ``15.0 * 10e-3`` is 0.15, not 15 um expressed in mm; check the intended units.
        self.somaSize = 15.0 * 10e-3  # 15 micrometres = 15 * 10e-3 mm
        self.dendriteSize = 300.0 * 10e-3  # 800  micrometres = 400 * 10e-3 mm
        self.sigma = 33.0 * 10e-5
        self.gc = 10.0e-5
        self.p = 0.15

        # Synchrony analysis panel (not added to the window, see end of method)
        self.synchro_GB = QGroupBox(r"synchronisation algorithm")
        self.synchro_l = QVBoxLayout()
        self.synchro_method_CB = QComboBox()
        list = [
            'Thresholding',
            'AutoCorrelation',
            'Nearest delay',
            'ISI-distance',
            'van Rossum distance',
            'Victor Purpura distance',
        ]
        self.synchro_method_CB.addItems(list)

        self.synchro_l.addWidget(self.synchro_method_CB)
        self.synchro_widget = QWidget()
        self.synchro_l.addWidget(self.synchro_widget)
        self.update_synchro_method()

        self.synchro_GB.setLayout(self.synchro_l)
        self.synchro_method_CB.currentTextChanged.connect(self.update_synchro_method)

        self.Param_VBOX.addWidget(self.tissue_size_GB)
        self.Param_VBOX.addWidget(self.pourcentageCell_GB)
        self.Param_VBOX.addWidget(self.Nb_VAr_connexion_TW)
        self.Param_VBOX.addWidget(self.cell_placement_GB)
        self.Param_VBOX.addWidget(self.Stimulation_parameters_GB)
        self.Param_VBOX.addWidget(self.Simulation_parameters_GB)
        self.Param_VBOX.addWidget(self.electrode_placement_GB)
        self.Param_VBOX.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

    # ----------------------------------------------------------------------
    # Synchrony analysis (panel currently not shown)
    # ----------------------------------------------------------------------
    def update_synchro_method(self):
        """Rebuild the parameter widgets of the synchronisation panel for the selected method.

        NOTE: the synchronisation panel is not added to the window (see the end of
        ``set_Param_VBOX``); the analysis code is kept for optional use.
        """
        self.synchro_l.removeWidget(self.synchro_widget)
        self.synchro_widget.deleteLater()
        self.synchro_widget = None

        if self.synchro_method_CB.currentText() == 'Thresholding':
            self.synchro_widget = QWidget()
            label_Thresholding = QLabel('Threshold (mV)')
            self.Synchro_Thresholding_Threshold_e = LineEdit('0')
            label_Thresholding_perc = QLabel('Area over (%)')
            self.Synchro_Thresholding_Threshold_perc_e = LineEdit('10')
            self.Synchro_Thresholding_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_Thresholding, 0, 0)
            grid.addWidget(self.Synchro_Thresholding_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_Thresholding_PB, 0, 2)
            grid.addWidget(label_Thresholding_perc, 0, 3)
            grid.addWidget(self.Synchro_Thresholding_Threshold_perc_e, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_Thresholding_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'AutoCorrelation':
            self.synchro_widget = QWidget()
            label_AutoCorrelation = QLabel('Threshold (mV)')
            self.Synchro_AutoCorrelation_Threshold_e = LineEdit('0')
            label_AutoCorrelation_perc = QLabel('Area over (%)')
            self.Synchro_AutoCorrelation_Threshold_perc_e = LineEdit('10')
            self.Synchro_AutoCorrelation_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_AutoCorrelation, 0, 0)
            grid.addWidget(self.Synchro_AutoCorrelation_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_AutoCorrelation_PB, 0, 2)
            grid.addWidget(label_AutoCorrelation_perc, 0, 3)
            grid.addWidget(self.Synchro_AutoCorrelation_Threshold_perc_e, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_AutoCorrelation_PB.clicked.connect(self.Synchro_Method_PB_Apply)

        elif self.synchro_method_CB.currentText() == 'Nearest delay':
            self.synchro_widget = QWidget()
            label_NearestDelay = QLabel('Delta')
            self.Synchro_NearestDelay_Threshold_e = LineEdit('50')
            self.Synchro_NearestDelay_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_NearestDelay, 0, 0)
            grid.addWidget(self.Synchro_NearestDelay_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_NearestDelay_PB, 0, 2)
            self.synchro_widget.setLayout(grid)
            self.Synchro_NearestDelay_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'ISI-distance':
            self.synchro_widget = QWidget()
            label_ISI = QLabel('Delta')
            self.Synchro_ISI_Threshold_e = LineEdit('50')
            self.Synchro_ISI_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_ISI, 0, 0)
            grid.addWidget(self.Synchro_ISI_Threshold_e, 0, 1)
            grid.addWidget(self.Synchro_ISI_PB, 0, 2)
            self.synchro_widget.setLayout(grid)
            self.Synchro_ISI_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'van Rossum distance':
            self.synchro_widget = QWidget()
            label_vanRossum_Tau = QLabel('Tau (s)')
            label_vanRossum_delta = QLabel('delta (mV)')
            self.Synchro_vanRossum_Tau_e = LineEdit('1')
            self.Synchro_vanRossum_delta_e = LineEdit('50')
            self.Synchro_vanRossum_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_vanRossum_delta, 0, 0)
            grid.addWidget(self.Synchro_vanRossum_delta_e, 0, 1)
            grid.addWidget(label_vanRossum_Tau, 0, 2)
            grid.addWidget(self.Synchro_vanRossum_Tau_e, 0, 3)
            grid.addWidget(self.Synchro_vanRossum_PB, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_vanRossum_PB.clicked.connect(self.Synchro_Method_PB_Apply)
        elif self.synchro_method_CB.currentText() == 'Victor Purpura distance':
            self.synchro_widget = QWidget()
            label_VictorPurpura_q = QLabel('q (Hz)')
            label_VictorPurpura_delta = QLabel('delta (mV)')
            self.Synchro_VictorPurpura_q_e = LineEdit('1')
            self.Synchro_VictorPurpura_delta_e = LineEdit('50')
            self.Synchro_VictorPurpura_PB = QPushButton('Apply')
            grid = QGridLayout()
            grid.addWidget(label_VictorPurpura_delta, 0, 0)
            grid.addWidget(self.Synchro_VictorPurpura_delta_e, 0, 1)
            grid.addWidget(label_VictorPurpura_q, 0, 2)
            grid.addWidget(self.Synchro_VictorPurpura_q_e, 0, 3)
            grid.addWidget(self.Synchro_VictorPurpura_PB, 0, 4)
            self.synchro_widget.setLayout(grid)
            self.Synchro_VictorPurpura_PB.clicked.connect(self.Synchro_Method_PB_Apply)

        self.synchro_l.insertWidget(1, self.synchro_widget)

    def Synchro_Method_PB_Apply(self):
        """Compute and plot the synchrony measure selected in the synchronisation panel.

        Requires a finished simulation (``Sigs_dict``). Only the CA1 pyramidal cells
        (type 1) are analysed. Methods: Thresholding, AutoCorrelation, Nearest delay,
        ISI-distance, van Rossum distance, Victor-Purpura distance.
        """
        try:
            self.Sigs_dict
        except:
            return
        if self.synchro_method_CB.currentText() == 'Thresholding':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            threshold = float(self.Synchro_Thresholding_Threshold_e.text())
            threshold_perc = float(self.Synchro_Thresholding_Threshold_perc_e.text()) / 100
            sigbin = array_Sigs > threshold

            Aps = []
            for i in range(sigbin.shape[0]):
                Ap = 0
                for j in range(sigbin.shape[1] - 1):
                    if sigbin[i, j] == False and sigbin[i, j + 1] == True:
                        Ap += 1
                Aps.append(Ap)
            Aps = int(np.sum(np.array(Aps)))

            synchro_sig = np.sum(sigbin, axis=0)
            synchro_sig = synchro_sig / array_Sigs.shape[0]

            sigbin_perc = np.zeros(synchro_sig.shape) * np.nan
            threshold_perc_sig = synchro_sig > threshold_perc
            selection = np.ones(len(threshold_perc_sig), dtype=bool)
            selection[1:] = threshold_perc_sig[1:] != threshold_perc_sig[:-1]
            selection[0] = False
            selection &= threshold_perc_sig != 0
            threshold_perc_NB = np.sum(threshold_perc_sig[selection])

            sigbin_perc[synchro_sig > threshold_perc] = synchro_sig[synchro_sig > threshold_perc]

            sigbin_perc_sum = np.nansum(sigbin_perc - threshold_perc) * (t[1] - t[0])

            chaine = (
                'Total number of APs on Pyr = '
                + str(Aps)
                + ', Nb passage over'
                + self.Synchro_Thresholding_Threshold_perc_e.text()
                + '% = '
                + str(threshold_perc_NB)
                + ', Area over'
                + self.Synchro_Thresholding_Threshold_perc_e.text()
                + '% = '
                + str(sigbin_perc_sum)
            )
            self.mascene_LFPViewer.update_synchro_thresholding(
                synchro_sig, sigbin_perc, shiftT=self.CC.tps_start - self.T, titre=chaine
            )
        elif self.synchro_method_CB.currentText() == 'AutoCorrelation':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            threshold = float(self.Synchro_AutoCorrelation_Threshold_e.text())
            threshold_perc = float(self.Synchro_AutoCorrelation_Threshold_perc_e.text()) / 100
            sigbin = array_Sigs > threshold

            Aps = []
            for i in range(sigbin.shape[0]):
                Ap = 0
                for j in range(sigbin.shape[1] - 1):
                    if sigbin[i, j] == False and sigbin[i, j + 1] == True:
                        Ap += 1
                Aps.append(Ap)
            Aps = int(np.sum(np.array(Aps)))

            synchro_sig = np.sum(sigbin, axis=0)
            synchro_sig = synchro_sig / array_Sigs.shape[0]

            sigbin_perc_autocorrel = signal.correlate(synchro_sig, synchro_sig, mode='full')

            chaine = 'Total number of APs on Pyr = ' + str(Aps)

            self.mascene_LFPViewer.update_synchro_AutoCorrelation(
                synchro_sig, sigbin_perc_autocorrel, shiftT=self.CC.tps_start - self.T, titre=chaine
            )

        elif self.synchro_method_CB.currentText() == 'Nearest delay':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            delta = float(self.Synchro_NearestDelay_Threshold_e.text())

            #                 pass
            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                else:
                    Aps_centers.append(np.array([]))

            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]

            Aps_delays = []
            meanDelay = []
            for i in range(len(Aps_centers)):
                if not Aps_centers[i] == []:
                    meanDelay.append(np.diff(np.array(Aps_centers[i])))

                    delaysline = []
                    for j in range(len(Aps_centers[i])):
                        delays = []
                        for k in range(len(Aps_centers)):
                            if not k == i:
                                if not len(Aps_centers[k]) == 0:
                                    found_delay = find_nearest(Aps_centers[k], Aps_centers[i][j])
                                    delays.append(abs(Aps_centers[i][j] - found_delay))
                        delaysline.append(delays)
                    Aps_delays.append(delaysline)
                else:
                    Aps_delays.append([])

            meanDelay = np.mean(np.concatenate(meanDelay).ravel())

            Aps_delays_flatten = []
            for i in range(len(Aps_delays)):
                if not Aps_delays[i] == []:
                    Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
                else:
                    Aps_delays_flatten.append([])
            meanDiffDelay = np.mean(np.concatenate(Aps_delays_flatten).ravel())

            Aps_gauss_fit = []

            def gaus(x, a, sigma):
                return a * np.exp(-((x) ** 2) / (2 * sigma**2))

            Aps_gauss_fit = []
            for i in range(len(Aps_delays_flatten)):
                if len(Aps_delays_flatten[i]) > 0:
                    h = np.histogram(
                        Aps_delays_flatten[i],
                        bins=int(np.max(Aps_delays_flatten[i])),
                        range=(0, int(np.max(Aps_delays_flatten[i]))),
                    )
                    x = np.hstack((-h[1][-2:0:-1], h[1][:-1]))
                    y = np.hstack((+h[0][-1:0:-1], h[0][:]))
                    popt, pcov = curve_fit(gaus, x, y, p0=[1, 1])

                    Aps_gauss_fit.append(popt[1])

            meangauss_fit = np.mean(Aps_gauss_fit)
            self.mascene_LFPViewer.update_synchro_NearestDelay2(
                Aps_centers,
                Aps_delays,
                Aps_gauss_fit,
                shiftT=self.CC.tps_start - self.T,
                titre='Delay Mean = '
                + str(meanDelay)
                + ', Delay Diff Mean = '
                + str(meanDiffDelay)
                + ',  gauss fit sigma = '
                + str(meangauss_fit),
            )

        elif self.synchro_method_CB.currentText() == 'ISI-distance':
            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            dt = t[1] - t[0]
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)
            delta = float(self.Synchro_ISI_Threshold_e.text())

            Aps = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps.append(maxi[:, 0].astype(int))
                else:
                    Aps.append(np.array([]))

            # ISI distance
            array_ISI = np.zeros(array_Sigs.shape) * np.NAN
            for i in range(len(Aps)):
                for j in range(len(Aps[i]) - 1):
                    array_ISI[i, Aps[i][j] : Aps[i][j + 1]] = t[Aps[i][j + 1]] - t[Aps[i][j]]

            array_ratio = np.zeros((array_ISI.shape[0], array_ISI.shape[0], array_ISI.shape[1]))
            array_ratio = self.ISI_ratio(array_ISI, array_ratio)

            IntegralI = np.zeros((array_ISI.shape[0], array_ISI.shape[0])) * np.NAN
            for i in range(array_ISI.shape[0]):  # xisi
                for j in range(array_ISI.shape[0]):  # y
                    if not j == i:
                        IntegralI[i, j] = np.nansum(np.abs(array_ratio[i, j, :])) / np.sum(
                            np.isnan(array_ratio[i, j, :])
                        )
                        if IntegralI[i, j] == 0.0:
                            IntegralI[i, j] = np.NAN

            self.mascene_LFPViewer.update_synchro_ISI_distance(IntegralI, titre='ISI_distance')

        elif self.synchro_method_CB.currentText() == 'van Rossum distance':
            delta = float(self.Synchro_vanRossum_delta_e.text())
            Tau = float(self.Synchro_vanRossum_Tau_e.text())

            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)

            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                else:
                    Aps_centers.append(np.array([]))

            if Tau <= 0:
                spike_counts = [len(st) for st in Aps_centers]
                vr_dist = np.sqrt(spike_counts + np.atleast_2d(spike_counts).T)
            elif Tau == np.inf:
                spike_counts = [len(st) for st in Aps_centers]
                vr_dist = np.absolute(spike_counts - np.atleast_2d(spike_counts).T)
            else:
                k_dist = self._summed_dist_matrix(Aps_centers, Tau, True)
                vr_dist = np.empty_like(k_dist)
                for i, j in np.ndindex(k_dist.shape):
                    vr_dist[i, j] = k_dist[i, i] + k_dist[j, j] - k_dist[i, j] - k_dist[j, i]
                vr_dist = np.sqrt(vr_dist)

            self.mascene_LFPViewer.update_synchro_vanRossum_distance(vr_dist, titre='vanRossum_distance')

        elif self.synchro_method_CB.currentText() == 'Victor Purpura distance':
            delta = float(self.Synchro_VictorPurpura_delta_e.text())
            q = float(self.Synchro_VictorPurpura_q_e.text())

            Sigs_dict = copy.deepcopy(self.Sigs_dict)
            t = Sigs_dict.pop('t')
            keys = list(Sigs_dict.keys())
            array_Sigs = []
            for i, v in enumerate(self.List_Neurone_type):
                if v == 1:
                    array_Sigs.append(Sigs_dict[keys[i]])
            array_Sigs = np.array(array_Sigs)

            Aps_centers = []
            for i in range(array_Sigs.shape[0]):
                maxi, mini = peakdet(array_Sigs[i, :], delta)
                if len(maxi) > 0:
                    Aps_centers.append(t[maxi[:, 0].astype(int)])
                else:
                    Aps_centers.append(np.array([]))

            if q <= 0:
                num_spikes = np.atleast_2d([len(st) for st in Aps_centers])
                vp_dist = np.absolute(num_spikes.T - num_spikes)
            elif q == np.inf:
                num_spikes = np.atleast_2d([len(st) for st in Aps_centers])
                vp_dist = num_spikes.T + num_spikes
            else:
                shape = (len(Aps_centers), len(Aps_centers))
                vp_dist = np.empty(shape)
                for i in range(shape[0]):
                    for j in range(i, shape[1]):
                        vp_dist[i, j] = vp_dist[j, i] = self._victor_purpura_dist_for_st_pair_intuitive(
                            Aps_centers[i], Aps_centers[j], q=q
                        )
            self.mascene_LFPViewer.update_synchro_VictorPurpura_distance(vp_dist, titre='VictorPurpura_distance')

    @staticmethod
    def _victor_purpura_dist_for_st_pair_intuitive(train_a, train_b, q=1.0):
        """Victor-Purpura spike-train distance between two spike trains (dynamic programming, cost ``q`` per unit time shift)."""
        nspk_a = len(train_a)
        nspk_b = len(train_b)
        scr = np.zeros((nspk_a + 1, nspk_b + 1))
        scr[:, 0] = range(0, nspk_a + 1)
        scr[0, :] = range(0, nspk_b + 1)

        if nspk_a > 0 and nspk_b > 0:
            for i in range(1, nspk_a + 1):
                for j in range(1, nspk_b + 1):
                    scr[i, j] = min(scr[i - 1, j] + 1, scr[i, j - 1] + 1)
                    scr[i, j] = min(
                        scr[i, j], scr[i - 1, j - 1] + np.float64((q * abs(train_a[i - 1] - train_b[j - 1])))
                    )
        return scr[nspk_a, nspk_b]

    def _summed_dist_matrix(self, spiketrains, tau, presorted=False):
        # The algorithm underlying this implementation is described in
        # Houghton, C., & Kreuz, T. (2012). On the efficient calculation of van
        # Rossum distances. Network: Computation in Neural Systems, 23(1-2),
        # 48-58. We would like to remark that in this paper in formula (9) the
        # left side of the equation should be divided by two.
        #
        # Given N spiketrains with n entries on average the run-time complexity is
        # O(N^2 * n). O(N^2 + N * n) memory will be needed.

        """Matrix of summed van Rossum distances (exponential kernel, time constant ``tau``) between spike trains.

        Efficient algorithm of Houghton & Kreuz (2012), see the comments inside.
        """
        if len(spiketrains) <= 0:
            return np.zeros((0, 0))

        if not presorted:
            spiketrains = [v.copy() for v in spiketrains]
            for v in spiketrains:
                v.sort()

        sizes = np.asarray([v.size for v in spiketrains])
        values = np.empty((len(spiketrains), max(1, sizes.max())))
        values.fill(np.nan)
        for i, v in enumerate(spiketrains):
            if v.size > 0:
                values[i, : v.size] = v / tau

        exp_diffs = np.exp(values[:, :-1] - values[:, 1:])
        markage = np.zeros(values.shape)
        for u in range(len(spiketrains)):
            markage[u, 0] = 0
            for i in range(sizes[u] - 1):
                markage[u, i + 1] = (markage[u, i] + 1.0) * exp_diffs[u, i]

        # Same spiketrain terms
        D = np.empty((len(spiketrains), len(spiketrains)))
        D[np.diag_indices_from(D)] = sizes + 2.0 * np.sum(markage, axis=1)

        # Cross spiketrain terms
        for u in range(D.shape[0]):
            all_ks = np.searchsorted(values[u], values, 'left') - 1
            for v in range(u):
                js = np.searchsorted(values[v], values[u], 'right') - 1
                ks = all_ks[v]
                slice_j = np.s_[np.searchsorted(js, 0) : sizes[u]]
                slice_k = np.s_[np.searchsorted(ks, 0) : sizes[v]]
                D[u, v] = np.sum(np.exp(values[v][js[slice_j]] - values[u][slice_j]) * (1.0 + markage[v][js[slice_j]]))
                D[u, v] += np.sum(np.exp(values[u][ks[slice_k]] - values[v][slice_k]) * (1.0 + markage[u][ks[slice_k]]))
                D[v, u] = D[u, v]

        return D

    @staticmethod
    @njit
    def ISI_ratio(array_ISI, array_ratio):
        """Pairwise instantaneous ISI ratio of all spike trains (numba-compiled).

        Fills and returns ``array_ratio[i, j, t]`` (signed, symmetric in i/j) from the
        instantaneous inter-spike intervals ``array_ISI[i, t]``.
        """
        for i in range(array_ISI.shape[0]):  # xisi
            for j in range(array_ISI.shape[0]):  # y
                if not j > i:
                    for tt in range(array_ISI.shape[1]):
                        if array_ISI[i, tt] <= array_ISI[j, tt]:
                            array_ratio[i, j, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                            array_ratio[j, i, tt] = array_ISI[i, tt] / array_ISI[j, tt] - 1
                        else:
                            array_ratio[i, j, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
                            array_ratio[j, i, tt] = -(array_ISI[j, tt] / array_ISI[i, tt] - 1)
        return array_ratio

    # ----------------------------------------------------------------------
    # Network construction: cell placement and connectivity
    # ----------------------------------------------------------------------
    def get_new_List_C(self):
        """Read the "Connection number" table of the GUI into ``self.List_C``."""
        for l in range(len(self.List_C)):
            for c in range(len(self.List_C[0])):
                self.List_C[l, c] = float(self.List_C_e[l * len(self.List_C[0]) + c].text())

    def get_new_List_Var(self):
        """Read the "intercellular connection distance" table of the GUI into ``self.List_Var``."""
        for l in range(len(self.List_Var)):
            for c in range(len(self.List_Var)):
                self.List_Var[l, c] = float(self.List_Var_e[l * len(self.List_Var) + c].text())

    def PlaceCell_func(self):
        """ "Place cells" button: place the cells in space, then build the connectivity.

        Steps: read tissue size / cell numbers / connection tables, generate
        ``CellPosition`` according to the selected placement algorithm, compute the
        connectivity (``Create_Connectivity_Matrices_sparse``), set the cell names and
        colours, and refresh the 3D view.
        """
        self.x = float(self.x_e.text())
        self.y = float(self.y_e.text())
        self.z = float(self.z_e.text())
        self.take_cell_number()
        self.get_new_List_C()
        self.get_new_List_Var()

        placement = self.cell_placement_CB.currentText()
        print('placement')
        # Cell placement. Every branch fills ``self.CellPosition`` with one [x, y, z] position per cell, in the order
        # of ``List_Neurone_type`` (the stimulated CA3 cells come first). In the 'isolated CA3' variants these cells
        # are put at negative x so that they form a separate block next to the CA1 tissue.
        if placement == 'grille 2D':
            self.CellPosition = []
            l = 0
            c = 0
            ratio = self.x / self.y
            cy = int(np.sqrt(self.Nb_of_PYR_BAS_SOM_BIS_sum[3] / ratio))
            space = self.y / cy
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):
                self.CellPosition.append([c * space, l * space, 0.0])
                l += 1
                if l > cy:
                    c += 1
                    l = 0
            random.shuffle(self.CellPosition)
        elif placement == 'grille 2D isolated CA3':
            self.CellPosition = []

            ratio = self.x / self.y
            cy = int(np.sqrt((self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated) / ratio))
            space = self.y / cy
            l = 0
            c = -1
            for i in range(self.Nb_of_PYR_Stimulated):
                self.CellPosition.append([c * space, l * space, 0.0])
                l += 1
                if l > cy:
                    c -= 1
                    l = 0
            l = 0
            c = 0
            for i in range((self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated)):
                self.CellPosition.append([c * space, l * space, 0.0])
                l += 1
                if l > cy:
                    c += 1
                    l = 0
            copylist = self.CellPosition[self.Nb_of_PYR_Stimulated :]
            random.shuffle(copylist)
            self.CellPosition[self.Nb_of_PYR_Stimulated :] = copylist
        elif placement == 'grille 3D':
            self.CellPosition = []
            l = 0
            c = 0
            p = 0

            volume = self.x * self.y * self.z
            dist = (volume / self.Nb_of_PYR_BAS_SOM_BIS_sum[3]) ** (1.0 / 3)
            cx = int(self.x / dist)
            cy = int(self.y / dist)
            cz = int(self.z / dist)
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):
                self.CellPosition.append([c * dist, l * dist, p * dist])
                l += 1
                if l > cy:
                    c += 1
                    l = 0
                    if c > cx:
                        p += 1
                        c = 0
            random.shuffle(self.CellPosition)
        elif placement == 'grille 3D isolated CA3':
            self.CellPosition = []
            volume = self.x * self.y * self.z
            dist = (volume / (self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated)) ** (1.0 / 3)
            cx = int(self.x / dist)
            cy = int(self.y / dist)
            cz = int(self.z / dist)
            l = 0
            c = -1
            p = 0
            for i in range(self.Nb_of_PYR_Stimulated):
                self.CellPosition.append([c * dist, l * dist, p * dist])
                l += 1
                if l > cy:
                    p += 1
                    l = 0
                    if p > cz:
                        c -= 1
                        p = 0
            l = 0
            c = 0
            p = 0
            for i in range((self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated)):
                self.CellPosition.append([c * dist, l * dist, p * dist])
                l += 1
                if l > cy:
                    c += 1
                    l = 0
                    if c > cx:
                        p += 1
                        c = 0
            copylist = self.CellPosition[self.Nb_of_PYR_Stimulated :]
            random.shuffle(copylist)
            self.CellPosition[self.Nb_of_PYR_Stimulated :] = copylist

        elif placement == 'random2D / Square':
            self.CellPosition = []
            for fd in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):
                self.CellPosition.append(
                    np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, 0.0]])[:, 0]
                )

        elif placement == 'random3D / Square':
            self.CellPosition = []
            for fd in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):
                self.CellPosition.append(
                    np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, self.z]])[:, 0]
                )

        elif placement == 'random2D / Square isolated CA3':
            ratio = self.Nb_of_PYR_BAS_SOM_BIS_sum[3] / self.Nb_of_PYR_Stimulated
            self.CellPosition = []
            for fd in range(self.Nb_of_PYR_Stimulated):
                self.CellPosition.append(
                    np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [-self.x / ratio, self.y, 0.0]])[
                        :, 0
                    ]
                )
            for fd in range((self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated)):
                self.CellPosition.append(
                    np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, 0.0]])[:, 0]
                )

        elif placement == 'random3D / Square isolated CA3':
            self.CellPosition = []
            ratio = self.Nb_of_PYR_BAS_SOM_BIS_sum[3] / self.Nb_of_PYR_Stimulated
            for fd in range(self.Nb_of_PYR_Stimulated):
                self.CellPosition.append(
                    np.array(
                        [np.random.uniform(low=0.0, high=val, size=1) for val in [-self.x / ratio, self.y, self.z]]
                    )[:, 0]
                )
            for fd in range((self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - self.Nb_of_PYR_Stimulated)):
                self.CellPosition.append(
                    np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, self.z]])[:, 0]
                )

        # Best-candidate placement (Mitchell): each new cell is the one, among 20 random candidates, that is farthest
        # from its nearest already-placed cell. This gives a roughly uniform spacing without a regular grid.
        elif placement == 'BestCandidate / Square':
            self.CellPosition = []
            self.CellPosition.append(
                np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, 0.0]])[:, 0]
            )
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1):
                candidate = np.array(
                    [
                        np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, 0.0]])[:, 0]
                        for fd in range(20)
                    ]
                )
                CellDistances = distance.cdist(self.CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                self.CellPosition.append(list(candidate[argmax, :]))
            random.shuffle(self.CellPosition)
        elif placement == 'BestCandidate / Square isolated CA3':
            ratio = self.Nb_of_PYR_BAS_SOM_BIS_sum[3] / self.Nb_of_PYR_Stimulated
            self.CellPosition = []
            self.CellPosition.append(
                np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [-self.x / ratio, self.y, 0.0]])[:, 0]
            )
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1):
                if i < self.Nb_of_PYR_Stimulated - 1:
                    candidate = np.array(
                        [
                            np.array(
                                [np.random.uniform(low=0.0, high=val, size=1) for val in [-self.x / ratio, self.y, 0.0]]
                            )[:, 0]
                            for fd in range(20)
                        ]
                    )
                else:
                    candidate = np.array(
                        [
                            np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, 0.0]])[
                                :, 0
                            ]
                            for fd in range(20)
                        ]
                    )
                CellDistances = distance.cdist(self.CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                self.CellPosition.append(list(candidate[argmax, :]))

        elif placement == 'BestCandidate / cubic':
            self.CellPosition = []
            self.CellPosition.append(
                np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, self.z]])[:, 0]
            )
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1):
                candidate = np.array(
                    [
                        np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, self.z]])[
                            :, 0
                        ]
                        for fd in range(20)
                    ]
                )
                CellDistances = distance.cdist(self.CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                self.CellPosition.append(list(candidate[argmax, :]))
            random.shuffle(self.CellPosition)

        elif placement == 'BestCandidate / cubic isolated CA3':
            ratio = self.Nb_of_PYR_BAS_SOM_BIS_sum[3] / self.Nb_of_PYR_Stimulated
            self.CellPosition = []
            self.CellPosition.append(
                np.array([np.random.uniform(low=0.0, high=val, size=1) for val in [-self.x / ratio, self.y, self.z]])[
                    :, 0
                ]
            )
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1):
                if i < self.Nb_of_PYR_Stimulated - 1:
                    candidate = np.array(
                        [
                            np.array(
                                [
                                    np.random.uniform(low=0.0, high=val, size=1)
                                    for val in [-self.x / ratio, self.y, self.z]
                                ]
                            )[:, 0]
                            for fd in range(20)
                        ]
                    )
                else:
                    candidate = np.array(
                        [
                            np.array(
                                [np.random.uniform(low=0.0, high=val, size=1) for val in [self.x, self.y, self.z]]
                            )[:, 0]
                            for fd in range(20)
                        ]
                    )
                CellDistances = distance.cdist(self.CellPosition, candidate, 'euclidean')
                argmin = np.argmin(CellDistances, axis=0)
                valmin = [CellDistances[k, j] for j, k in enumerate(argmin)]
                argmax = np.argmax(valmin)
                self.CellPosition.append(list(candidate[argmax, :]))

        self.CellPosition = np.array(self.CellPosition)
        self.createCells = True

        # Placement done: build the connectivity and the default names/colours of the cells.
        print('Create_Connectivity_Matrices')
        self.Create_Connectivity_Matrices_sparse()
        print(self.ConnectivityMatrix)
        print('List_Names')
        self.List_Names = (
            ['PYR'] * self.Nb_of_PYR + ['BAS'] * self.Nb_of_BAS + ['SOM'] * self.Nb_of_SOM + ['BIS'] * self.Nb_of_BIS
        )
        self.List_Colors = (
            [self.list_neurone_color[0]] * self.Nb_of_PYR
            + [self.list_neurone_color[1]] * self.Nb_of_BAS
            + [self.list_neurone_color[2]] * self.Nb_of_SOM
            + [self.list_neurone_color[3]] * self.Nb_of_BIS
        )

        print('List_PYR_Stimulated')
        for cell in self.List_PYR_Stimulated:
            self.List_Names[cell] = 'PYR_STIM'
            self.List_Colors[cell] = self.list_neurone_color[4]
        print('electrode_placement_func')
        self.electrode_placement_func()

    def scalesize_func(self):
        """Ask for x/y/z scale factors and rescale the tissue size and all cell positions (connectivity is kept)."""
        exPopup = Rescalesize(self, x=self.x_e.text(), y=self.y_e.text(), z=self.z_e.text())
        if exPopup.exec() == QDialog.DialogCode.Accepted:
            print('Accepted')
            xs = float(exPopup.xs_e.text())
            ys = float(exPopup.ys_e.text())
            zs = float(exPopup.zs_e.text())
            self.x_e.setText(str(float(self.x_e.text()) * xs))
            self.y_e.setText(str(float(self.y_e.text()) * ys))
            self.z_e.setText(str(float(self.z_e.text()) * zs))

            self.CellPosition = self.CellPosition * np.array([xs, ys, zs])

            self.update_graph()

        else:
            print('Cancelled')
        exPopup.deleteLater()

    def update_graph(self):
        """Redraw the 3D view (if enabled)."""
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()
            self.Graph_viewer.set_center()
            self.parent.processEvents()

    def electrode_placement_func(self):
        """Read the electrode position and shape fields and redraw the 3D view.

        Sets ``electrode_pos`` ([x, y, z], um) and ``electrode_disk``
        ([shape, radius, angle1, angle2, height]).
        """
        self.electrode_pos = [float(x.text()) for x in [self.electrode_x_e, self.electrode_y_e, self.electrode_z_e]]
        self.electrode_disk = [
            self.electrod_disk_CB.currentText(),
            float(self.electrode_radius_e.text()),
            float(self.electrode_angle1_e.text()),
            float(self.electrode_angle2_e.text()),
            float(self.electrode_h_e.text()),
        ]
        self.update_graph()

    def bruitGaussien(self, s, m):
        """Return one Gaussian random value with mean ``m`` and standard deviation ``s``."""
        return np.random.normal(m, s)

    def Create_Connectivity_Matrices_sparse(self):
        """Build the connectivity of the network (list of presynaptic cells per cell).

        For every cell ``i`` (loops over PYR, BAS, SOM and BIS targets) and every source
        type, the number of presynaptic partners is taken from ``List_C``:
            * integer part N      : N partners, chosen with a distance-dependent Gaussian
              rule (``pickcell_gauss_fun``, spread given by ``List_Var``);
            * fractional part p   : one additional random partner with probability p;
            * the stimulated CA3 cells are picked uniformly (``pickcell_simple_fun``).
        The result is stored in ``self.ConnectivityMatrix``; the AMPA/GABA presynaptic
        lists are then computed and the connectivity view is refreshed.
        """
        print('ConnectivityMatrix')
        cordinate_target_source = [[] for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[3])]
        print('CellDistances')
        self.CellDistances = distance.cdist(self.CellPosition, self.CellPosition, 'euclidean')

        print('Nb_of_PYR')
        if self.Nb_of_PYR > 0:
            for i in range(self.Nb_of_PYR):
                if i in self.List_PYR_Stimulated:
                    continue

                pickedcell_list = []
                loop = int(self.List_C[0, 0])
                if loop > (self.Nb_of_PYR - self.Nb_of_PYR_Stimulated) - 1:
                    loop = (self.Nb_of_PYR - self.Nb_of_PYR_Stimulated) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[0, 0] - int(self.List_C[0, 0])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_Stimulated, self.Nb_of_PYR_BAS_SOM_BIS_sum[0])]
                    )

                loop = int(self.List_C[0, 1])
                if loop > self.Nb_of_BAS:
                    loop = self.Nb_of_BAS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[0, 1] - int(self.List_C[0, 1])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1])]
                    )

                loop = int(self.List_C[0, 2])
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                ajout = self.List_C[0, 2] - int(self.List_C[0, 2])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2])]
                    )

                loop = int(self.List_C[0, 3])
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[0, 3] - int(self.List_C[0, 3])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[3], self.Nb_of_PYR_BAS_SOM_BIS_sum[3])]
                    )

                # Connections coming from the stimulated CA3 PYR cells
                loop = int(self.List_C[0, 4])
                if loop > self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_simple_fun(size, loop, 0, self.Nb_of_PYR_Stimulated, pickedcell)
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[0, 4] - int(self.List_C[0, 4])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append([np.random.randint(0, self.Nb_of_PYR_Stimulated)])

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    cordinate_target_source[i].append(k)

        print('Nb_of_BAS')
        if self.Nb_of_BAS > 0:
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1]):

                pickedcell_list = []
                loop = int(self.List_C[1, 0])
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[1, 0] - int(self.List_C[1, 0])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_Stimulated, self.Nb_of_PYR_BAS_SOM_BIS_sum[0])]
                    )

                loop = int(self.List_C[1, 1])
                if loop > (self.Nb_of_BAS) - 1:
                    loop = (self.Nb_of_BAS) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[1, 1] - int(self.List_C[1, 1])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1])]
                    )

                loop = int(self.List_C[1, 2])
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[1, 2] - int(self.List_C[1, 2])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2])]
                    )

                loop = int(self.List_C[1, 3])
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[1, 3] - int(self.List_C[1, 3])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[2], self.Nb_of_PYR_BAS_SOM_BIS_sum[3])]
                    )

                # Connections coming from the stimulated CA3 PYR cells
                loop = int(self.List_C[1, 4])
                if loop > self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_simple_fun(size, loop, 0, self.Nb_of_PYR_Stimulated, pickedcell)
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[1, 4] - int(self.List_C[1, 4])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append([np.random.randint(0, self.Nb_of_PYR_Stimulated)])

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    cordinate_target_source[i].append(k)

        print('Nb_of_OLM')
        if self.Nb_of_SOM > 0:
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2]):

                pickedcell_list = []
                loop = int(self.List_C[2, 0])
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[2, 0] - int(self.List_C[2, 0])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_Stimulated, self.Nb_of_PYR_BAS_SOM_BIS_sum[0])]
                    )

                loop = int(self.List_C[2, 1])
                if loop > self.Nb_of_BAS:
                    loop = self.Nb_of_BAS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[2, 1] - int(self.List_C[2, 1])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1])]
                    )

                loop = int(self.List_C[2, 2])
                if loop > (self.Nb_of_SOM) - 1:
                    loop = (self.Nb_of_SOM) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[2, 2] - int(self.List_C[2, 2])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2])]
                    )

                loop = int(self.List_C[2, 3])
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[2, 3] - int(self.List_C[2, 3])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[2], self.Nb_of_PYR_BAS_SOM_BIS_sum[3])]
                    )

                # Connections coming from the stimulated CA3 PYR cells
                loop = int(self.List_C[2, 4])
                if loop > self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_simple_fun(size, loop, 0, self.Nb_of_PYR_Stimulated, pickedcell)
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[2, 4] - int(self.List_C[2, 4])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append([np.random.randint(0, self.Nb_of_PYR_Stimulated)])

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    cordinate_target_source[i].append(k)

        print('Nb_of_BIS')
        if self.Nb_of_BIS > 0:
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[2], self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):

                pickedcell_list = []
                loop = int(self.List_C[3, 0])
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[3, 0] - int(self.List_C[3, 0])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_Stimulated, self.Nb_of_PYR_BAS_SOM_BIS_sum[0])]
                    )

                loop = int(self.List_C[3, 1])
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[3, 1] - int(self.List_C[3, 1])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1])]
                    )

                loop = int(self.List_C[3, 2])
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[3, 2] - int(self.List_C[3, 2])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2])]
                    )

                loop = int(self.List_C[3, 3])
                if loop > (self.Nb_of_BIS) - 1:
                    loop = (self.Nb_of_BIS) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[3, 3] - int(self.List_C[3, 3])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append(
                        [np.random.randint(self.Nb_of_PYR_BAS_SOM_BIS_sum[2], self.Nb_of_PYR_BAS_SOM_BIS_sum[3])]
                    )

                # Connections coming from the stimulated CA3 PYR cells
                loop = int(self.List_C[3, 4])
                if loop > self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_simple_fun(size, loop, 0, self.Nb_of_PYR_Stimulated, pickedcell)
                    pickedcell_list.append(pickedcell)
                ajout = self.List_C[3, 4] - int(self.List_C[3, 4])
                randval = np.random.uniform()
                if randval < ajout:
                    pickedcell_list.append([np.random.randint(0, self.Nb_of_PYR_Stimulated)])

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    cordinate_target_source[i].append(k)
        self.ConnectivityMatrix = np.array([np.array(c) for c in cordinate_target_source], dtype='object')

        print('Compute_synaptic_connections')
        self.Compute_synaptic_connections_sparse()

        print('update fig')
        self.masceneCM.update()
        print('update done')

    def Create_Connectivity_Matrices(self):
        """Dense-matrix version of ``Create_Connectivity_Matrices_sparse`` (not used by the current GUI)."""
        print('ConnectivityMatrix')
        self.ConnectivityMatrix = np.zeros((self.Nb_of_PYR_BAS_SOM_BIS_sum[3], self.Nb_of_PYR_BAS_SOM_BIS_sum[3]))
        # tirer n nombre parmis nombre de neurones / Tirage Aléatoire par loi gaussienne

        print('CellDistances')
        self.CellDistances = distance.cdist(self.CellPosition, self.CellPosition, 'euclidean')

        print('Nb_of_PYR')
        if self.Nb_of_PYR > 0:
            for i in range(self.Nb_of_PYR):

                if i in self.List_PYR_Stimulated:
                    loop = 3
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_simple_fun(
                        size, loop, self.Nb_of_PYR_Stimulated, self.Nb_of_PYR_BAS_SOM_BIS_sum[0], pickedcell
                    )
                    for k in pickedcell:
                        self.ConnectivityMatrix[k, i] = 1
                    continue

                pickedcell_list = []
                loop = self.List_C[0, 0]
                if loop > (self.Nb_of_PYR - self.Nb_of_PYR_Stimulated) - 1:
                    loop = (self.Nb_of_PYR - self.Nb_of_PYR_Stimulated) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[0, 1]
                if loop > self.Nb_of_BAS:
                    loop = self.Nb_of_BAS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[0, 2]
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[0, 3]
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[0, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    self.ConnectivityMatrix[i, k] = 1

        print('Nb_of_BAS')
        if self.Nb_of_BAS > 0:
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[0], self.Nb_of_PYR_BAS_SOM_BIS_sum[1]):

                pickedcell_list = []
                loop = self.List_C[1, 0]
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[1, 1]
                if loop > (self.Nb_of_BAS) - 1:
                    loop = (self.Nb_of_BAS) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[1, 2]
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[1, 3]
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[1, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    self.ConnectivityMatrix[i, k] = 1

        print('Nb_of_OLM')
        if self.Nb_of_SOM > 0:
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[1], self.Nb_of_PYR_BAS_SOM_BIS_sum[2]):

                pickedcell_list = []
                loop = self.List_C[2, 0]
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[2, 1]
                if loop > self.Nb_of_BAS:
                    loop = self.Nb_of_BAS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[2, 2]
                if loop > (self.Nb_of_SOM) - 1:
                    loop = (self.Nb_of_SOM) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[2, 3]
                if loop > self.Nb_of_BIS:
                    loop = self.Nb_of_BIS
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[2, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    self.ConnectivityMatrix[i, k] = 1

        print('Nb_of_BIS')
        if self.Nb_of_BIS > 0:
            count = 0
            for i in range(self.Nb_of_PYR_BAS_SOM_BIS_sum[2], self.Nb_of_PYR_BAS_SOM_BIS_sum[3]):
                if count < self.Nb_of_BIS and count < self.Nb_of_PYR_Stimulated:
                    pickedcell_list = []
                    pickedcell_list.append([count])  # connection entre cell CA3 est BIS
                    count += 1
                else:
                    pickedcell_list = []

                loop = self.List_C[3, 0]
                if loop > self.Nb_of_PYR - self.Nb_of_PYR_Stimulated:
                    loop = self.Nb_of_PYR - self.Nb_of_PYR_Stimulated
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(self.CellDistances[i, self.Nb_of_PYR_Stimulated : self.Nb_of_PYR_BAS_SOM_BIS_sum[0]])
                        + self.Nb_of_PYR_Stimulated
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 0], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[3, 1]
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[0] : self.Nb_of_PYR_BAS_SOM_BIS_sum[1]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 1], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[3, 2]
                if loop > self.Nb_of_SOM:
                    loop = self.Nb_of_SOM
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[1] : self.Nb_of_PYR_BAS_SOM_BIS_sum[2]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    )
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 2], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                loop = self.List_C[3, 3]
                if loop > (self.Nb_of_BIS) - 1:
                    loop = (self.Nb_of_BIS) - 1
                if not loop <= 0:
                    sortedIndex = (
                        np.argsort(
                            self.CellDistances[i, self.Nb_of_PYR_BAS_SOM_BIS_sum[2] : self.Nb_of_PYR_BAS_SOM_BIS_sum[3]]
                        )
                        + self.Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    )
                    sortedIndex = np.delete(sortedIndex, 0)
                    pickedcell = np.array([-1] * loop)
                    size = np.array([-1] * loop)
                    pickcell_gauss_fun(
                        size, loop, 0, self.List_Var[3, 3], self.CellDistances[i, :], sortedIndex, pickedcell
                    )
                    pickedcell_list.append(pickedcell)

                for k in [item for sublist in pickedcell_list for item in sublist]:
                    self.ConnectivityMatrix[i, k] = 1

        print('Compute_synaptic_connections')
        self.Compute_synaptic_connections()

        print('update fig')
        self.masceneCM.update()
        print('update done')

    def Compute_synaptic_connections_sparse(self):
        """Derive the synaptic lists of each cell from ``ConnectivityMatrix`` (object-array version).

        For each cell, presynaptic cells of type 1/2 (pyramidal) give AMPA synapses and
        interneurons give GABA synapses. For each synapse a 0/1 flag is also stored
        (``PreSynaptic_Soma_Dend_*``: 1 = somatic, 0 = dendritic target compartment;
        interneurons are single-compartment, hence always 1) together with its
        complement (``*_not``). These lists are passed to the model in ``update_Model``.
        """
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_Dend_AMPA = []
        self.PreSynaptic_Soma_Dend_AMPA_not = []
        self.PreSynaptic_Soma_Dend_GABA = []
        self.PreSynaptic_Soma_Dend_GABA_not = []
        for i, c in enumerate(self.List_Neurone_type):
            convect = self.ConnectivityMatrix[i]
            convect_AMPA = []
            convect_GABA = []
            convect_Soma_Dend_AMPA = []
            convect_Soma_Dend_GABA = []
            for k in convect:
                if self.List_Neurone_type[k] in [1, 2]:  # presynaptic cell is CA1 or CA3 (excitatory)
                    convect_AMPA.append(k)
                    if c in [3, 4, 5]:  # target is an interneuron
                        convect_Soma_Dend_AMPA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # presynaptic cell is CA1, CA3, SOM or BIS
                            convect_Soma_Dend_AMPA.append(0)
                        else:  # presynaptic cell is BAS
                            convect_Soma_Dend_AMPA.append(1)

                else:  # presynaptic cell is an interneuron (inhibitory)
                    convect_GABA.append(k)
                    if c in [3, 4, 5]:  # target is an interneuron
                        convect_Soma_Dend_GABA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # presynaptic cell is CA1, CA3, SOM or BIS
                            convect_Soma_Dend_GABA.append(0)
                        else:  # presynaptic cell is BAS
                            convect_Soma_Dend_GABA.append(1)

            self.PreSynaptic_Cell_AMPA.append(convect_AMPA)
            self.PreSynaptic_Cell_GABA.append(convect_GABA)
            self.PreSynaptic_Soma_Dend_AMPA.append(np.array(convect_Soma_Dend_AMPA))
            self.PreSynaptic_Soma_Dend_AMPA_not.append(np.abs(np.array(convect_Soma_Dend_AMPA) - 1))
            self.PreSynaptic_Soma_Dend_GABA.append(np.array(convect_Soma_Dend_GABA))
            self.PreSynaptic_Soma_Dend_GABA_not.append(np.abs(np.array(convect_Soma_Dend_GABA) - 1))

    def Compute_synaptic_connections(self):
        """Same as ``Compute_synaptic_connections_sparse`` for a dense 0/1 ``ConnectivityMatrix``."""
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_Dend_AMPA = []
        self.PreSynaptic_Soma_Dend_AMPA_not = []
        self.PreSynaptic_Soma_Dend_GABA = []
        self.PreSynaptic_Soma_Dend_GABA_not = []
        for i, c in enumerate(self.List_Neurone_type):
            convect = self.ConnectivityMatrix[i, :]
            convect = np.where(convect == 1)[0]
            convect_AMPA = []
            convect_GABA = []
            convect_Soma_Dend_AMPA = []
            convect_Soma_Dend_GABA = []
            for k in convect:
                if self.List_Neurone_type[k] in [1, 2]:  # presynaptic cell is CA1 or CA3 (excitatory)
                    convect_AMPA.append(k)
                    if c in [3, 4, 5]:  # target is an interneuron
                        convect_Soma_Dend_AMPA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # presynaptic cell is CA1, CA3, SOM or BIS
                            convect_Soma_Dend_AMPA.append(0)
                        else:  # presynaptic cell is BAS
                            convect_Soma_Dend_AMPA.append(1)

                else:  # presynaptic cell is an interneuron (inhibitory)
                    convect_GABA.append(k)
                    if c in [3, 4, 5]:  # target is an interneuron
                        convect_Soma_Dend_GABA.append(1)
                    else:
                        if self.List_Neurone_type[k] in [1, 2, 4, 5]:  # presynaptic cell is CA1, CA3, SOM or BIS
                            convect_Soma_Dend_GABA.append(0)
                        else:  # presynaptic cell is BAS
                            convect_Soma_Dend_GABA.append(1)

            self.PreSynaptic_Cell_AMPA.append(convect_AMPA)
            self.PreSynaptic_Cell_GABA.append(convect_GABA)
            self.PreSynaptic_Soma_Dend_AMPA.append(np.array(convect_Soma_Dend_AMPA))
            self.PreSynaptic_Soma_Dend_AMPA_not.append(np.abs(np.array(convect_Soma_Dend_AMPA) - 1))
            self.PreSynaptic_Soma_Dend_GABA.append(np.array(convect_Soma_Dend_GABA))
            self.PreSynaptic_Soma_Dend_GABA_not.append(np.abs(np.array(convect_Soma_Dend_GABA) - 1))

    # ----------------------------------------------------------------------
    # Cell numbers and tissue dimensions
    # ----------------------------------------------------------------------
    def update_cellNnumber(self):
        """ "Apply Percentage": compute the number of cells of each type from the total and the percentages."""
        cellnumber = int(self.nbcell_e.text())
        self.Nb_of_PYR = int(cellnumber * float(self.Per_PYR.text()) / 100)
        self.Nb_of_BAS = int(cellnumber * float(self.Per_BAS.text()) / 100)
        self.Nb_of_SOM = int(cellnumber * float(self.Per_SOM.text()) / 100)
        self.Nb_of_BIS = int(cellnumber * float(self.Per_BIS.text()) / 100)
        self.Nb_of_PYR_Stimulated = int(self.Nb_of_PYR * float(self.Per_PYR_stimulated.text()) / 100)

        self.Nb_of_PYR_l.setText(str(self.Nb_of_PYR))
        self.Nb_of_BAS_l.setText(str(self.Nb_of_BAS))
        self.Nb_of_SOM_l.setText(str(self.Nb_of_SOM))
        self.Nb_of_BIS_l.setText(str(self.Nb_of_BIS))
        self.Nb_of_PYR_Stimulated_l.setText(str(self.Nb_of_PYR_Stimulated))

        self.createCells = True

    def take_cell_number(self):
        """Read the number of cells of each type from the GUI and build ``List_Neurone_type`` and ``List_PYR_Stimulated``.

        Also flags ``createCells`` when a number changed, so the model cells are re-created.
        """
        if not int(self.Nb_of_PYR_l.text()) == self.Nb_of_PYR:
            self.createCells = True
        self.Nb_of_PYR = int(self.Nb_of_PYR_l.text())
        if not int(self.Nb_of_BAS_l.text()) == self.Nb_of_BAS:
            self.createCells = True
        self.Nb_of_BAS = int(self.Nb_of_BAS_l.text())
        if not int(self.Nb_of_SOM_l.text()) == self.Nb_of_SOM:
            self.createCells = True
        self.Nb_of_SOM = int(self.Nb_of_SOM_l.text())
        if not int(self.Nb_of_BIS_l.text()) == self.Nb_of_BIS:
            self.createCells = True
        self.Nb_of_BIS = int(self.Nb_of_BIS_l.text())
        if not int(self.Nb_of_PYR_Stimulated_l.text()) == self.Nb_of_PYR_Stimulated:
            self.createCells = True
        self.Nb_of_PYR_Stimulated = int(self.Nb_of_PYR_Stimulated_l.text())

        self.Nb_of_PYR_BAS_SOM_BIS_sum = [
            self.Nb_of_PYR,
            self.Nb_of_PYR + self.Nb_of_BAS,
            self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM,
            self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM + self.Nb_of_BIS,
        ]

        # Cell-type codes: 2 = stimulated CA3 PYR, 1 = CA1 PYR, 3 = BAS, 4 = SOM (OLM), 5 = BIS.
        self.List_Neurone_type = (
            [2] * self.Nb_of_PYR_Stimulated
            + [1] * (self.Nb_of_PYR - self.Nb_of_PYR_Stimulated)
            + [3] * self.Nb_of_BAS
            + [4] * self.Nb_of_SOM
            + [5] * self.Nb_of_BIS
        )
        self.List_PYR_Stimulated = [i for i in range(self.Nb_of_PYR_Stimulated)]

    def update_percent_sum(self):
        """Show the sum of the four percentages; highlight it in red when it is not 100."""
        sum = (
            float(self.Per_PYR.text())
            + float(self.Per_BAS.text())
            + float(self.Per_SOM.text())
            + float(self.Per_BIS.text())
        )
        self.labelSUM2.setText(str(sum))
        if sum == 100:
            self.labelSUM2.setStyleSheet("QLabel { background-color : none}")
        else:
            self.labelSUM2.setStyleSheet("QLabel { background-color : red}")

    def get_dimension_func(self, s):
        """Compute one of (x/y/z size, density, number of cells) from the other two.

        ``s`` is the name of the button pressed: 'get_XYZ_PB', 'get_Density_PB' or 'get_nbcell_PB'.
        Density is in neurons/mm^3 and sizes in um.
        """
        print('vf')
        if s == 'get_XYZ_PB':
            try:
                nbcell = int(self.nbcell_e.text())
                x = nbcell ** (1 / 3)
                space = 1 / (float(self.Density_e.text()) * 1e-6) ** (1 / 3)
                self.x_e.setText(str(x * space))
                self.y_e.setText(str(x * space))
                self.z_e.setText(str(x * space))
            except:
                msg_cri('To compute XYZ, the Density and the number of cell must be correct')

        elif s == 'get_Density_PB':
            try:
                x = float(self.x_e.text())
                y = float(self.y_e.text())
                z = float(self.z_e.text())
                nbcell = int(self.nbcell_e.text())
                if (x * y * z) == 0:
                    self.Density_e.setText('0')
                else:
                    self.Density_e.setText(str(int(nbcell / (x * y * z / 1e9))))
            except:
                msg_cri('To compute Density, the XYZ sizes and the number of cell must be correct')
        elif s == 'get_nbcell_PB':
            try:
                x = float(self.x_e.text())
                y = float(self.y_e.text())
                z = float(self.z_e.text())
                Density = float(self.Density_e.text())
                self.nbcell_e.setText(str(int(((x * y * z) / Density * 1e-9))))
            except:
                msg_cri('To compute the number of cell, the XYZ sizes and the Density must be correct')

    # ----------------------------------------------------------------------
    # Stimulation and model management
    # ----------------------------------------------------------------------
    def Generate_Stim_Signals(self):
        """Read the stimulation fields and generate the CA3 stimulation signals into ``self.Stim_Signals``.

        'Burst' and 'Continuous' use the numba generator ``Generate_Stim_Signals``; 'Periodic' uses
        ``Generate_Periodic_Stim_Signals`` (period and start from the two extra fields).

        Also updates ``dt``, ``T`` and ``nbEch`` from the sampling frequency and the simulation duration.
        """
        self.nbOfSamplesStim = int(float(self.StimDuration_e.text()) / self.dt)
        self.i_inj = float(self.i_inj_e.text())
        self.tau = float(self.tau_e.text())
        self.nbStim = int(self.nbStim_e.text())
        self.varianceStim = int(float(self.varianceStim_e.text()) / self.dt)
        self.seed = int(self.seed_e.text())

        nb_Stim_Signals = len(self.List_PYR_Stimulated)

        self.dt = 1 / float(self.Fs_e.text())
        self.T = float(self.SimDuration_e.text())
        self.nbEch = int(self.T / self.dt)

        if self.stimtype_CB.currentText() == 'Periodic':
            # Periodic stimulation: jitter, period and start are given in ms
            Stim_Signals_out = Generate_Periodic_Stim_Signals(
                nb_Stim_Signals,
                self.nbEch,
                self.dt,
                self.seed,
                float(self.StimDuration_e.text()),
                self.i_inj,
                self.tau,
                self.nbStim,
                float(self.varianceStim_e.text()),
                float(self.StimStart_e.text()),
                float(self.StimPeriode_e.text()),
            )
        else:
            Stim_Signals_in = np.zeros((nb_Stim_Signals, self.nbEch))
            Stim_Signals_out = np.zeros((nb_Stim_Signals, self.nbEch))

            Generate_Stim_Signals(
                Stim_Signals_in,
                self.seed,
                self.nbOfSamplesStim,
                self.i_inj,
                self.tau,
                self.nbStim,
                self.varianceStim,
                nb_Stim_Signals,
                self.nbEch,
                self.dt,
                int(self.stimtype_CB.currentText() == 'Burst'),
                Stim_Signals_out,
            )

        self.Stim_Signals = Stim_Signals_out

        if not self.seed == 0:
            np.random.seed(self.seed)

    def update_Model(self):
        """ "Update Model": (re)create the network cells from the current cell numbers and synaptic lists, then reset their states."""
        try:
            self.List_Neurone_type
        except:
            msg_cri("You must place the cell first.")
            return

        self.Compute_synaptic_connections_sparse()
        self.CC.create_cells(
            self.Nb_of_PYR,
            self.Nb_of_SOM,
            self.Nb_of_BAS,
            self.Nb_of_BIS,
            copy.deepcopy(self.List_PYR_Stimulated),
            copy.deepcopy(self.PreSynaptic_Cell_AMPA),
            copy.deepcopy(self.PreSynaptic_Cell_GABA),
            copy.deepcopy(self.PreSynaptic_Soma_Dend_AMPA),
            copy.deepcopy(self.PreSynaptic_Soma_Dend_AMPA_not),
            copy.deepcopy(self.PreSynaptic_Soma_Dend_GABA),
            copy.deepcopy(self.PreSynaptic_Soma_Dend_GABA_not),
            createCells=self.createCells,
        )
        self.createCells = False
        self.Reset_states_clicked()

    def modify_Model(self):
        """Open the window that edits every parameter of every cell ("All Model Param")."""
        if self.NewModify1NMM == None and self.NewModifyXNMM == None:
            self.NewModify1NMM = Modify_1_NMM(
                parent=self,
                Dict_Param=self.CC.List_Neurone_param,
                List_Names=self.List_Names,
                List_Color=self.List_Colors,
            )
            self.NewModify1NMM.Mod_OBJ.connect(self.ApplyMod1NMM)
            self.NewModify1NMM.Close_OBJ.connect(self.closeMod1NMM)
            self.NewModify1NMM.show()
        else:
            self.put_window_on_top()

    @pyqtSlot(list, list, list)
    def ApplyMod1NMM(self, Dict_Param, popName, popColor):
        """Slot: apply the parameters, names and colours coming from the "all parameters" editor."""
        for idx, p in enumerate(Dict_Param):
            for idx_v, key in enumerate(Dict_Param[idx].keys()):
                self.CC.List_Neurone_param[idx][key] = Dict_Param[idx][key]
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def closeMod1NMM(
        self,
    ):
        """Slot: delete the "all parameters" editor when it is closed."""
        self.NewModify1NMM.deleteLater()
        self.NewModify1NMM.destroyed.connect(self.on_destroyed_NewModify1NMM)

    @pyqtSlot('QObject*')
    def on_destroyed_NewModify1NMM(self, o):
        """Slot: forget the "all parameters" editor once it has been destroyed."""
        self.NewModify1NMM = None

    def ModXNMMclicked(
        self,
    ):
        """Open the window that edits the parameters of a group of cells at once ("Modify X Models")."""
        if self.NewModify1NMM == None and self.NewModifyXNMM == None:
            self.NewModifyXNMM = Modify_X_NMM(
                parent=self,
                List_Neurone_type=self.List_Neurone_type,
                Dict_Param=self.CC.List_Neurone_param,
                List_Names=self.List_Names,
                List_Color=self.List_Colors,
                initcell=0,
                CellPosition=self.CellPosition,
                scale=float(self.s_e.text()),
            )
            self.NewModifyXNMM.Mod_OBJ.connect(self.ApplyModXNMM)
            self.NewModifyXNMM.Close_OBJ.connect(self.close_ModXNMM)
            self.NewModifyXNMM.updateVTK_OBJ.connect(self.update_VTKgraph_from_ModXNMM)
            self.NewModifyXNMM.show()
        else:
            self.put_window_on_top()

    @pyqtSlot(list, list, list)
    def ApplyModXNMM(self, Dict_Param, popName, popColor):
        """Slot: apply the parameters, names and colours coming from the group editor."""
        for idx, p in enumerate(Dict_Param):
            for idx_v, key in enumerate(Dict_Param[idx].keys()):
                self.CC.List_Neurone_param[idx][key] = Dict_Param[idx][key]
        self.List_Names = popName
        self.List_Colors = popColor
        self.update_graph()

    @pyqtSlot()
    def close_ModXNMM(
        self,
    ):
        """Slot: delete the group editor when it is closed."""
        self.NewModifyXNMM.deleteLater()
        self.NewModifyXNMM.destroyed.connect(self.on_destroyed_NewModifyXNMM)
        # pass

    @pyqtSlot('QObject*')
    def on_destroyed_NewModifyXNMM(self, o):
        """Slot: forget the group editor once it has been destroyed."""
        self.NewModifyXNMM = None
        print(self.NewModifyXNMM)

    @pyqtSlot(
        list,
    )
    def update_VTKgraph_from_ModXNMM(self, selectedcell):
        """Slot: highlight in the 3D view the cells selected in the group editor."""
        self.Graph_viewer.selected_cells = selectedcell
        if self.displayVTK_CB.isChecked():
            self.Graph_viewer.draw_Graph()

    def update_ModXNMM_from_VTKgraph(self, seleced_cell):
        """Select in the group editor the cell clicked in the 3D view."""
        if not self.NewModifyXNMM == None:
            self.NewModifyXNMM.PopNumber.setCurrentIndex(seleced_cell)

    def put_window_on_top(self):
        """Bring the already-open parameter editor window to the front."""
        self.parent.processEvents()
        if not self.NewModify1NMM == None:
            self.NewModify1NMM.activateWindow()
        elif not self.NewModifyXNMM == None:
            self.NewModifyXNMM.activateWindow()

    def Reset_states_clicked(self):
        """Reset the state variables of all cells to their initial values (and restart simulated time at 0)."""
        try:
            self.CC.tps_start = 0.0
            self.CC.Reset_states()
        except:
            pass

    # ----------------------------------------------------------------------
    # Simulation and LFP
    # ----------------------------------------------------------------------
    def simulate(self):
        """Run button: simulate the network and display the membrane potentials.

        Generates the stimulation (the main stimulation plus the background input, see
        ``Generate_Background_Input``), runs ``CC.Simulate`` and stores the results:
            * ``Sigs_dict``  : time vector and membrane potential of every cell;
            * ``PPS``        : post-synaptic potentials of the CA1 PYR cells;
            * ``I_all``      : somatic/dendritic currents of the CA1 PYR cells
              (both used by the CSD LFP computation and by "Save Results").
        """
        if self.CC.ImReady == False:
            msg_cri('The model is not ready to Simulate')
            return
        if not self.Colonne_cortical_Thread.isRunning():
            t0 = time.time()
            self.msg = msg_wait("Computation in progress\nPlease wait.")
            self.msg.setStandardButtons(QMessageBox.StandardButton.Cancel)
            self.parent.processEvents()

            self.dt = 1 / float(self.Fs_e.text())
            self.T = float(self.SimDuration_e.text())
            self.nbEch = int(self.T / self.dt)

            self.Generate_Stim_Signals()
            # Add the background input (random pulses in each stimulated CA3 cell) to the stimulation signals
            self.Stim_Signals = self.Stim_Signals + Generate_Background_Input(
                len(self.List_PYR_Stimulated), self.nbEch, self.dt, self.seed, **BACKGROUND_INPUT
            )

            self.t0 = time.time()
            # Run the model. Outputs: time vector, PYR somatic/dendritic potentials, OLM, basket and BIS somatic potentials,
            # PYR excitatory/inhibitory post-synaptic potentials, and PYR somatic/dendritic currents.
            (
                self.t,
                self.pyrVs,
                self.pyrVd,
                self.olmVs,
                self.basketVs,
                self.BisVs,
                self.pyrPPSE,
                self.pyrPPSI,
                self.pyrPPSI_s,
                self.pyrI_S,
                self.pyrI_d,
            ) = self.CC.Simulate(self.T, self.dt, self.Stim_Signals)

            self.PPS = {}
            self.PPS['PPSE'] = self.pyrPPSE
            self.PPS['PPSI'] = self.pyrPPSI
            self.PPS['PPSI_s'] = self.pyrPPSI_s

            self.I_all = {}
            self.I_all['pyrI_S'] = self.pyrI_S
            self.I_all['pyrI_d'] = self.pyrI_d

            self.msg = msg_wait("Computation finished\nResults are currently displayed.\nPlease wait.")

            self.Color = []
            self.Sigs_dict = {}
            self.Sigs_dict["t"] = self.t + self.CC.tps_start - self.T
            count = 0
            for i in range(self.pyrVs.shape[0]):
                if i in self.List_PYR_Stimulated:
                    self.Color.append(self.List_Colors[count])
                    self.Sigs_dict["Pyr_" + str(i) + 'Stim'] = self.pyrVs[i, :]
                else:
                    self.Color.append(self.List_Colors[count])
                    self.Sigs_dict["Pyr_" + str(i)] = self.pyrVs[i, :]
                count += 1

            for i in range(self.basketVs.shape[0]):
                self.Color.append(self.List_Colors[count])
                self.Sigs_dict["BAS_" + str(i)] = self.basketVs[i, :]
                count += 1

            for i in range(self.olmVs.shape[0]):
                self.Color.append(self.List_Colors[count])
                self.Sigs_dict["SOM_" + str(i)] = self.olmVs[i, :]
                count += 1

            for i in range(self.BisVs.shape[0]):
                self.Color.append(self.List_Colors[count])
                self.Sigs_dict["BIS_" + str(i)] = self.BisVs[i, :]
                count += 1

            if self.displaycurves_CB.isChecked():
                print('update draw')
                self.mascene_EEGViewer.update(
                    Sigs_dict=self.Sigs_dict, Colors=self.Color, percentage=float(self.displaycurve_per_e.text())
                )
                print('finish update draw')

            print(str(datetime.timedelta(seconds=int(time.time() - t0))))
            self.msg.close()
            self.parent.processEvents()

    def displaycurves_CB_fonc(self):
        """Show or hide the membrane-potential curves."""
        if hasattr(self, 'Sigs_dict'):
            if self.displaycurves_CB.isChecked():
                self.mascene_EEGViewer.update(
                    Sigs_dict=self.Sigs_dict, Colors=self.Color, percentage=float(self.displaycurve_per_e.text())
                )

    def displayVTK_CB_fonc(self):
        """Show or hide the 3D view."""
        if self.displayVTK_CB.isChecked():
            try:
                self.Graph_viewer.draw_Graph()
                self.Graph_viewer.set_center()
                self.parent.processEvents()
            except:
                pass

    def Compute_LFP_fonc(self):
        """ "Comp" button: compute the LFP at the electrode and plot it.

        Two methods (selected in the panel):
            * "Dipole": each CA1 PYR cell is a dipole built from the soma-dendrite
              potential difference, weighted by distance to the electrode;
            * "CSD": current-source-density forward model (``RecordedPotential.LFP``)
              followed by the electrode model (``Electrode.ElectrodeModel``).
        An optional band-pass filter is then applied. The result is stored in ``self.LFP``.
        """
        if self.Compute_LFP2_type_CB.currentText() == "Dipole":
            # try:
            electrode_pos = np.array(self.electrode_pos)
            CellPosition = np.array([np.array(c) for c in self.CellPosition])

            Distance_from_electrode = distance.cdist([electrode_pos, electrode_pos], CellPosition, 'euclidean')[0, :]
            # vect direction??
            U = (CellPosition - electrode_pos) / Distance_from_electrode[:, None]

            Ssoma = np.pi * (self.somaSize / 2.0) * ((self.somaSize / 2.0) + self.somaSize * np.sqrt(5.0 / 4.0))
            Stotal = Ssoma / self.p

            # potentials
            self.LFP = np.zeros(self.pyrVs.shape[1])
            Vs_d = self.pyrVs - self.pyrVd

            for k in range(self.Nb_of_PYR - self.Nb_of_PYR_Stimulated):
                i = k + self.Nb_of_PYR_Stimulated
                Vdi = np.zeros((len(Vs_d[i, :]), 3))
                Vdi[:, 2] = Vs_d[i, :]
                Vm = np.sum(Vdi * U[i, :], axis=1)
                Vm = Vm / (4.0 * self.sigma * np.pi * Distance_from_electrode[i] * Distance_from_electrode[i])
                Vm = Vm * (self.dendriteSize + self.somaSize) / 2.0 * self.gc * Stotal
                self.LFP += Vm * 10.0e02

        elif self.Compute_LFP2_type_CB.currentText() == "CSD":

            electrode_pos = np.array(self.electrode_pos)
            print(self.electrode_pos)
            self.electrode_disk = [
                self.electrod_disk_CB.currentText(),
                float(self.electrode_radius_e.text()),
                float(self.electrode_angle1_e.text()),
                float(self.electrode_angle2_e.text()),
                float(self.electrode_h_e.text()),
            ]
            Fs = int(self.Fs_e.text())
            ComputeLFP = RecordedPotential.LFP(
                Fs=Fs,
                type=int(self.electrode_disk[0] == 'Cylinder'),
                re=self.electrode_disk[1],
                h=self.electrode_disk[4],
                tx=self.electrode_disk[2],
                ty=self.electrode_disk[3],
                pos=self.electrode_pos,
            )

            PyrCellPosition = self.CellPosition[np.array(self.List_Neurone_type) == 1, :]
            long = PyrCellPosition.shape[0]
            I_all = {'pyrI_S': self.I_all['pyrI_S'][-long:, :], 'pyrI_d': self.I_all['pyrI_d'][-long:, :]}
            PPS = {
                'PPSE': self.PPS['PPSE'][-long:, :],
                'PPSI': self.PPS['PPSI'][-long:, :],
                'PPSI_s': self.PPS['PPSI_s'][-long:, :],
            }
            LFP = ComputeLFP.compute_LFP_CSD(I_all, PPS, PyrCellPosition)

            if self.electrod_disk_CB.currentText() == 'Cylinder':
                cyl = 1
            else:
                cyl = 0

            e_g = Electrode.ElectrodeModel(re=self.electrode_disk[1], Etype=0, h=self.electrode_disk[4], cyl=cyl)
            LFP = -e_g.GetVelec(LFP, Fs=int(1000 / self.dt))

            self.LFP = LFP

        if self.lfpplot_filtre_Active.isChecked():
            LFP = signalfilterbandpass(
                self.LFP,
                1000 / self.dt,
                float(self.lfpplot_filtre_FCL.text()),
                float(self.lfpplot_filtre_FCH.text()),
                int(self.lfpplot_filtre_order.text()),
            )
            self.LFP = LFP[0, :]
        self.mascene_LFPViewer.update(shiftT=self.CC.tps_start - self.T)

    # ----------------------------------------------------------------------
    # Saving and loading
    # ----------------------------------------------------------------------
    def SaveSimul(self, Filepath=None):
        """Save the simulation set-up (sizes, percentages, connectivity, positions, stimulation, electrode, cell parameters) in a tab-separated text file.

        ``Filepath``: path to use; if not a string, a file dialog is shown.
        """
        extension = "txt"

        if not isinstance(Filepath, str):
            fileName = QFileDialog.getSaveFileName(
                caption='Save parameters', filter=extension + " (*." + extension + ")"
            )
            if fileName[0] == '':
                return
            if os.path.splitext(fileName[0])[1] == '':
                fileName = (fileName[0] + '.' + extension, fileName[1])
        else:
            fileName = [Filepath, extension + " (*." + extension + ")"]
        if fileName[1] == extension + " (*." + extension + ")":
            f = open(fileName[0], 'w')
            f.write("x" + "\t" + str(self.x) + "\n")
            f.write("y" + "\t" + str(self.y) + "\n")
            f.write("z" + "\t" + str(self.z) + "\n")

            f.write("Per_PYR" + "\t" + str(self.Per_PYR.text()) + "\n")
            f.write("Per_BAS" + "\t" + str(self.Per_BAS.text()) + "\n")
            f.write("Per_SOM" + "\t" + str(self.Per_SOM.text()) + "\n")
            f.write("Per_BIS" + "\t" + str(self.Per_BIS.text()) + "\n")
            f.write("Per_PYR_stimulated" + "\t" + str(self.Per_PYR_stimulated.text()) + "\n")

            f.write("cellnumber" + "\t" + self.nbcell_e.text() + "\n")

            f.write("List_C" + "\t" + str(self.List_C.tolist()) + "\n")
            f.write("List_Var" + "\t" + str(self.List_Var.tolist()) + "\n")

            f.write("cell_placement_CB" + "\t" + self.cell_placement_CB.currentText() + "\n")

            f.write("CellPosition" + "\t" + str(self.CellPosition.tolist()) + "\n")
            f.write("ConnectivityMatrix" + "\t" + str([CM.tolist() for CM in self.ConnectivityMatrix]) + "\n")

            f.write("StimDuration" + "\t" + self.StimDuration_e.text() + "\n")
            f.write("i_inj" + "\t" + self.i_inj_e.text() + "\n")
            f.write("tau" + "\t" + self.tau_e.text() + "\n")
            f.write("nbStim" + "\t" + self.nbStim_e.text() + "\n")
            f.write("varianceStim" + "\t" + self.varianceStim_e.text() + "\n")

            f.write("SimDuration" + "\t" + self.SimDuration_e.text() + "\n")
            f.write("Fs" + "\t" + self.Fs_e.text() + "\n")

            f.write("electrode_x" + "\t" + self.electrode_x_e.text() + "\n")
            f.write("electrode_y" + "\t" + self.electrode_y_e.text() + "\n")
            f.write("electrode_z" + "\t" + self.electrode_z_e.text() + "\n")

            f.write("List_Names" + "\t" + str(self.List_Names) + "\n")
            f.write("List_Colors" + "\t" + str(self.List_Colors) + "\n")

            for dict_param in self.CC.List_Neurone_param:
                f.write("dict_param" + "\t" + str(dict_param) + "\n")

            f.close()

    def LoadSimul(self, Filepath=None):
        """Load a set-up written by ``SaveSimul`` and rebuild the model from it.

        Lines are read in the fixed order used by ``SaveSimul``, so the file must not be reordered.
        ``Filepath``: path to use; if not a string, a file dialog is shown.
        """
        extension = "txt"

        if not isinstance(Filepath, str):
            fileName = QFileDialog.getOpenFileName(
                caption='Load parameters', filter=extension + " (*." + extension + ")"
            )
            if fileName[0] == '':
                return
            if os.path.splitext(fileName[0])[1] == '':
                fileName = (fileName[0] + '.' + extension, fileName[1])
        else:
            fileName = [Filepath, 'txt (*.txt)']
        if fileName[1] == extension + " (*." + extension + ")":
            f = open(fileName[0], 'r')
            line = f.readline()
            self.x_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            self.x = float(self.x_e.text())
            line = f.readline()
            self.y_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            self.y = float(self.y_e.text())
            line = f.readline()
            self.z_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            self.z = float(self.z_e.text())
            self.electrode_placement_func()
            line = f.readline()
            self.Per_PYR.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Per_BAS.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Per_SOM.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Per_BIS.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Per_PYR_stimulated.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            self.nbcell_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            self.update_cellNnumber()

            line = f.readline()
            sublist = (
                line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            )
            self.List_C = []
            for s in sublist:
                self.List_C.append([float(s2) for s2 in s.split(',')])
            self.List_C = np.array(self.List_C)
            count = 0
            for l in range(len(self.List_C)):
                for c in range(len(self.List_C)):
                    self.List_C_e[count].setText(str(self.List_C[l, c]))
                    count += 1

            line = f.readline()
            sublist = (
                line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            )
            self.List_Var = []
            for s in sublist:
                self.List_Var.append([float(s2) for s2 in s.split(',')])
            self.List_Var = np.array(self.List_Var)
            count = 0
            for l in range(len(self.List_Var)):
                for c in range(len(self.List_Var)):
                    self.List_Var_e[count].setText(str(self.List_Var[l, c]))
                    count += 1

            line = f.readline()
            index = self.cell_placement_CB.findText(
                line.split("\t")[-1].replace('\n', ''), Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive
            )
            self.cell_placement_CB.setCurrentIndex(index)

            line = f.readline()
            sublist = (
                line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            )
            self.CellPosition = []
            for s in sublist:
                self.CellPosition.append([float(s2) for s2 in s.split(',')])
            self.CellPosition = np.array(self.CellPosition)

            line = f.readline()
            sublist = (
                line.split("\t")[-1].replace('\n', '').replace(' ', '').replace('[[', '').replace(']]', '').split('],[')
            )
            self.ConnectivityMatrix = []
            for s in sublist:
                self.ConnectivityMatrix.append(np.array([int(s2) for s2 in s.split(',') if not s2 == '']))
            self.ConnectivityMatrix = np.array(self.ConnectivityMatrix, dtype='object')

            line = f.readline()
            self.StimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.i_inj_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.tau_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.nbStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.varianceStim_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            #
            line = f.readline()
            self.SimDuration_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.Fs_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            self.electrode_x_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.electrode_y_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))
            line = f.readline()
            self.electrode_z_e.setText(line.split("\t")[-1].replace('\n', '').replace(" ", ""))

            line = f.readline()
            self.List_Names = (
                line.split("\t")[-1]
                .replace('\n', '')
                .replace('[', '')
                .replace(']', '')
                .replace("'", '')
                .replace(" ", '')
                .split(',')
            )
            line = f.readline()
            self.List_Colors = (
                line.split("\t")[-1]
                .replace('\n', '')
                .replace('[', '')
                .replace(']', '')
                .replace("'", '')
                .replace(" ", '')
                .split(',')
            )

            # Next: the neuron-model parameters (one dict per cell) until the end of the file
            self.take_cell_number()
            self.Compute_synaptic_connections_sparse()
            self.masceneCM.update()
            self.update_graph()

            List_Neurone_param = []
            for line in f:
                finalstring = (
                    line.split("\t")[-1]
                    .replace('\n', '')
                    .replace("{", "")
                    .replace("}", "")
                    .replace("'", "")
                    .replace(" ", "")
                    .replace(" ", "")
                )

                # Splitting the string based on , we get key value pairs
                listdict = finalstring.split(",")

                dictionary = {}
                for i in listdict:
                    # Get Key Value pairs separately to store in dictionary
                    keyvalue = i.split(":")

                    # Replacing the single quotes in the leading.
                    m = keyvalue[0].strip('\'')
                    m = m.replace("\"", "")
                    dictionary[m] = float(keyvalue[1].strip('"\''))
                List_Neurone_param.append(dictionary)
            self.createCells = True
            self.update_Model()
            self.CC.List_Neurone_param = List_Neurone_param
        return True

    def SaveRes(self):
        """Save the simulation results (signals, LFP, coordinates, PSPs, currents).

        Formats: .mat, .data (pickle), .csv, .dat/.bin (with a .des description file) and
        .edf. Coordinates, PSPs and currents can only be saved in .mat and .data.
        """
        if hasattr(self, 'Sigs_dict'):
            exPopup = QuestionWhatToSave(self)
            if exPopup.exec() == QDialog.DialogCode.Accepted:
                saveLFP = exPopup.save_lfp_CB.isChecked()
                savesignals = exPopup.save_signal_CB.isChecked()
                savecoords = exPopup.save_coord_CB.isChecked()
                savePPSs = exPopup.save_PPS_CB.isChecked()
                saveIs = exPopup.save_I_CB.isChecked()
                if not saveLFP and not savesignals and not savecoords and not savePPSs and not saveIs:
                    return
            else:
                return
            exPopup.deleteLater()
            if not savesignals:
                Sigs_dict = {}
            else:
                Sigs_dict = copy.deepcopy(self.Sigs_dict)

            if not 't' in Sigs_dict.keys():
                Sigs_dict["t"] = self.t

            if hasattr(self, 'LFP') and saveLFP:
                if len(self.LFP) == len(Sigs_dict["t"]):
                    Sigs_dict['LFP'] = self.LFP

            if hasattr(self, 'CellPosition') and savecoords:
                Sigs_dict['Coordinates'] = self.CellPosition

            if hasattr(self, 'PPS') and savePPSs:
                Sigs_dict['PPS'] = self.PPS

            if hasattr(self, 'I_all') and saveIs:
                Sigs_dict['I_all'] = self.I_all

            if savecoords:
                fileName = QFileDialog.getSaveFileName(caption='Save parameters', filter=".mat (*.mat);;.data (*.data)")
            else:
                fileName = QFileDialog.getSaveFileName(
                    caption='Save parameters',
                    filter=".mat (*.mat);;.data (*.data);;.csv (*.csv);;.dat (*.dat);;.bin (*.bin);;.edf (*.edf)",
                )

            if fileName[0] == '':
                return
            tp = Sigs_dict['t']
            if fileName[1] == '.data (*.data)':
                file_pi = open(fileName[0], 'wb')
                pickle.dump(Sigs_dict, file_pi, -1)
                file_pi.close()
            elif fileName[1] == '.bin (*.bin)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1.0 / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                file.close()
                keys = list(Sigs_dict.keys())
                array = np.array(Sigs_dict[keys[0]])
                for s in keys[1:]:
                    array = np.vstack((array, Sigs_dict[s] * 1000))
                array = array.T.flatten()
                array.astype('float32')
                s = struct.pack('f' * len(array), *array)
                file = open(os.path.join(path, name + '.bin'), "wb")
                file.write(s)
                file.close()

            elif fileName[1] == '.mat (*.mat)':
                scipy.io.savemat(fileName[0], mdict=Sigs_dict)
            elif fileName[1] == '.csv (*.csv)':
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter='\t', lineterminator='\n')
                w.writerow(Sigs_dict.keys())
                for values in Sigs_dict.values():
                    w.writerow(['{:e}'.format(var) for var in values])
                f.close()
            elif fileName[1] == '.dat (*.dat)':
                # write .des file
                path, name = os.path.split(fileName[0])
                name = name.split('.')[0]
                file = open(os.path.join(path, name + '.des'), "w")
                file.write("[patient]  x" + '\n')
                file.write("[date] " + datetime.datetime.today().strftime('%m/%d/%Y') + '\n')
                file.write("[time] " + datetime.datetime.today().strftime('%H:%M:%S') + '\n')
                Fs = int(1.0 / (tp[1] - tp[0]))
                file.write("[samplingfreq] " + str(Fs) + '\n')
                file.write("[nbsegments] 1" + '\n')
                file.write("[enabled] 1" + '\n')
                nsample = len(tp)
                file.write("[nbsamples] " + str(nsample) + '\n')
                file.write("[segmentsize] " + str(nsample) + '\n')
                file.write("[segmentInitialTimes] 0.0" + '\n')
                file.write("[nbchannels] " + str(len(Sigs_dict)) + '\n')
                file.write("[channelnames] :" + '\n')
                for s in Sigs_dict.keys():
                    file.write(s + " ------" + '\n')
                #
                file.close()
                f = open(fileName[0], 'w')
                w = csv.writer(f, delimiter=' ', lineterminator='\n')
                for idx in range(len(tp)):
                    line = []
                    for i_v, values in enumerate(Sigs_dict.values()):
                        if i_v == 0:
                            line.append(values[idx])
                        else:
                            line.append(values[idx] * 1000)
                    w.writerow(['{:f}'.format(var) for var in line])
                f.close()
            elif fileName[1] == '.edf (*.edf)':
                Fs = int(1.0 / (tp[1] - tp[0]))
                Sigs_dict.pop("t")
                N = len(Sigs_dict.keys())
                f = pyedflib.EdfWriter(fileName[0], N, file_type=1)

                lims = 0
                for key, value in Sigs_dict.items():
                    if key not in 't':
                        lims = max(lims, np.max(np.abs(value)))

                lims *= 2
                for i, key in enumerate(Sigs_dict.keys()):
                    f.setLabel(i, key)
                    f.setSamplefrequency(i, Fs)
                    f.setPhysicalMaximum(i, lims)
                    f.setPhysicalMinimum(i, -lims)
                f.update_header()

                kiter = len(tp) // Fs
                for k in range(kiter):
                    for i, key in enumerate(Sigs_dict.keys()):
                        f.writePhysicalSamples(Sigs_dict[key][k * Fs : (k + 1) * Fs].flatten())

                f.update_header()
                f.close()

        return

    def LoadModel(self):
        """Load another network model class from a python file (must define ``Colonne_cortical``)."""
        fileName = QFileDialog.getOpenFileName(self, "Open Model", "", "Cortical column (*.py)")
        if fileName[0] == '':
            return
        if fileName[1] == "Cortical column (*.py)":
            if os.path.splitext(fileName[0])[1] == '':
                fileName[0] = (fileName[0] + '.py', fileName[1])
            fileName = str(fileName[0])
            filepath, filename = os.path.split(fileName)
            sys.path.append(filepath)
            shortname, extension = os.path.splitext(filename)
            self.Colonne_class = __import__(shortname)
            self.Colonne = getattr(self.Colonne_class, 'Colonne_cortical')
            self.CC = self.Colonne()
            self.CC.updateTime.something_happened.connect(self.updateTime)
            self.createCells = True

    def LoadRes(self):
        """Placeholder (loading results is not implemented)."""
        pass

    def setmarginandspacing(self, layout):
        """Apply the standard 5 px margins and spacing to ``layout``."""
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)


class QuestionWhatToSave(QDialog):
    """Dialog asking which results to save (LFP, signals, coordinates, PSPs, currents)."""

    def __init__(self, parent=None, item=None, Graph_Items=None):
        """Build the check boxes and the Ok / Cancel buttons."""
        super(QuestionWhatToSave, self).__init__(parent)
        self.parent = parent
        self.Param_box = QGroupBox("Select information you wan to save:")
        self.Param_box.setFixedWidth(300)
        self.layout_Param_box = QVBoxLayout()
        self.item = item

        self.CB_layout = QVBoxLayout()
        self.save_lfp_CB = QCheckBox('LFP')
        self.save_signal_CB = QCheckBox('Signals')
        self.save_PPS_CB = QCheckBox('Post Synaptic Potentials')
        self.save_I_CB = QCheckBox('Synaptic Currents')
        self.save_coord_CB = QCheckBox('Coordinates')
        self.save_coord_l = QLabel('Coordinates, PSP, and Current can only be save in .mat and .data format.')
        self.CB_layout.addWidget(self.save_lfp_CB)
        self.CB_layout.addWidget(self.save_signal_CB)
        self.CB_layout.addWidget(self.save_coord_l)
        self.CB_layout.addWidget(self.save_coord_CB)
        self.CB_layout.addWidget(self.save_PPS_CB)
        self.CB_layout.addWidget(self.save_I_CB)
        [cb.setChecked(True) for cb in [self.save_lfp_CB, self.save_signal_CB, self.save_coord_CB]]

        self.horizontalGroupBox_Actions = QWidget()
        self.horizontalGroupBox_Actions.setFixedSize(285, 80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok = QPushButton('Ok')
        self.Button_Cancel = QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66, 30)
        self.Button_Cancel.setFixedSize(66, 30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.CB_layout)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        """Accept the dialog."""
        self.accept()


class CMViewer(QGraphicsView):
    """Matplotlib view of the connectivity matrix (sources on x, targets on y).

    Coloured squares delimit the cell groups (PYR, BAS, SOM, BIS).
    """

    def __init__(self, parent=None):
        """Create the matplotlib figure, canvas and toolbar."""
        super(CMViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.03, bottom=0.02, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self):
        """Redraw the connectivity matrix of the parent window."""
        self.ConnectivityMatrix = self.parent.ConnectivityMatrix
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        ax = self.figure.add_subplot(111)
        if scipy.sparse.issparse(self.ConnectivityMatrix):
            im = ax.spy(self.ConnectivityMatrix, markersize=1)
        elif len(self.ConnectivityMatrix.shape) == 1:

            raw = []
            col = []
            dat = []
            for l in range(len(self.ConnectivityMatrix)):
                for c in self.ConnectivityMatrix[l]:
                    raw.append(l)
                    col.append(c)
                    dat.append(1)
            s = sparse.coo_matrix(
                (dat, (raw, col)),
                shape=(self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3], self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3]),
            )
            im = ax.spy(s, markersize=1)
        else:
            im = ax.imshow(self.ConnectivityMatrix)
        ax.set_title('ConnectivityMatrix', fontdict={'fontsize': 10})

        plt.xlabel('Sources')
        plt.ylabel('Targets')

        rect = patches.Rectangle(
            (0, 0),
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1,
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[3] - 1,
            linewidth=1,
            edgecolor='y',
            facecolor='none',
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (0, 0),
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1,
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[2] - 1,
            linewidth=1,
            edgecolor='b',
            facecolor='none',
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (0, 0),
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1,
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[1] - 1,
            linewidth=1,
            edgecolor='g',
            facecolor='none',
        )
        ax.add_patch(rect)
        rect = patches.Rectangle(
            (0, 0),
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1,
            self.parent.Nb_of_PYR_BAS_SOM_BIS_sum[0] - 1,
            linewidth=1,
            edgecolor='r',
            facecolor='none',
        )
        ax.add_patch(rect)

        self.canvas.draw_idle()
        self.canvas.show()

    def closeEvent(self, event):
        """Close the matplotlib figure."""
        plt.close(self.figure)


class StimViewer(QGraphicsView):
    """Matplotlib view of the CA3 stimulation signals (one curve per stimulated cell, offset vertically)."""

    def __init__(self, parent=None):
        """Create the matplotlib figure, canvas and toolbar."""
        super(StimViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self):
        """Redraw the stimulation signals of the parent window."""
        self.Stim_Signals = self.parent.Stim_Signals
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt
        plt.figure()
        ax = self.figure.add_subplot(111)
        for i, s in enumerate(self.Stim_Signals):
            ax.plot(t, s + i * self.parent.i_inj)

        self.canvas.draw_idle()
        self.canvas.show()

    def closeEvent(self, event):
        """Close the matplotlib figure."""
        plt.close(self.figure)


class LFPViewer(QGraphicsView):
    """Matplotlib view of the LFP, also used to display the results of the synchrony analyses."""

    def __init__(self, parent=None):
        """Create the matplotlib figure, canvas and toolbar."""
        super(LFPViewer, self).__init__(parent)
        self.parent = parent
        self.setStyleSheet("border: 0px")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        plt.subplots_adjust(left=0.10, bottom=0.10, right=0.95, top=0.95, wspace=0.0, hspace=0.0)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self, shiftT=0.0):
        """Plot the LFP of the parent window; ``shiftT`` is the time offset (ms) of the first sample."""
        self.LFP = self.parent.LFP
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t, self.LFP)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_thresholding(self, sig, sigbin_perc, shiftT=0.0, titre=''):
        """Display the result of the 'Thresholding' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        plt.figure()
        ax = self.figure.add_subplot(111)
        ax.plot(t, sig)
        ax.plot(t, sigbin_perc)
        ax.set_ylabel(r'% pyramid firing')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_AutoCorrelation(self, sig, autocorr, shiftT=0.0, titre=''):
        """Display the result of the 'AutoCorrelation' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)

        t = np.arange(self.parent.nbEch) * self.parent.dt + shiftT
        plt.figure()
        ax = self.figure.add_subplot(211)
        ax.plot(t, sig)
        ax.set_ylabel(r'% pyramid firing ')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)
        ax2 = self.figure.add_subplot(212)
        ax2.plot(np.arange(len(autocorr)) * (t[1] - t[0]) - len(sig) * (t[1] - t[0]), autocorr)
        ax2.set_ylabel(r'autocorrelation ')
        ax2.set_xlabel(r'Time (s)')
        self.figure.set_tight_layout(True)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay_scatter(self, Aps_centers, shiftT=0.0, titre=''):
        """Display the 'Nearest delay' result as a scatter plot."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax = self.figure.add_subplot(111)
        for i in range(len(Aps_centers)):
            ax.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Time (s)')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay_boxplot(self, Aps_delays, shiftT=0.0, titre=''):
        """Display the 'Nearest delay' result as a box plot."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax = self.figure.add_subplot(111)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax.boxplot(Aps_delays_flatten)
        ax.set_ylabel(r'delays (ms)')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay(self, Aps_centers, Aps_delays, shiftT=0.0, titre=''):
        """Display the 'Nearest delay' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax1 = self.figure.add_subplot(121)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')
        ax1.set_title(titre)

        ax2 = self.figure.add_subplot(122)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_NearestDelay2(self, Aps_centers, Aps_delays, Aps_gauss_fit, shiftT=0.0, titre=''):
        """Display the 'Nearest delay' synchrony analysis (alternative layout)."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax1 = self.figure.add_subplot(131)
        for i in range(len(Aps_centers)):
            ax1.scatter(Aps_centers[i], i * np.ones(len(Aps_centers[i])), c='b')

        ax1.set_ylabel(r'Nb pyramid cell')
        ax1.set_xlabel(r'Time (s)')

        ax2 = self.figure.add_subplot(132)
        Aps_delays_flatten = []
        for i in range(len(Aps_delays)):
            if not Aps_delays[i] == []:
                Aps_delays_flatten.append(np.concatenate(Aps_delays[i]).ravel())
            else:
                Aps_delays_flatten.append([])
        ax2.boxplot(Aps_delays_flatten)
        ax2.set_ylabel(r'delays (ms)')
        ax2.set_xlabel(r'Nb pyramid cell')
        ax2.set_title(titre)

        ax3 = self.figure.add_subplot(133)
        ax3.scatter(range(len(Aps_gauss_fit)), Aps_gauss_fit, c='k')
        ax3.set_ylabel(r'gauss fit sigma')
        ax3.set_xlabel(r'Nb pyramid cell')
        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_ISI_distance(self, IntegralI, titre=''):
        """Display the 'ISI-distance' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(IntegralI, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_vanRossum_distance(self, vr_dist, titre='vanRossum_distance'):
        """Display the 'van Rossum distance' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vr_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def update_synchro_VictorPurpura_distance(self, vp_dist, titre='VictorPurpura_distance'):
        """Display the 'Victor Purpura distance' synchrony analysis."""
        self.figure.clear()
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.3, hspace=0.1)
        plt.figure()
        ax = self.figure.add_subplot(111)
        current_cmap = matplotlib.cm.get_cmap('viridis')
        current_cmap.set_bad(color='black')
        im = ax.imshow(vp_dist, cmap=current_cmap)
        self.figure.colorbar(im)
        ax.set_ylabel(r'Nb pyramid cell')
        ax.set_xlabel(r'Nb pyramid cell')
        ax.set_title(titre)

        self.canvas.draw_idle()
        self.canvas.show()

    def closeEvent(self, event):
        """Close the matplotlib figure."""
        plt.close(self.figure)


class Rescalesize(QDialog):
    """Dialog to rescale the tissue: scale factors (x, y, z) and the resulting sizes are kept in sync."""

    def __init__(self, parent=None, x=None, y=None, z=None):
        """Build the scale-factor and tissue-size fields; ``x``, ``y``, ``z`` are the current sizes."""
        super(Rescalesize, self).__init__(parent)
        self.parent = parent
        self.x = x
        self.y = y
        self.z = z
        self.Param_box = QWidget()
        self.Param_box.setFixedWidth(400)
        self.layout_Param_box = QVBoxLayout()

        self.Layout_param = QHBoxLayout()
        self.scale_GB = QGroupBox("rescale")

        # tissue size
        self.tissue_scale_GB = QGroupBox(r'scale value')
        labelX, self.xs_e = Layout_grid_Label_Edit(label=['x'], edit=['1'])
        labelY, self.ys_e = Layout_grid_Label_Edit(label=['y'], edit=['1'])
        labelZ, self.zs_e = Layout_grid_Label_Edit(label=['z'], edit=['1'])
        self.xs_e = self.xs_e[0]
        self.ys_e = self.ys_e[0]
        self.zs_e = self.zs_e[0]
        grid = QGridLayout()
        self.tissue_scale_GB.setLayout(grid)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.xs_e.textChanged.connect(lambda state, s='x': self.rescale_axe(s))
        self.ys_e.textChanged.connect(lambda state, s='y': self.rescale_axe(s))
        self.zs_e.textChanged.connect(lambda state, s='z': self.rescale_axe(s))
        self.Layout_param.addWidget(self.tissue_scale_GB)

        self.tissue_size_GB = QGroupBox(r'tissue size')
        labelX, self.x_e = Layout_grid_Label_Edit(label=['x'], edit=[x])
        labelY, self.y_e = Layout_grid_Label_Edit(label=['y'], edit=[y])
        labelZ, self.z_e = Layout_grid_Label_Edit(label=['z'], edit=[z])
        self.x_e = self.x_e[0]
        self.y_e = self.y_e[0]
        self.z_e = self.z_e[0]
        grid = QGridLayout()
        self.tissue_size_GB.setLayout(grid)
        grid.setSpacing(0)
        grid.addWidget(labelX, 0, 0)
        grid.addWidget(labelY, 1, 0)
        grid.addWidget(labelZ, 2, 0)
        self.x_e.textChanged.connect(lambda state, s='x': self.resize_axe(s))
        self.y_e.textChanged.connect(lambda state, s='y': self.resize_axe(s))
        self.z_e.textChanged.connect(lambda state, s='z': self.resize_axe(s))
        self.Layout_param.addWidget(self.tissue_size_GB)

        self.horizontalGroupBox_Actions = QGroupBox("Actions")
        self.horizontalGroupBox_Actions.setFixedSize(285, 80)
        self.layout_Actions = QHBoxLayout()
        self.Button_Ok = QPushButton('Ok')
        self.Button_Cancel = QPushButton('Cancel')
        self.Button_Ok.setFixedSize(66, 30)
        self.Button_Cancel.setFixedSize(66, 30)
        self.layout_Actions.addWidget(self.Button_Ok)
        self.layout_Actions.addWidget(self.Button_Cancel)
        self.horizontalGroupBox_Actions.setLayout(self.layout_Actions)

        self.layout_Param_box.addLayout(self.Layout_param)
        self.layout_Param_box.addWidget(self.horizontalGroupBox_Actions)
        self.Param_box.setLayout(self.layout_Param_box)
        self.setLayout(self.layout_Param_box)

        self.Button_Ok.clicked.connect(self.myaccept)
        self.Button_Cancel.clicked.connect(self.reject)

    def myaccept(self):
        """Accept the dialog."""
        self.accept()

    def resize_axe(self, s):
        """A size was edited: update the corresponding scale factor."""
        try:
            if s == 'x':
                self.xs_e.blockSignals(True)
                self.xs_e.setText(str(float(self.x_e.text()) / float(self.x)))
                self.xs_e.blockSignals(False)
            elif s == 'y':
                self.ys_e.blockSignals(True)
                self.ys_e.setText(str(float(self.y_e.text()) / float(self.y)))
                self.ys_e.blockSignals(False)
            elif s == 'z':
                self.zs_e.blockSignals(True)
                self.zs_e.setText(str(float(self.z_e.text()) / float(self.z)))
                self.zs_e.blockSignals(False)
        except:
            pass

    def rescale_axe(self, s):
        """A scale factor was edited: update the corresponding size."""
        try:
            if s == 'x':
                self.x_e.blockSignals(True)
                self.x_e.setText(str(float(self.xs_e.text()) * float(self.x)))
                self.x_e.blockSignals(False)
            elif s == 'y':
                self.y_e.blockSignals(True)
                self.y_e.setText(str(float(self.ys_e.text()) * float(self.y)))
                self.y_e.blockSignals(False)
            elif s == 'z':
                self.z_e.blockSignals(True)
                self.z_e.setText(str(float(self.zs_e.text()) * float(self.z)))
                self.z_e.blockSignals(False)
        except:
            pass


def msg_wait(s):
    """Show a non-modal "please wait" message box with an animated icon and return it (close it with ``.close()``).

    The animation is read from ``icons/wait.gif`` (path relative to the working directory).
    """
    msg = QMessageBox()
    msg.setIconPixmap(QPixmap(os.path.join('icons', 'wait.gif')).scaledToWidth(100))
    icon_label = msg.findChild(QLabel, "qt_msgboxex_icon_label")
    movie = QMovie(os.path.join('icons', 'wait.gif'))
    setattr(msg, 'icon_label', movie)
    icon_label.setMovie(movie)
    movie.start()

    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setModal(False)
    msg.show()
    return msg


def msg_cri(s):
    """Show a modal critical-error message box with the text ``s``."""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText(s)
    msg.setWindowTitle(" ")
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg.exec()


def questionsure(s):
    """Show an Ok/Cancel warning box with the text ``s``; return True if the user pressed Ok."""
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Warning)
    strname = s
    msg.setText(strname)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
    ok = msg.exec()
    if ok == QMessageBox.StandardButton.Ok:
        return True
    else:
        return False


def main():
    """Create the Qt application, install a logging exception hook (``logdata.log``) and show the main window."""
    app = QApplication(sys.argv)

    if platform.system() in ['Darwin', 'Linux']:
        app.setStyle('Windows')

    import logging

    logging.basicConfig(filename='logdata.log', filemode='w', level=logging.ERROR)

    def my_excepthook(type, value, tback):
        # log the exception here
        co = tback.tb_frame.f_code
        func_name = co.co_name
        line_no = tback.tb_frame.f_lineno
        filename = co.co_filename
        logging.error("main crashed. Error: %s\n%s\n%s", type, value, tback)
        logging.error(
            'Tracing exception: %s "%s" \non line %s of file %s function %s'
            % (type.__name__, value, line_no, filename, func_name)
        )
        import traceback

        string = traceback.format_stack(tback.tb_frame)
        for s in string:
            logging.error('%s\n' % (s))
        # then call the default handler
        sys.__excepthook__(type, value, tback)
        sys.exit(app.exec())

    sys.excepthook = my_excepthook

    ex = ModelMicro_GUI(app)
    ex.setWindowTitle('Model micro GUI')
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
