import numpy as np
from scipy.spatial import distance
from Modeles.Wendling import pyrCellCA1 as pyrCA1
from Modeles.Wendling import Inter_Basket as Basket
from Modeles.Wendling import Inter_OLM_SOM as OLM_SOM
from Modeles.Wendling import Inter_BIS as BIS
import matplotlib.pyplot as plt

from PyQt6.QtCore import *
from numba.experimental import jitclass
from numba import jit,njit, types, typeof
from numba import boolean, int32, float64,uint8
# Type = 1 = Pyramidal cell for CA1
# Type = 2 = Pyramidal cell for CA3
# Type = 3 = Interneuron Basket
# Type = 4 = Interneuron Somatostatine
# Type = 5 = Interneuron BIS

class SenderObject(QObject):
    something_happened = pyqtSignal( float)

@njit
def Model_compute2( PreSynaptic_Cell_AMPA):
    return PreSynaptic_Cell_AMPA


spec1 = [
    ('PreSynaptic_Cell_AMPA' , types.List(typeof(np.array([], dtype=np.int32)))) ,
    ('PreSynaptic_Soma_Dend_AMPA' ,types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_Dend_AMPA_not' ,types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Cell_GABA' ,types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_Dend_GABA' ,types.List(typeof(np.array([], dtype=np.int32)))),
    ('PreSynaptic_Soma_Dend_GABA_not' ,types.List(typeof(np.array([], dtype=np.int32))))]

@jitclass(spec1)
class presynaptic_class():
    def __init__(self,PreSynaptic_Cell_AMPA,
                    PreSynaptic_Soma_Dend_AMPA,
                    PreSynaptic_Soma_Dend_AMPA_not,
                    PreSynaptic_Cell_GABA,
                    PreSynaptic_Soma_Dend_GABA,
                    PreSynaptic_Soma_Dend_GABA_not):
        self.PreSynaptic_Cell_AMPA          = PreSynaptic_Cell_AMPA
        self.PreSynaptic_Soma_Dend_AMPA     = PreSynaptic_Soma_Dend_AMPA
        self.PreSynaptic_Soma_Dend_AMPA_not = PreSynaptic_Soma_Dend_AMPA_not
        self.PreSynaptic_Cell_GABA          = PreSynaptic_Cell_GABA
        self.PreSynaptic_Soma_Dend_GABA     = PreSynaptic_Soma_Dend_GABA
        self.PreSynaptic_Soma_Dend_GABA_not = PreSynaptic_Soma_Dend_GABA_not

@njit
def Model_compute(  nbEch,
                    dt,
                    tps_start,
                    Nb_of_PYR,
                    Nb_of_SOM,
                    Nb_of_BAS,
                    Nb_of_BIS,
                    NB_cells,
                    Nb_of_PYR_STIM,
                    Nb_of_PYR_BAS_SOM_BIS_sum,
                    List_Neurone_PYR,
                    List_Neurone_BAS,
                    List_Neurone_OLM,
                    List_Neurone_BIS,
                    Stim_Signals,
                    PS,
                    pyrVs,
                    pyrVd,
                    olmVs,
                    basketVs,
                    BisVs,
                    t,
                    pyrPPSE,
                    pyrPPSI,
                    pyrPPSI_s,
                    pyrI_S,
                    pyrI_d,
                ):

    for i in range(Nb_of_PYR):
        List_Neurone_PYR[i].dt = dt
    for i in range(Nb_of_BAS):
        List_Neurone_BAS[i].dt = dt
    for i in range(Nb_of_SOM):
        List_Neurone_OLM[i].dt = dt
    for i in range(Nb_of_BIS):
        List_Neurone_BIS[i].dt = dt


    for tt, tp in enumerate(t):
        tps_start += dt

        for i in range(Nb_of_PYR):
            List_Neurone_PYR[i].init_I_syn()
            List_Neurone_PYR[i].updateParameters()
        for i in range(Nb_of_BAS):
            List_Neurone_BAS[i].init_I_syn()
            List_Neurone_BAS[i].updateParameters()
        for i in range(Nb_of_SOM):
            List_Neurone_OLM[i].init_I_syn()
            List_Neurone_OLM[i].updateParameters()
        for i in range(Nb_of_BIS):
            List_Neurone_BIS[i].init_I_syn()
            List_Neurone_BIS[i].updateParameters()

        for i in range(Nb_of_PYR_STIM):
            List_Neurone_PYR[i].I_soma_stim = Stim_Signals[i,tt]

        for i in range(NB_cells):
            # print(i)
            if len(PS.PreSynaptic_Cell_AMPA[i])>0:
                Cell_AMPA = PS.PreSynaptic_Cell_AMPA[i]
                Vpre_AMPA = np.zeros(len(Cell_AMPA))
                for k, c in enumerate(Cell_AMPA):
                    if c >= 0 and c < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                        Vpre_AMPA[k] = List_Neurone_PYR[c].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and c < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                        Vpre_AMPA[k] = List_Neurone_BAS[c - Nb_of_PYR].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and c < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                        Vpre_AMPA[k] = List_Neurone_OLM[c - Nb_of_PYR - Nb_of_BAS].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and c < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                        Vpre_AMPA[k] = List_Neurone_BIS[c - Nb_of_PYR - Nb_of_BAS - Nb_of_SOM].VsOutput()
                if i >= 0 and i < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                    neurone =  i
                    I_AMPA = List_Neurone_PYR[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    I_NMDA = List_Neurone_PYR[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                    List_Neurone_PYR[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    List_Neurone_PYR[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    List_Neurone_PYR[neurone].add_I_synDend(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA_not[i])
                    List_Neurone_PYR[neurone].add_I_synDend(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA_not[i])

                    pyrPPSE[neurone, tt] = np.sum(I_AMPA + I_NMDA)

                    pyrI_S[neurone, tt]=- List_Neurone_PYR[neurone].I_synSoma - (List_Neurone_PYR[neurone].g_c / List_Neurone_PYR[neurone].p * ((List_Neurone_PYR[neurone].Vs_)- (List_Neurone_PYR[neurone].Vd_)))
                    pyrI_d[neurone, tt]=- List_Neurone_PYR[neurone].I_synDend - (List_Neurone_PYR[neurone].g_c / (1 - List_Neurone_PYR[neurone].p) * (-(List_Neurone_PYR[neurone].Vs_) + (List_Neurone_PYR[neurone].Vd_)))


                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and i < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    I_AMPA = List_Neurone_BAS[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    I_NMDA = List_Neurone_BAS[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)


                    List_Neurone_BAS[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    List_Neurone_BAS[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])

                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and i < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    I_AMPA = List_Neurone_OLM[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    I_NMDA = List_Neurone_OLM[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                    List_Neurone_OLM[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    List_Neurone_OLM[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])

                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and i < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    I_AMPA = List_Neurone_BIS[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    I_NMDA = List_Neurone_BIS[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                    List_Neurone_BIS[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                    List_Neurone_BIS[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])


            if len(PS.PreSynaptic_Cell_GABA[i]) > 0:
                Cell_GABA = PS.PreSynaptic_Cell_GABA[i]
                Vpre_GABA = np.zeros(len(Cell_GABA))
                for k, c in enumerate(Cell_GABA):
                    if c >= 0 and c < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                        Vpre_GABA[k] = List_Neurone_PYR[c].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and c < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                        Vpre_GABA[k] = List_Neurone_BAS[c - Nb_of_PYR].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and c < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                        Vpre_GABA[k] = List_Neurone_OLM[c - Nb_of_PYR - Nb_of_BAS].VsOutput()
                    elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and c < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                        Vpre_GABA[k] = List_Neurone_BIS[c - Nb_of_PYR - Nb_of_BAS - Nb_of_SOM].VsOutput()

                if i >= 0 and i < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                    neurone =  i
                    I_GABA = List_Neurone_PYR[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    List_Neurone_PYR[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])
                    List_Neurone_PYR[neurone].add_I_synDend(I_GABA, PS.PreSynaptic_Soma_Dend_GABA_not[i])

                    pyrPPSI[neurone, tt] =   np.sum(I_GABA* PS.PreSynaptic_Soma_Dend_GABA[i])
                    pyrPPSI_s[neurone, tt] = np.sum(I_GABA* PS.PreSynaptic_Soma_Dend_GABA_not[i])

                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and i < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[0]
                    I_GABA = List_Neurone_BAS[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    List_Neurone_BAS[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and i < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[1]
                    I_GABA = List_Neurone_OLM[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    List_Neurone_OLM[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and i < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                    neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[2]
                    I_GABA = List_Neurone_BIS[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    List_Neurone_BIS[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])



        for i in range(Nb_of_PYR):
            List_Neurone_PYR[i].rk4()
        for i in range(Nb_of_BAS):
            List_Neurone_BAS[i].rk4()
        for i in range(Nb_of_SOM):
            List_Neurone_OLM[i].rk4()
        for i in range(Nb_of_BIS):
            List_Neurone_BIS[i].rk4()

        for i in range(Nb_of_PYR):
            pyrVs[i, tt] = List_Neurone_PYR[i].y[0]
            pyrVd[i, tt] = List_Neurone_PYR[i].y[1]
        for i in range(Nb_of_BAS):
            basketVs[i, tt] = List_Neurone_BAS[i].y[0]
        for i in range(Nb_of_SOM):
            olmVs[i, tt] = List_Neurone_OLM[i].y[0]
        for i in range(Nb_of_BIS):
            BisVs[i, tt] = List_Neurone_BIS[i].y[0]


    return t, pyrVs, pyrVd, olmVs, basketVs, BisVs, pyrPPSE, pyrPPSI, pyrPPSI_s, pyrI_S,pyrI_d


class Colonne_cortical:
    updateTime = SenderObject()
    def __init__(self):

        # for Simulation
        self.dt = 1 / 25
        self.T = 100
        self.tps_start = 0
        self.nbEch = int(self.T / self.dt)

        # neurons
        self.Nb_of_PYR_BAS_SOM_BIS_sum = []

        self.List_Neurone_PYR = []
        self.List_Neurone_BAS = []
        self.List_Neurone_OLM = []
        self.List_Neurone_BIS = []

        #for connectivity matrix
        # self.presynaptic_instance = presynaptic_class([],[],[],[],[],[])
        self.PreSynaptic_Cell_AMPA = []
        self.PreSynaptic_Cell_GABA = []
        self.PreSynaptic_Soma_Dend_AMPA = []
        self.PreSynaptic_Soma_Dend_AMPA_not = []
        self.PreSynaptic_Soma_Dend_GABA = []
        self.PreSynaptic_Soma_Dend_GABA_not = []

        #For stimulation
        self.List_PYR_Stimulated = []
        self.Stim_Signals = []
        self.ImReady = False





    def create_cells(   self,Nb_of_PYR,
                        Nb_of_SOM,
                        Nb_of_BAS,
                        Nb_of_BIS,List_PYR_Stimulated,
                        PreSynaptic_Cell_AMPA,
                        PreSynaptic_Cell_GABA,
                        PreSynaptic_Soma_Dend_AMPA,
                        PreSynaptic_Soma_Dend_AMPA_not,
                        PreSynaptic_Soma_Dend_GABA,
                        PreSynaptic_Soma_Dend_GABA_not,
                        createCells = True):
        self.Nb_of_PYR = Nb_of_PYR
        self.Nb_of_SOM = Nb_of_SOM
        self.Nb_of_BAS = Nb_of_BAS
        self.Nb_of_BIS = Nb_of_BIS
        self.NB_cells = Nb_of_PYR + Nb_of_SOM + Nb_of_BAS + Nb_of_BIS
        self.Nb_of_PYR_BAS_SOM_BIS_sum = [self.Nb_of_PYR,
                                          self.Nb_of_PYR + self.Nb_of_BAS,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM,
                                          self.Nb_of_PYR + self.Nb_of_BAS + self.Nb_of_SOM + self.Nb_of_BIS]
        Nb_of_PYR_BAS_SOM_BIS_sum = np.array(self.Nb_of_PYR_BAS_SOM_BIS_sum, dtype=int)
        self.List_PYR_Stimulated = List_PYR_Stimulated
        self.Nb_of_PYR_STIM = len(List_PYR_Stimulated)
        self.presynaptic_instance = presynaptic_class([np.array(l, dtype=np.int32) for l in PreSynaptic_Cell_AMPA],
                                                      [np.array(l, dtype=np.int32) for l in PreSynaptic_Soma_Dend_AMPA],
                                                      [np.array(l, dtype=np.int32) for l in PreSynaptic_Soma_Dend_AMPA_not],
                                                      [np.array(l, dtype=np.int32) for l in PreSynaptic_Cell_GABA],
                                                      [np.array(l, dtype=np.int32) for l in PreSynaptic_Soma_Dend_GABA],
                                                      [np.array(l, dtype=np.int32) for l in PreSynaptic_Soma_Dend_GABA_not])
        # self.presynaptic_instance = presynaptic_class([np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Cell_AMPA],
        #                                               [np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Soma_Dend_AMPA],
        #                                               [np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Soma_Dend_AMPA_not],
        #                                               [np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Cell_GABA],
        #                                               [np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Soma_Dend_GABA],
        #                                               [np.array(l, dtype=np.int32) if not len(l)==0 else np.array(-1, dtype=np.int32) for l in PreSynaptic_Soma_Dend_GABA_not])



        self.PreSynaptic_Cell_AMPA = PreSynaptic_Cell_AMPA
        self.PreSynaptic_Cell_GABA = PreSynaptic_Cell_GABA
        self.PreSynaptic_Soma_Dend_AMPA = PreSynaptic_Soma_Dend_AMPA
        self.PreSynaptic_Soma_Dend_AMPA_not = PreSynaptic_Soma_Dend_AMPA_not
        self.PreSynaptic_Soma_Dend_GABA = PreSynaptic_Soma_Dend_GABA
        self.PreSynaptic_Soma_Dend_GABA_not = PreSynaptic_Soma_Dend_GABA_not

        if createCells == True:
            self.List_Neurone_PYR = []
            self.List_Neurone_BAS = []
            self.List_Neurone_OLM = []
            self.List_Neurone_BIS = []
            self.List_Neurone_param = []
            if self.Nb_of_PYR == 0:
                self.List_Neurone_PYR = [pyrCA1.pyrCellCA1()]
            for i in range(self.Nb_of_PYR):
                self.List_Neurone_PYR.append(pyrCA1.pyrCellCA1())
                self.List_Neurone_param.append({s:getattr(self.List_Neurone_PYR[-1],s) for s in self.get_PYRCA1_Variables()})
                if i<self.Nb_of_PYR_STIM:
                    self.List_Neurone_PYR[-1].Type = 2
                # self.List_Neurone_PYR[-1].init_vector()
                # self.List_Neurone_PYR[-1].init_I_syn()
                # self.List_Neurone_PYR[-1].setParameters()
                # self.List_Neurone_PYR[-1].updateParameters()
                # self.List_Neurone_PYR[-1].I_AMPA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_PYR[-1].I_GABA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_PYR[-1].I_NMDA2(np.array([0.]), np.array([0.]),0.)
                # self.List_Neurone_PYR[-1].rk4()

            if self.Nb_of_BAS == 0:
                self.List_Neurone_BAS = [Basket.Inter_basket()]
            for i in range(self.Nb_of_BAS):
                self.List_Neurone_BAS.append(Basket.Inter_basket())
                self.List_Neurone_param.append({s:getattr(self.List_Neurone_BAS[-1],s) for s in self.get_BAS_Variables()})
                # self.List_Neurone_BAS[-1].init_vector()
                # self.List_Neurone_BAS[-1].init_I_syn()
                # self.List_Neurone_BAS[-1].setParameters()
                # self.List_Neurone_BAS[-1].updateParameters()
                # self.List_Neurone_BAS[-1].I_AMPA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_BAS[-1].I_GABA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_BAS[-1].I_NMDA2(np.array([0.]), np.array([0.]),0.)
                # self.List_Neurone_BAS[-1].rk4()

            if self.Nb_of_SOM == 0:
                self.List_Neurone_OLM = [OLM_SOM.Inter_OLM_SOM()]
            for i in range(self.Nb_of_SOM):
                self.List_Neurone_OLM.append(OLM_SOM.Inter_OLM_SOM())
                self.List_Neurone_param.append({s:getattr(self.List_Neurone_OLM[-1],s) for s in self.get_SOM_Variables()})
                # self.List_Neurone_OLM[-1].init_vector()
                # self.List_Neurone_OLM[-1].init_I_syn()
                # self.List_Neurone_OLM[-1].setParameters()
                # self.List_Neurone_OLM[-1].updateParameters()
                # self.List_Neurone_OLM[-1].I_AMPA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_OLM[-1].I_GABA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_OLM[-1].I_NMDA2(np.array([0.]), np.array([0.]),0.)
                # self.List_Neurone_OLM[-1].rk4()


            if self.Nb_of_BIS == 0:
                self.List_Neurone_BIS = [BIS.Inter_BIS()]
            for i in range(self.Nb_of_BIS):
                self.List_Neurone_BIS.append(BIS.Inter_BIS())
                self.List_Neurone_param.append({s:getattr(self.List_Neurone_BIS[-1],s) for s in self.get_BIS_Variables()})
                # self.List_Neurone_BIS[-1].init_vector()
                # self.List_Neurone_BIS[-1].init_I_syn()
                # self.List_Neurone_BIS[-1].setParameters()
                # self.List_Neurone_BIS[-1].updateParameters()
                # self.List_Neurone_BIS[-1].I_AMPA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_BIS[-1].I_GABA2(np.array([0.]), np.array([0.]))
                # self.List_Neurone_BIS[-1].I_NMDA2(np.array([0.]), np.array([0.]),0.)
                # self.List_Neurone_BIS[-1].rk4()



        count =0
        for i in range(self.Nb_of_PYR):
            self.List_Neurone_PYR[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count])
            self.List_Neurone_PYR[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
            self.List_Neurone_PYR[i].init_vector()
            count+= 1
        for i in range(self.Nb_of_BAS):
            self.List_Neurone_BAS[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count])
            self.List_Neurone_BAS[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
            self.List_Neurone_BAS[i].init_vector()
            count+= 1
        for i in range(self.Nb_of_SOM):
            self.List_Neurone_OLM[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count])
            self.List_Neurone_OLM[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
            self.List_Neurone_OLM[i].init_vector()
            count+= 1
        for i in range(self.Nb_of_BIS):
            self.List_Neurone_BIS[i].NbODEs_s_AMPA = len(self.PreSynaptic_Cell_AMPA[count])
            self.List_Neurone_BIS[i].NbODEs_s_GABA = len(self.PreSynaptic_Cell_GABA[count])
            self.List_Neurone_BIS[i].init_vector()
            count+= 1

        self.ImReady = True


    def ApplyParamDict(self):
        count = 0
        for i in range(self.Nb_of_PYR):
            for key, val in self.List_Neurone_param[count].items():
                setattr(self.List_Neurone_PYR[i] ,key,val)
            print(self.List_Neurone_PYR[i].E_GABA)
            count += 1
        for i in range(self.Nb_of_BAS):
            for key, val in self.List_Neurone_param[count].items():
                setattr(self.List_Neurone_BAS[i] ,key,val)
            count += 1
        for i in range(self.Nb_of_SOM):
            for key, val in self.List_Neurone_param[count].items():
                setattr(self.List_Neurone_OLM[i] ,key,val)
            count += 1
        for i in range(self.Nb_of_BIS):
            for key, val in self.List_Neurone_param[count].items():
                setattr(self.List_Neurone_BIS[i] ,key,val)
            count += 1



    def Reset_states(self):
        count = 0
        for i in range(self.Nb_of_PYR):
            self.List_Neurone_PYR[i].init_I_syn()
            self.List_Neurone_PYR[i].init_vector()
            self.List_Neurone_PYR[i].setParameters()
            count += 1
        for i in range(self.Nb_of_BAS):
            self.List_Neurone_BAS[i].init_I_syn()
            self.List_Neurone_BAS[i].init_vector()
            self.List_Neurone_BAS[i].setParameters()
            count += 1
        for i in range(self.Nb_of_SOM):
            self.List_Neurone_OLM[i].init_I_syn()
            self.List_Neurone_OLM[i].init_vector()
            self.List_Neurone_OLM[i].setParameters()
            count += 1
        for i in range(self.Nb_of_BIS):
            self.List_Neurone_BIS[i].init_I_syn()
            self.List_Neurone_BIS[i].init_vector()
            self.List_Neurone_BIS[i].setParameters()
            count += 1


    def Simulate_old(self, T, dt, Stim_Signals, ):

        self.ApplyParamDict()
        self.dt = dt
        self.nbEch = int(T/self.dt)

        t = np.arange(self.nbEch) * self.dt
        self.pyrVs = np.zeros((self.Nb_of_PYR,len(t)))
        self.pyrVd = np.zeros((self.Nb_of_PYR,len(t)))
        self.olmVs = np.zeros((self.Nb_of_SOM,len(t)))
        self.basketVs = np.zeros((self.Nb_of_BAS,len(t)))
        self.BisVs = np.zeros((self.Nb_of_BIS,len(t)))



        for i, c in enumerate(self.List_Neurone):
            c.dt = self.dt
            c.init_I_syn()
            # c.setParameters()
            print(c.E_GABA)

        for tt, tp in enumerate(t):
            self.tps_start += self.dt
            #apply stim
            if tp > 0 and (np.mod(tp, 1) == 0):
                self.updateTime.something_happened.emit(tp / T)

            for i, c in enumerate(self.List_Neurone):
                c.init_I_syn()
                c.updateParameters()

            for i, c in enumerate(self.List_PYR_Stimulated):
                self.List_Neurone[c].I_soma_stim = Stim_Signals[i,tt]

            for i, c in enumerate(self.List_Neurone):
                if len(self.PreSynaptic_Cell_AMPA[i])>0:
                    Vpre_AMPA = np.array([  neurone.VsOutput() for neurone in [ self.List_Neurone[n] for n in self.PreSynaptic_Cell_AMPA[i]]])
                    # c.computeI_AMPA(Vpre_AMPA)
                    # Vd = np.array([c.VSoma() if s_d==1 else c.VDend() for s_d in  self.PreSynaptic_Soma_Dend_AMPA[i]])
                    # I_AMPA = c.I_AMPA(Vd)
                    I_AMPA = c.I_AMPA2(Vpre_AMPA,self.PreSynaptic_Soma_Dend_AMPA[i])
                    # I_NMDA = np.array([c.computeI_NMDA(Vpre_AMPA[k],tp,Vd[k]) for k in range(len(Vpre_AMPA))])
                    I_NMDA = np.array(c.I_NMDA2(Vpre_AMPA,self.PreSynaptic_Soma_Dend_AMPA[i], self.tps_start))
                    # if len(I_NMDA)>=1:
                    #     if not I_NMDA[0] == 0.:
                    #         print(I_NMDA[0])

                    if c.Type in [1,2]:
                        c.add_I_synSoma(I_AMPA, self.PreSynaptic_Soma_Dend_AMPA[i])
                        c.add_I_synSoma(I_NMDA, self.PreSynaptic_Soma_Dend_AMPA[i])
                        c.add_I_synDend(I_AMPA, self.PreSynaptic_Soma_Dend_AMPA_not[i])
                        c.add_I_synDend(I_NMDA, self.PreSynaptic_Soma_Dend_AMPA_not[i])
                    else:
                        c.add_I_synSoma(I_AMPA, self.PreSynaptic_Soma_Dend_AMPA[i])
                        c.add_I_synSoma(I_NMDA, self.PreSynaptic_Soma_Dend_AMPA[i])

                if len(self.PreSynaptic_Cell_GABA[i]) > 0:
                    Vpre_GABA = np.array([neurone.VsOutput() for neurone in [self.List_Neurone[n] for n in self.PreSynaptic_Cell_GABA[i]]])
                    # c.computeI_GABA(Vpre_GABA)
                    # Vd = np.array([c.VSoma() if s_d == 1 else c.VDend() for s_d in self.PreSynaptic_Soma_Dend_GABA[i]])
                    # I_GABA = c.I_GABA(Vd)
                    I_GABA = c.I_GABA2(Vpre_GABA,self.PreSynaptic_Soma_Dend_GABA[i])
                    if c.Type in [1, 2]:
                        c.add_I_synSoma(I_GABA, self.PreSynaptic_Soma_Dend_GABA[i])
                        c.add_I_synDend(I_GABA, self.PreSynaptic_Soma_Dend_GABA_not[i])
                    else:
                        c.add_I_synSoma(I_GABA , self.PreSynaptic_Soma_Dend_GABA[i])

            #Comme code ObjectiveC
                # c.rk4()
            # Update ODE après tous en même
            for i, c in enumerate(self.List_Neurone):
                c.rk4()



            count_pyrVs = 0
            count_pyrVd = 0
            count_olmVs = 0
            count_basketVs = 0
            count_BisVs = 0
            for i, c in enumerate(self.List_Neurone):
                if c.Type in [1,2]:
                    self.pyrVs[count_pyrVs,tt] = c.y[0]
                    self.pyrVd[count_pyrVd,tt] = c.y[1]
                    count_pyrVs += 1
                    count_pyrVd += 1
                elif c.Type ==3  :
                    self.basketVs[count_basketVs,tt] = c.y[0]
                    count_basketVs += 1
                elif c.Type == 4:
                    self.olmVs[count_olmVs,tt] = c.y[0]
                    count_olmVs += 1
                elif c.Type == 5:
                    self.BisVs[count_BisVs,tt]= c.y[0]
                    count_BisVs += 1
        return(t, self.pyrVs, self.pyrVd, self.olmVs, self.basketVs, self.BisVs)

    def Simulate(self, T, dt, Stim_Signals, ):

        self.ApplyParamDict()
        self.dt = dt
        self.nbEch = int(T / self.dt)

        t = np.arange(self.nbEch) * self.dt
        self.pyrVs = np.zeros((self.Nb_of_PYR, len(t)))
        self.pyrVd = np.zeros((self.Nb_of_PYR, len(t)))
        self.olmVs = np.zeros((self.Nb_of_SOM, len(t)))
        self.basketVs = np.zeros((self.Nb_of_BAS, len(t)))
        self.BisVs = np.zeros((self.Nb_of_BIS, len(t)))

        self.pyrPPSE = np.zeros((self.Nb_of_PYR, len(t)))
        self.pyrPPSI = np.zeros((self.Nb_of_PYR, len(t)))
        self.pyrPPSI_s = np.zeros((self.Nb_of_PYR, len(t)))
        self.pyrI_S = np.zeros((self.Nb_of_PYR, len(t)))
        self.pyrI_d = np.zeros((self.Nb_of_PYR, len(t)))

        t, self.pyrVs, self.pyrVd, self.olmVs, self.basketVs, self.BisVs,\
        self.pyrPPSE, self.pyrPPSI, self.pyrPPSI_s, self.pyrI_S, self.pyrI_d = Model_compute(int(self.nbEch),
                                                                float(self.dt),
                                                                float(self.tps_start),
                                                                int(self.Nb_of_PYR),
                                                                int(self.Nb_of_SOM),
                                                                int(self.Nb_of_BAS),
                                                                int(self.Nb_of_BIS),
                                                                int(self.NB_cells),
                                                                int(self.Nb_of_PYR_STIM),
                                                                self.Nb_of_PYR_BAS_SOM_BIS_sum,
                                                                self.List_Neurone_PYR,
                                                                self.List_Neurone_BAS,
                                                                self.List_Neurone_OLM,
                                                                self.List_Neurone_BIS,
                                                                Stim_Signals,
                                                                self.presynaptic_instance,
                                                                self.pyrVs,
                                                                self.pyrVd,
                                                                self.olmVs,
                                                                self.basketVs,
                                                                self.BisVs,
                                                                t,
                                                                self.pyrPPSE,
                                                                self.pyrPPSI,
                                                                self.pyrPPSI_s,
                                                                self.pyrI_S,
                                                                self.pyrI_d,
                                                                )
        self.tps_start += (t[-1]+dt)
        return (t, self.pyrVs, self.pyrVd, self.olmVs, self.basketVs, self.BisVs,self.pyrPPSE, self.pyrPPSI, self.pyrPPSI_s, self.pyrI_S, self.pyrI_d)

    # @staticmethod
    @njit
    def Model_compute(self,
                        nbEch,
                        dt,
                        tps_start,
                        Nb_of_PYR,
                        Nb_of_SOM,
                        Nb_of_BAS,
                        Nb_of_BIS,
                        NB_cells,
                        Nb_of_PYR_STIM,
                        Nb_of_PYR_BAS_SOM_BIS_sum,
                        List_Neurone_PYR,
                        List_Neurone_BAS,
                        List_Neurone_OLM,
                        List_Neurone_BIS,
                        Stim_Signals,
                        PS,
                        pyrVs,
                        pyrVd,
                        olmVs,
                        basketVs,
                        BisVs,
                        t):

        for i in range(Nb_of_PYR):
            List_Neurone_PYR[i].dt = dt
        for i in range(Nb_of_BAS):
            List_Neurone_BAS[i].dt = dt
        for i in range(Nb_of_SOM):
            List_Neurone_OLM[i].dt = dt
        for i in range(Nb_of_BIS):
            List_Neurone_BIS[i].dt = dt


        for tt, tp in enumerate(t):
            tps_start += dt
            if tp > 0 and (np.mod(tp, 1) == 0):
                self.updateTime.something_happened.emit(tt / len(t))
            for i in range(Nb_of_PYR):
                List_Neurone_PYR[i].init_I_syn()
                List_Neurone_PYR[i].updateParameters()
            for i in range(Nb_of_BAS):
                List_Neurone_BAS[i].init_I_syn()
                List_Neurone_BAS[i].updateParameters()
            for i in range(Nb_of_SOM):
                List_Neurone_OLM[i].init_I_syn()
                List_Neurone_OLM[i].updateParameters()
            for i in range(Nb_of_BIS):
                List_Neurone_BIS[i].init_I_syn()
                List_Neurone_BIS[i].updateParameters()

            for i in range(Nb_of_PYR_STIM):
                List_Neurone_PYR[i].I_soma_stim = Stim_Signals[i,tt]

            for i in range(NB_cells):
                # print(i)
                if len(PS.PreSynaptic_Cell_AMPA[i])>0:
                    Cell_AMPA = PS.PreSynaptic_Cell_AMPA[i]
                    Vpre_AMPA = np.zeros(len(Cell_AMPA))
                    for k, c in enumerate(Cell_AMPA):
                        if c >= 0 and c < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                            Vpre_AMPA[k] = List_Neurone_PYR[c].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and c < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                            Vpre_AMPA[k] = List_Neurone_BAS[c - Nb_of_PYR].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and c < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                            Vpre_AMPA[k] = List_Neurone_OLM[c - Nb_of_PYR - Nb_of_BAS].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and c < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                            Vpre_AMPA[k] = List_Neurone_BIS[c - Nb_of_PYR - Nb_of_BAS - Nb_of_SOM].VsOutput()
                    if i >= 0 and i < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                        neurone =  i
                        I_AMPA = List_Neurone_PYR[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        I_NMDA = List_Neurone_PYR[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                        List_Neurone_PYR[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        List_Neurone_PYR[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        List_Neurone_PYR[neurone].add_I_synDend(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA_not[i])
                        List_Neurone_PYR[neurone].add_I_synDend(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA_not[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and i < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[0]
                        I_AMPA = List_Neurone_BAS[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        I_NMDA = List_Neurone_BAS[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)


                        List_Neurone_BAS[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        List_Neurone_BAS[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and i < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[1]
                        I_AMPA = List_Neurone_OLM[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        I_NMDA = List_Neurone_OLM[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                        List_Neurone_OLM[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        List_Neurone_OLM[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and i < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[2]
                        I_AMPA = List_Neurone_BIS[neurone].I_AMPA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        I_NMDA = List_Neurone_BIS[neurone].I_NMDA2(Vpre_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i], tps_start)

                        List_Neurone_BIS[neurone].add_I_synSoma(I_AMPA, PS.PreSynaptic_Soma_Dend_AMPA[i])
                        List_Neurone_BIS[neurone].add_I_synSoma(I_NMDA, PS.PreSynaptic_Soma_Dend_AMPA[i])


                if len(PS.PreSynaptic_Cell_GABA[i]) > 0:
                    Cell_GABA = PS.PreSynaptic_Cell_GABA[i]
                    Vpre_GABA = np.zeros(len(Cell_GABA))
                    for k, c in enumerate(Cell_GABA):
                        if c >= 0 and c < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                            Vpre_GABA[k] = List_Neurone_PYR[c].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and c < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                            Vpre_GABA[k] = List_Neurone_BAS[c - Nb_of_PYR].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and c < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                            Vpre_GABA[k] = List_Neurone_OLM[c - Nb_of_PYR - Nb_of_BAS].VsOutput()
                        elif c >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and c < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                            Vpre_GABA[k] = List_Neurone_BIS[c - Nb_of_PYR - Nb_of_BAS - Nb_of_SOM].VsOutput()

                    if i >= 0 and i < Nb_of_PYR_BAS_SOM_BIS_sum[0]:
                        neurone =  i
                        I_GABA = List_Neurone_PYR[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                        List_Neurone_PYR[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])
                        List_Neurone_PYR[neurone].add_I_synDend(I_GABA, PS.PreSynaptic_Soma_Dend_GABA_not[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[0] and i < Nb_of_PYR_BAS_SOM_BIS_sum[1]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[0]
                        I_GABA = List_Neurone_BAS[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                        List_Neurone_BAS[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[1] and i < Nb_of_PYR_BAS_SOM_BIS_sum[2]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[1]
                        I_GABA = List_Neurone_OLM[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                        List_Neurone_OLM[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                    elif i >= Nb_of_PYR_BAS_SOM_BIS_sum[2] and i < Nb_of_PYR_BAS_SOM_BIS_sum[3]:
                        neurone =  i - Nb_of_PYR_BAS_SOM_BIS_sum[2]
                        I_GABA = List_Neurone_BIS[neurone].I_GABA2(Vpre_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

                        List_Neurone_BIS[neurone].add_I_synSoma(I_GABA, PS.PreSynaptic_Soma_Dend_GABA[i])

            for i in range(Nb_of_PYR):
                List_Neurone_PYR[i].rk4()
            for i in range(Nb_of_BAS):
                List_Neurone_BAS[i].rk4()
            for i in range(Nb_of_SOM):
                List_Neurone_OLM[i].rk4()
            for i in range(Nb_of_BIS):
                List_Neurone_BIS[i].rk4()

            for i in range(Nb_of_PYR):
                pyrVs[i, tt] = List_Neurone_PYR[i].y[0]
                pyrVd[i, tt] = List_Neurone_PYR[i].y[1]
            for i in range(Nb_of_BAS):
                basketVs[i, tt] = List_Neurone_BAS[i].y[0]
            for i in range(Nb_of_SOM):
                olmVs[i, tt] = List_Neurone_OLM[i].y[0]
            for i in range(Nb_of_BIS):
                BisVs[i, tt] = List_Neurone_BIS[i].y[0]


        return t, pyrVs, pyrVd, olmVs, basketVs, BisVs


    def get_PYRCA1_Variables(self):
        return pyrCA1.get_Variable_Names()

    def get_BAS_Variables(self):
        return Basket.get_Variable_Names()

    def get_SOM_Variables(self):
        return OLM_SOM.get_Variable_Names()

    def get_BIS_Variables(self):
        return BIS.get_Variable_Names()


