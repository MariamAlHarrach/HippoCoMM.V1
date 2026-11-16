__author__ = 'Mariam'


import numpy as np
import matplotlib
from math import *
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D

class LFP:
    def __init__(self, Fs=25000, type=0,re=10,h=2000, tx=0, ty=0,pos=None):
        # Electrode properties
        #re: electrode radius
        #tx:  tetax angle
        #ty: teta y angle
        #type=0 ---> disc, 1 ---> Cylinder
        self.type=type
        if pos is None:
            self.electrode_pos = [0e-3, 0e-3, 2000e-3]
        else:
            self.electrode_pos=pos #in micrometers
        self.r_e = re
        self.tx = tx
        self.ty = ty
        self.h=h

        #Geometrical characteristics of the Soma
        S_soma = 18. # soma surface for layers 2/3,4,5 and 6 resp in micrometres
        l_soma = 6.32   # soma height for layers 2/3,4,5 and 6 resp in micrometres
        #Geometrical characteristics of the dendrite
        S_dend = 5191  # dendrite surface for layers 2/3,4,5 and 6 resp in micrometres
        l_dend = 275.8  # dendrite tree length


        # # proportion of soma to total area
        # p = [0.09, 0.04, 0.042, 0.062] #0.15

        #distance between sink and source
        self.lsd = l_soma / 2 + l_dend / 2
        Ssd = S_soma+S_dend

        gc = 1e-8  # intercompartment conductance 1mS/cm2 --->1e-8 mS/micrometre

        # Electrolyte conductivity
        self.sigma = 33 * 1e-5  # conductivity = 33.0 * 10e-5 ohm - 1.mm - 1

        self.Ksd = gc * self.lsd * Ssd / (4 * np.pi * self.sigma)

    def getDiscPts(self, rad, step=0.01):


        i_coords, j_coords = np.meshgrid(np.arange(-rad,rad,step), np.arange(-rad,rad,step), indexing='ij')

        corrds = np.stack((i_coords.flatten(), j_coords.flatten(), 0*i_coords.flatten()) ).T

        dist = np.linalg.norm(corrds-np.array([0,0,0]),axis=1)

        D  = corrds[dist<=rad]

        return D


    def getCylPts(self, rad=400,th=2000,max_pts=1000):

        z = np.linspace(0, th, int(np.sqrt(max_pts)))
        theta = np.linspace(0, 2 * np.pi, int(np.sqrt(max_pts)))
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = rad * np.cos(theta_grid)
        y_grid = rad * np.sin(theta_grid)
        Cyl=np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())).T
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter3D(Cyl[:,0], Cyl[:,1], Cyl[:,2],'.', alpha=0.5)
        # plt.show()
        return Cyl



    def get_electrode_coordinates(self, nbpots=1000):
        if self.r_e ==0:
            Ds = np.array([self.electrode_pos])
        else:
            if self.type==0: ###disc
                D = self.getDiscPts(self.r_e,step=1/((nbpots/(np.pi*self.r_e*self.r_e))**0.5))
            elif self.type==1: ###Cylinder
                D = self.getCylPts(rad=self.r_e, th=self.h, max_pts=nbpots)
            rx = np.array([[1, 0, 0], [0, cos(radians(self.tx)), - sin(radians(self.tx))],
                           [0, sin(radians(self.tx)), cos(radians(self.tx))]])
            ry = np.array([[cos(radians(self.ty)), 0, sin(radians(self.ty))],
                           [0, 1, 0], [-sin(radians(self.ty)), 0, cos(radians(self.ty))]])
            Ds =  np.matmul(D, np.matmul(rx, ry)) + self.electrode_pos
        return Ds


    def addnoise(self, lfp, SNR=35):
        Plfp = np.mean(lfp ** 2)
        Pnoise = Plfp / 10 ** (SNR / 10)
        noise = np.random.normal(0, np.sqrt(Pnoise), len(lfp))
        lfpn = lfp + noise
        return lfpn

    def compute_dipoles(self, Vsd, cellspos, Cellsubtypes,layers):#VSD of pyr cells #Cellpos of pyrcells

        cellspos = cellspos #position of the Neurons
        disc = self.get_electrode_coordinates()
        v = np.zeros(Vsd.shape[1])
        Cellsubtypes = np.array(Cellsubtypes) #subtypes of PCs
        K = np.array([self.Ksd[ind - 1] for ind in layers])
        Normals = np.array([0,0,1]) #unitary vector in the z axis
        for di in  disc:

            dist = np.linalg.norm(di - cellspos, axis=1) ** 2
            vect_projectOnto = cellspos - di
            projection = (vect_projectOnto * np.sum(Normals * vect_projectOnto, axis=1)[:, None])/ np.sum(vect_projectOnto * vect_projectOnto, axis=1)[:, None]
            norm_vect_projectOnto = np.linalg.norm(vect_projectOnto,axis = 1)
            norm_projection = np.linalg.norm(projection + vect_projectOnto,axis = 1)
            U  = norm_vect_projectOnto - norm_projection
            U[U<0] = 0
            U[Cellsubtypes == 2] = -U[Cellsubtypes == 2]
            v += np.matmul(K*U/dist, Vsd)/len(disc) #[Fuglevand et al. 1992]
        return v

    def compute_multidipoles(self, Vsd, Vsa,cellspos, Cellsubtypes,layers):#VSD of pyr cells #Cellpos of pyrcells

        cellspos = cellspos  # position of the Neurons
        disc = self.get_electrode_coordinates()
        vsd = np.zeros(Vsd.shape[1])
        vsa = np.zeros(Vsd.shape[1])

        Cellsubtypes = np.array(Cellsubtypes)  # subtypes of PCs
        Ksd = np.array([self.Ksd[ind - 1] for ind in layers])
        Ksa = np.array([self.Ksa[ind - 1] for ind in layers])
        mid_sd = np.array([[0,0,self.lsd[ind - 1]] for ind in layers])
        mid_sa = np.array([[0,0,self.lsa[ind - 1]] for ind in layers])

        d1_pos= cellspos+mid_sd
        d2_pos=cellspos-mid_sa


        Normals = np.array([0, 0, 1])  # unitary vector in the z axis
        for di in disc:

            #dipole 1
            dist = np.linalg.norm(di - d1_pos, axis=1) ** 2
            vect_projectOnto = d1_pos - di
            projection = (vect_projectOnto * np.sum(Normals * vect_projectOnto, axis=1)[:, None]) / np.sum(
                vect_projectOnto * vect_projectOnto, axis=1)[:, None]
            norm_vect_projectOnto = np.linalg.norm(vect_projectOnto, axis=1)
            norm_projection = np.linalg.norm(projection + vect_projectOnto, axis=1)
            U = norm_vect_projectOnto - norm_projection
            # U[U < 0] = 0
            U[Cellsubtypes == 2] = -U[Cellsubtypes == 2]
            vsd += np.matmul(Ksd * U / dist, Vsd) / len(disc)


            #dipole 2
            dist = np.linalg.norm(di - d2_pos, axis=1) ** 3
            vect_projectOnto = d2_pos - di
            projection = (vect_projectOnto * np.sum(Normals * vect_projectOnto, axis=1)[:, None]) / np.sum(
                vect_projectOnto * vect_projectOnto, axis=1)[:, None]
            norm_vect_projectOnto = np.linalg.norm(vect_projectOnto, axis=1)
            norm_projection = np.linalg.norm(projection + vect_projectOnto, axis=1)
            U = norm_vect_projectOnto - norm_projection
            # U[U < 0] = 0
            U[Cellsubtypes == 2] = -U[Cellsubtypes == 2]
            vsa += np.matmul(Ksa * U / dist, Vsa) / len(disc)

        return vsd+vsa



    def Compute_distance_projection(self, E_xyz,CellPosition):
        ##### compute projection #############

        Normals = np.array([0, 0, 1])  # unitary vector in the z axis
        res = np.zeros((E_xyz.shape[0], CellPosition.shape[0]))
        for ind, di in enumerate(E_xyz):
            for k in range(1):  # CellPosition.shape[0]):

                vect_projectOnto2 = CellPosition[k, :] - di
                projection2 = (vect_projectOnto2 * np.dot(Normals, vect_projectOnto2) / np.dot(vect_projectOnto2,
                                                                                               vect_projectOnto2))
                norm_vect_projectOnto2 = np.linalg.norm(vect_projectOnto2)
                norm_projection2 = np.linalg.norm(projection2 + vect_projectOnto2)
                res[ind, :] = norm_vect_projectOnto2 - norm_projection2
            # dipole 1
            # dist = np.linalg.norm(di - CellPosition, axis=1)
            # vect_projectOnto = (CellPosition - di).T
            # projection = vect_projectOnto * np.dot(Normals, vect_projectOnto) / np.sum(
            #     vect_projectOnto * vect_projectOnto,
            #     axis=0)
            # norm_vect_projectOnto = np.linalg.norm(vect_projectOnto, axis=0)
            # norm_projection = np.linalg.norm(projection + vect_projectOnto, axis=0)
            # res[ind, :] = (norm_vect_projectOnto - norm_projection) * dist
        return res


    def get_projection(self,Cellpos,mono=False):

        ## initialize the coordiantes of the neuron's parts
        CellPosition_d = []
        CellPosition_s = []
        ## baseline neuron
        target = 0#Cell_morphology.Neuron(0, 1)
        ## update neuron and get coordinates
        for n in range(Cellpos.shape[0]):
            pos = Cellpos[n]

            CellPosition_d.append(np.array([pos[0], pos[1], pos[2] + self.lsd]))
            CellPosition_s.append(np.array([pos[0], pos[1], pos[2]]))



        CellPosition_s = np.array(CellPosition_s)
        CellPosition_d = np.array(CellPosition_d)


        ###get electrode coordinates
        E_xyz = self.get_electrode_coordinates(nbpots=100)

        ###compute the distance between the electrode and the location of the transmembrane current
        if mono == True:
            Dis_d = distance.cdist(E_xyz, CellPosition_d, 'euclidean')
            Dis_s = distance.cdist(E_xyz, CellPosition_s, 'euclidean')

        else:
            Dis_d = self.Compute_distance_projection(E_xyz, CellPosition_d)
            Dis_s = self.Compute_distance_projection(E_xyz, CellPosition_s)

        return  Dis_d, Dis_s


    def computeLFPmono(self,PPS,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos):

        Dis_d, Dis_d1, Dis_d23, Dis_d4, Dis_d5, Dis_d6, Dis_s_mid, Dis_a,Dis_dend=self.get_projection(Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos,mono=True)

        #postsynapticpotentials
        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        pyrPPSI_a = PPS['PPSI_a']
        pyrPPSE_Dpyr=PPS['PPSE_Dpyr']
        pyrPPSE_Th=PPS['PPSE_Th']

        LFP = np.zeros(PPSE[0].shape[1])

        Res = np.zeros(PPSE[0].shape[1])
        ### PPSE dendrite
        Res += np.sum(np.dot(1 / Dis_d1, PPSE[0,:, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSE[0, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d23, PPSE[1, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSE[1, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d4, PPSE[2, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSE[2, :, :]), axis=0)


        Res += np.sum(np.dot(1 / Dis_d5, PPSE[3, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSE[3, :, :]), axis=0)


        Res += np.sum(np.dot(1 / Dis_d6, PPSE[4, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSE[4, :, :]), axis=0)


        ### PPSI dendrite

        Res += np.sum(np.dot(1 / Dis_d1, PPSI[0, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSI[0, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d23, PPSI[1, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSI[1, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d4, PPSI[2, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSI[2, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d5, PPSI[3, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSI[3, :, :]), axis=0)

        Res += np.sum(np.dot(1 / Dis_d6, PPSI[4, :, :]), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, PPSI[4, :, :]), axis=0)

                ### PPSI soma
        Res += np.sum(np.dot(1 / Dis_s_mid, pyrPPSI_s), axis=0)
        Res -= np.sum(np.dot(1 / Dis_d, pyrPPSI_s), axis=0)


                ### PPSI Axon
        Res += np.sum(np.dot(1 / Dis_a, pyrPPSI_a), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSI_a), axis=0)

                ###PPSE External
        Res += np.sum(np.dot(1 / Dis_d, pyrPPSE_Dpyr), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Dpyr), axis=0)

        Res += np.sum(np.dot(1 / Dis_d, pyrPPSE_Th), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Th), axis=0)

        LFP += Res/(4 * np.pi * self.sigma)

        return LFP*1e-6 #cm is in uF/cm2 to convert into micrometer2

    def computeLFPmono2(self,PPS,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos,layers):



        disc = self.get_electrode_coordinates(nbpots=100)
        #postsynapticpotentials
        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        pyrPPSI_a = PPS['PPSI_a']
        pyrPPSE_Dpyr=PPS['PPSE_Dpyr']
        pyrPPSE_Th=PPS['PPSE_Th']
        #positions
        Dis_d, Dis_d1, Dis_d23, Dis_d4, Dis_d5, Dis_d6, Dis_s_mid, Dis_a,Dis_dend = self.get_projection(Cellpos, List_celltypes,
                                                                                               List_cellsubtypes,
                                                                                               Layertop_pos,mono=True)

        Res = np.zeros(PPSE[0].shape[1])
        LFP = np.zeros(PPSE[0].shape[1])

        p=np.array([self.P[layers[k]-1] for k in range(Dis_d1.shape[1])])/100#get the layer percentages

        for l in range(5):
            coef = (1/ Dis_dend[l,:] - np.sum(np.divide(p[:,l,0:-1][:,np.newaxis,:].T,Dis_dend[:,:,:]),axis=0) - np.divide(p[:,l,-1],Dis_s_mid))
            Res +=  np.sum(np.dot(coef, PPSE[l,:, :]),axis=0)
            Res +=  np.sum(np.dot(coef, PPSI[l,:, :]),axis=0)


        ### PPSI soma

        Res += np.sum(np.dot(1/Dis_s_mid, pyrPPSI_s),axis=0)
        Res -= np.sum(np.dot(1/Dis_d, pyrPPSI_s),axis=0)

        ### PPSI Axon

        Res += np.sum(np.dot(1/Dis_a, pyrPPSI_a),axis=0)
        Res -= np.sum(np.dot(1/Dis_s_mid, pyrPPSI_a),axis=0)

                ###PPSE External
        Res += 0.1*np.sum(np.dot(1 / Dis_d, pyrPPSE_Dpyr), axis=0)
        Res -= 0.1*np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Dpyr), axis=0)

        Res += np.sum(np.dot(1 / Dis_d, pyrPPSE_Th), axis=0)
        Res -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Th), axis=0)


        LFP += Res/(4 * np.pi * self.sigma)

        return LFP*1e-6

    def compute_LFP_PSP(self,PPS,Epos,Cellpos,List_celltypes,List_cellsubtypes,Layertop_pos):

        # electrode position
        electrode_pos = Epos

        disc = self.get_electrode_coordinates(nbpots=100)
        # postsynapticpotentials
        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        pyrPPSI_a = PPS['PPSI_a']
        pyrPPSE_Dpyr = PPS['PPSE_Dpyr']
        pyrPPSE_Th = PPS['PPSE_Th']

        # positions
        Dis_d, Dis_d1, Dis_d23, Dis_d4, Dis_d5, Dis_d6, Dis_s_mid, Dis_a, Dis_dend = self.get_projection(Cellpos,
                                                                                                         List_celltypes,
                                                                                                         List_cellsubtypes,
                                                                                                         Layertop_pos,mono=True)
        LFP = np.zeros(PPSE[0].shape[1])

        Res = np.zeros(PPSE[0].shape[1])
        Res_PPSE_d = np.zeros(PPSE[0].shape[1])
        Res_PPSI_d = np.zeros(PPSE[0].shape[1])
        Res_PPSI_s = np.zeros(PPSE[0].shape[1])
        Res_PPSI_a = np.zeros(PPSE[0].shape[1])
        Res_PPSE_Dpyr = np.zeros(PPSE[0].shape[1])
        Res_PPSE_Th = np.zeros(PPSE[0].shape[1])

        ### PPSE dendrite

        Res_PPSE_d += np.sum(np.dot(1 / Dis_d1, PPSE[0,:, :]), axis=0)
        Res_PPSE_d -= np.sum(np.dot(1 / Dis_s_mid, PPSE[0, :, :]), axis=0)

        Res_PPSE_d += np.sum(np.dot(1 / Dis_d23, PPSE[1, :, :]), axis=0)
        Res_PPSE_d -= np.sum(np.dot(1 / Dis_s_mid, PPSE[1, :, :]), axis=0)

        Res_PPSE_d += np.sum(np.dot(1 / Dis_d4, PPSE[2, :, :]), axis=0)
        Res_PPSE_d -= np.sum(np.dot(1 / Dis_s_mid, PPSE[2, :, :]), axis=0)


        Res_PPSE_d += np.sum(np.dot(1 / Dis_d5, PPSE[3, :, :]), axis=0)
        Res_PPSE_d -= np.sum(np.dot(1 / Dis_s_mid, PPSE[3, :, :]), axis=0)


        Res_PPSE_d += np.sum(np.dot(1 / Dis_d6, PPSE[4, :, :]), axis=0)
        Res_PPSE_d -= np.sum(np.dot(1 / Dis_s_mid, PPSE[4, :, :]), axis=0)


        ### PPSI dendrite

        Res_PPSI_d += np.sum(np.dot(1 / Dis_d1, PPSI[0, :, :]), axis=0)
        Res_PPSI_d -= np.sum(np.dot(1 / Dis_s_mid, PPSI[0, :, :]), axis=0)

        Res_PPSI_d += np.sum(np.dot(1 / Dis_d23, PPSI[1, :, :]), axis=0)
        Res_PPSI_d -= np.sum(np.dot(1 / Dis_s_mid, PPSI[1, :, :]), axis=0)

        Res_PPSI_d += np.sum(np.dot(1 / Dis_d4, PPSI[2, :, :]), axis=0)
        Res_PPSI_d -= np.sum(np.dot(1 / Dis_s_mid, PPSI[2, :, :]), axis=0)

        Res_PPSI_d += np.sum(np.dot(1 / Dis_d5, PPSI[3, :, :]), axis=0)
        Res_PPSI_d -= np.sum(np.dot(1 / Dis_s_mid, PPSI[3, :, :]), axis=0)

        Res_PPSI_d += np.sum(np.dot(1 / Dis_d6, PPSI[4, :, :]), axis=0)
        Res_PPSI_d -= np.sum(np.dot(1 / Dis_s_mid, PPSI[4, :, :]), axis=0)

                ### PPSI soma
        Res_PPSI_s += np.sum(np.dot(1 / Dis_s_mid, pyrPPSI_s), axis=0)
        Res_PPSI_s -= np.sum(np.dot(1 / Dis_d, pyrPPSI_s), axis=0)


                ### PPSE Axon
        Res_PPSI_a += np.sum(np.dot(1 / Dis_a, pyrPPSI_a), axis=0)
        Res_PPSI_a -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSI_a), axis=0)

                ### PPSE External
        Res_PPSE_Dpyr += 0.1*np.sum(np.dot(1 / Dis_d, pyrPPSE_Dpyr), axis=0)
        Res_PPSE_Dpyr -= 0.1*np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Dpyr), axis=0)

        Res_PPSE_Th += np.sum(np.dot(1 / Dis_d, pyrPPSE_Th), axis=0)
        Res_PPSE_Th -= np.sum(np.dot(1 / Dis_s_mid, pyrPPSE_Th), axis=0)

        LFP += Res/(4 * np.pi * self.sigma)

        return Res_PPSE_d, Res_PPSI_d, Res_PPSI_s, Res_PPSI_a, Res_PPSE_Dpyr, Res_PPSE_Th

    def compute_LFP_CSD(self,I_all,PPS,Cellpos):

        PPSE = PPS['PPSE']
        PPSI = PPS['PPSI']
        pyrPPSI_s = PPS['PPSI_s']
        #positions
        Dis_d, Dis_s = self.get_projection(Cellpos,mono=False)

        # S_S,S_d,S_A= self.get_surfaces(Cellpos, List_celltypes, List_cellsubtypes)

        self.pyrI_S= I_all['pyrI_S']
        self.pyrI_d= I_all['pyrI_d']

        ###compute currents sources from voltage dependant channels



        if np.size(self.pyrI_S)==1:
            # I_S = np.array([1e-8*S_S[k] * self.pyrI_S[0][0][k] for k in range(len(S_S))]) # PYRI in mA/cm2, I_S in mA
            # I_d = np.array([1e-8*S_d[k] * self.pyrI_d[0][0][k] for k in range(len(S_S))])
            # I_A = np.array([1e-8*S_A[k] * self.pyrI_A[0][0][k] for k in range(len(S_S))])


            I_S = 1e-8*self.pyrI_S[0][0]
            I_d =1e-8*self.pyrI_d[0][0]

        else:
            # I_S = np.array([1e-8*S_S[k] * self.pyrI_S[k] for k in range(len(S_S))])
            # I_d = np.array([1e-8*S_d[k] * self.pyrI_d[k] for k in range(len(S_S))])
            # I_A = np.array([1e-8*S_A[k] * self.pyrI_A[k] for k in range(len(S_S))])

            I_S = 1e-8*self.pyrI_S
            I_d =1e-8*self.pyrI_d

        LFPsoma = np.mean(np.dot(1 / Dis_s,I_S), axis=0)
        LFPdend = np.mean(np.dot(1 / Dis_d, I_d), axis=0)


        ####Compute currents from synapses
        # I_Syn_S = np.array([S_S[k] * pyrPPSI_s[k] for k in range(len(S_S))])
        # I_Syn_A = np.array([S_A[k] * pyrPPSI_a[k] for k in range(len(S_S))])
        # I_Syn_DE=np.zeros(np.shape(PPSE))
        # I_Syn_DI=np.zeros(np.shape(PPSE))
        #
        # for l in range(5):
        #     I_Syn_DE[l] = np.array([S_d[k] * PPSE[l][k] for k in range(len(S_S))])
        #     I_Syn_DI[l] = np.array([S_d[k] * PPSI[l][k] for k in range(len(S_S))])




        LFP_SynD_E =  np.mean(1e-8*np.dot(1/ Dis_d, PPSE),axis=0)
        LFP_SynD_I =  np.mean(1e-8*np.dot(1/ Dis_d, PPSI),axis=0)

        LFP_SynS_I = np.mean(1e-8*np.dot(1 / Dis_s, pyrPPSI_s), axis=0)



        LFP= -1*(LFPsoma+LFPdend)+(LFP_SynD_E+LFP_SynD_I+LFP_SynS_I)
        LFP=LFP / (4 * np.pi * self.sigma*1e-3)

        # fig, (ax1, ax2) = plt.subplots(2, 1)
        #
        # ax1.plot(LFPsoma)
        # ax1.plot(LFPdend)
        # ax1.plot(LFPAIS)
        # ax1.plot((LFPsoma+LFPdend+LFPAIS))
        # ax1.legend(['soma','dend','AIS','LFPall'])
        # #
        # ax2.plot(LFP_SynD_E)
        # ax2.plot(LFP_SynD_I)
        # ax2.plot(LFP_SynS_I)
        # ax2.plot(LFP_SynA_I)
        # ax2.plot(LFP_SynD_E+LFP_SynD_I+LFP_SynS_I+LFP_SynA_I)
        # #
        # # ax2.plot(0.2*(LFPsoma+LFPdend+LFPAIS))
        # #
        # ax2.legend(['LFP_SynD_E','LFP_SynD_I','LFP_SynS_I','LFP_SynA_I','DPYR','all'])
        # plt.show()
        # plt.plot(LFP)
        #
        #
        # plt.show()
        return LFP



    def computeMEAV(self, Vsd, cellsp, nbelectrodes=16): #for microelectrode array
        V = []
        x = np.linspace(-150e-3, 150e-3, 4)
        y = np.linspace(-150e-3, 150e-3, 4)
        xx, yy = np.meshgrid(x, y)

        c = np.zeros(shape=(3, 16))
        c[0, :] = np.reshape(xx, (1, 16))
        c[1, :] = np.reshape(yy, (1, 16))
        D = self.getDiscPts(20 * 1e-3, 0.01)
        for cpos in np.transpose(c):
            disc = self.ApplyElectrodeTransform(D, center=cpos + [300e-3, 300e-3, 100e-3])
            Vdip = np.array([0, 0, 1])
            w_int = np.zeros(shape=(1, len(cellsp)))
            v = np.zeros(shape=(len(np.transpose(Vsd)), len(np.transpose(cellsp))))
            i = 0
            for di in np.transpose(disc):
                j = 0
                for dj in cellsp:
                    w_int[0, j] = np.matmul(np.subtract(di, dj), np.transpose(Vdip)) / (
                        np.sum(np.square(np.subtract(di, dj)))) ** (3 / 2)
                    j += 1
                v[:, i] = self.K * np.matmul(w_int, Vsd)
                if i < len(np.transpose(cellsp)) - 1:
                    i += 1

            Vd = np.sum(v, axis=1) / i
            #        V = self.addnoise(V)
            # plt.plot(V)
            b, a = signal.butter(2, 1024 / self.fs, 'low')
            V_lfp = signal.lfilter(b, a, V)
            # plt.plot(V,'b',V_lfp,'r')
            # plt.show()
            V.append(Vd * 1000)
        return V


# file = loadmat('Test')  #file path
# Vs = file['Vs']
# Vd = file['Vd']
# NBPYR=file['NBPYR']
# Pos=file['PosCells']
# Pos=Pos[0]
# Pos=[np.array(Pos[i][0:NBPYR[0][i],:]) for i in [1,2,3,4]]
# LayerNB=file['layer_NbCells']
# PYRsubtypes=file['PYRSubtypes']
#
#
# Vsd = np.subtract(Vs, Vd)
# # for i in range(len(Vsd)):
# #     plt.plot(Vsd[i,:])
# #     plt.show()
#
# ComputeLFP=LFP( re=800, tx=90, ty=0)
# ComputeLFP.get_electrode_coordinates_SEEG()
# LFP=ComputeLFP.compute_dipoles(Vsd=Vsd,cellspos=Pos,Cellsubtypes=PYRsubtypes)
# plt.plot(LFP)
# plt.show()

