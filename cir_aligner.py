import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path


# object: 
class RECORD():
    def __init__(self):
        self.loglik_track = 0
        self.d18O = None
        self.C14 = None
        self.ETC = None
        CAL = pd.read_csv('Calibration_Models/IntCal20.csv')
        IntCal20 = pd.DataFrame(columns=['A', 'MU', 'SIG', 'PMU', 'PSIG'])
        A = np.asarray(CAL.cal_bp/1000).reshape(-1)
        MU = np.asarray(CAL.a14C_age/1000).reshape(-1)
        SIG = np.asarray(CAL.error/1000).reshape(-1)
        N = A.shape[0]
        PMU = np.zeros(N)
        PSIG = np.zeros(N)
        PMU[0] = (MU[1]-MU[0])/(A[1]-A[0])
        PMU[-1] = (MU[-1]-MU[-2])/(A[-1]-A[-2])
        PMU[1:-1] = ((MU[2:]-MU[1:-1])/(A[2:]-A[1:-1])+(MU[1:-1]-MU[:-2])/(A[1:-1]-A[:-2]))/2
        PSIG[0] = (SIG[1]-SIG[0])/(A[1]-A[0])
        PSIG[-1] = (SIG[-1]-SIG[-2])/(A[-1]-A[-2])
        PSIG[1:-1] = ((SIG[2:]-SIG[1:-1])/(A[2:]-A[1:-1])+(SIG[1:-1]-SIG[:-2])/(A[1:-1]-A[:-2]))/2
        IntCal20.A = A
        IntCal20.MU = MU
        IntCal20.SIG = SIG
        IntCal20.PMU = PMU
        IntCal20.PSIG = PSIG
        CAL = pd.read_csv('Calibration_Models/Marine20.csv')
        Marine20 = pd.DataFrame(columns=['A', 'MU', 'SIG', 'PMU', 'PSIG'])
        A = np.asarray(CAL.cal_bp/1000).reshape(-1)
        MU = np.asarray(CAL.a14C_age/1000).reshape(-1)
        SIG = np.asarray(CAL.error/1000).reshape(-1)
        N = A.shape[0]
        PMU = np.zeros(N)
        PSIG = np.zeros(N)
        PMU[0] = (MU[1]-MU[0])/(A[1]-A[0])
        PMU[-1] = (MU[-1]-MU[-2])/(A[-1]-A[-2])
        PMU[1:-1] = ((MU[2:]-MU[1:-1])/(A[2:]-A[1:-1])+(MU[1:-1]-MU[:-2])/(A[1:-1]-A[:-2]))/2
        PSIG[0] = (SIG[1]-SIG[0])/(A[1]-A[0])
        PSIG[-1] = (SIG[-1]-SIG[-2])/(A[-1]-A[-2])
        PSIG[1:-1] = ((SIG[2:]-SIG[1:-1])/(A[2:]-A[1:-1])+(SIG[1:-1]-SIG[:-2])/(A[1:-1]-A[:-2]))/2
        Marine20.A = A
        Marine20.MU = MU
        Marine20.SIG = SIG
        Marine20.PMU = PMU
        Marine20.PSIG = PSIG
        CAL = pd.read_csv('Calibration_Models/SHCal20.csv')
        SHCal20 = pd.DataFrame(columns=['A', 'MU', 'SIG', 'PMU', 'PSIG'])
        A = np.asarray(CAL.cal_bp/1000).reshape(-1)
        MU = np.asarray(CAL.a14C_age/1000).reshape(-1)
        SIG = np.asarray(CAL.error/1000).reshape(-1)
        N = A.shape[0]
        PMU = np.zeros(N)
        PSIG = np.zeros(N)
        PMU[0] = (MU[1]-MU[0])/(A[1]-A[0])
        PMU[-1] = (MU[-1]-MU[-2])/(A[-1]-A[-2])
        PMU[1:-1] = ((MU[2:]-MU[1:-1])/(A[2:]-A[1:-1])+(MU[1:-1]-MU[:-2])/(A[1:-1]-A[:-2]))/2
        PSIG[0] = (SIG[1]-SIG[0])/(A[1]-A[0])
        PSIG[-1] = (SIG[-1]-SIG[-2])/(A[-1]-A[-2])
        PSIG[1:-1] = ((SIG[2:]-SIG[1:-1])/(A[2:]-A[1:-1])+(SIG[1:-1]-SIG[:-2])/(A[1:-1]-A[:-2]))/2
        SHCal20.A = A
        SHCal20.MU = MU
        SHCal20.SIG = SIG
        SHCal20.PMU = PMU
        SHCal20.PSIG = PSIG
        self.CALIB = [IntCal20, Marine20, SHCal20]


    # intialzie the record:
    def init_data(self, path_to_d18O, path_to_14C, path_to_ETC):
        if path_to_d18O.strip() == '':
            self.d18O = pd.DataFrame(columns=['depth', 'd18O'])
        else:
            self.d18O = pd.read_csv(path_to_d18O)
        if path_to_14C.strip() == '':
            self.C14 = pd.DataFrame(columns=['depth', 'age', 'error', 'dR', 'dSTD', 'cc'])
        else:
            self.C14 = pd.read_csv(path_to_14C)
        if path_to_ETC.strip() == '':
            self.ETC = pd.DataFrame(columns=['depth', 'age', 'unct'])
        else:
            self.ETC = pd.read_csv(path_to_ETC)
        self.depth = np.concatenate([self.d18O.depth.to_numpy(), self.C14.depth.to_numpy(), self.ETC.depth.to_numpy()])
        self.depth = np.unique(self.depth)
        self.d18O['ID'] = np.searchsorted(self.depth, self.d18O.depth.to_numpy())
        self.C14['ID'] = np.searchsorted(self.depth, self.C14.depth.to_numpy())
        self.ETC['ID'] = np.searchsorted(self.depth, self.ETC.depth.to_numpy())

    def get_model_parameters(self, cir_alpha, cir_beta, cir_rho):
        self.cir_alpha = cir_alpha
        self.cir_beta = float(cir_beta)
        self.cir_rho = float(cir_rho)
    
    def get_d18O_stack(self, path_to_d18O_stack):
        CAL = pd.read_csv(path_to_d18O_stack)
        self.d18O_stack = pd.DataFrame(columns=['A', 'MU', 'SIG', 'PMU', 'PSIG'])
        A = np.asarray(CAL['age']).reshape(-1)
        MU = np.asarray(CAL['mean']).reshape(-1)
        SIG = np.asarray(CAL['sigma']).reshape(-1)
        N = A.shape[0]
        PMU = np.zeros(N)
        PSIG = np.zeros(N)
        PMU[0] = (MU[1]-MU[0])/(A[1]-A[0])
        PMU[-1] = (MU[-1]-MU[-2])/(A[-1]-A[-2])
        PMU[1:-1] = ((MU[2:]-MU[1:-1])/(A[2:]-A[1:-1])+(MU[1:-1]-MU[:-2])/(A[1:-1]-A[:-2]))/2
        PSIG[0] = (SIG[1]-SIG[0])/(A[1]-A[0])
        PSIG[-1] = (SIG[-1]-SIG[-2])/(A[-1]-A[-2])
        PSIG[1:-1] = ((SIG[2:]-SIG[1:-1])/(A[2:]-A[1:-1])+(SIG[1:-1]-SIG[:-2])/(A[1:-1]-A[:-2]))/2
        self.d18O_stack.A = A
        self.d18O_stack.MU = MU
        self.d18O_stack.SIG = SIG
        self.d18O_stack.PMU = PMU
        self.d18O_stack.PSIG = PSIG

    def get_record_specific_parameters(self, a_14C, b_14C, a_d18O, b_d18O, scale_d18O, shift_d18O, std_param, start_depth, minimum_age_thredhold):
        self.a_14C = float(a_14C)
        self.b_14C = float(b_14C)
        self.a_d18O = float(a_d18O)
        self.b_d18O = float(b_d18O)
        self.scale_d18O = float(scale_d18O)
        self.shift_d18O = float(shift_d18O)
        self.inv_std_param = 1.0/float(std_param)
        self.start_depth = float(start_depth)
        self.minimum_age_thredhold = float(minimum_age_thredhold)

    def get_inducing_depths(self, num_inducing_depths):
        self.inducing_interval = (self.depth.max()+1e-8-self.start_depth)/(num_inducing_depths-1)
        self.inducing_depths = np.linspace(self.start_depth + 0.5*self.inducing_interval, self.start_depth + (num_inducing_depths+0.5)*self.inducing_interval, num_inducing_depths)
        

    # object functions:
    def SVGD(self, max_iters, num_samples):

        rng = np.random.default_rng()

        stack_sig_inv = np.asarray(self.d18O_stack.SIG, dtype=float) ** (-1)

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        eta = 1e-3

        THSD = 1e-4

        YY = self.d18O.to_numpy()
        YC = self.C14.to_numpy()
        YA = self.ETC.to_numpy()

        if YY.size > 0:
            ID_Y = YY[:, -1].astype(int)
            YY = YY[:, 1:-1]
        else:
            ID_Y = np.array([], dtype=int)
        
        if YC.size > 0:
            ID_C = YC[:, -1].astype(int)
            YC = YC[:, 1:-1]
        else:
            ID_C = np.array([], dtype=int)

        if YA.size > 0:
            ID_A = YA[:, -1].astype(int)
            YA = YA[:, 1:-1]
        else:
            ID_A = np.array([], dtype=int)
        
        D1 = np.asarray(self.depth, dtype=float) * float(self.inv_std_param)
        N1 = D1.shape[0]
        D1 = np.sort(D1)

        N0 = np.asarray(self.inducing_depths).shape[0]

        INT = float(self.inducing_interval) * float(self.inv_std_param)

        # AP = np.arange(0, INT, INT/num_samples)
        AP = np.linspace(0, INT, num_samples, endpoint=False)

        CNT = np.zeros((N1, ), dtype=int)
        HH = np.zeros((N1, N0), dtype=float)
        for n in range(N1):
            CNT[n] = int(np.floor((D1[n]-self.start_depth)/INT))
            HH[n, 0:CNT[n]] = INT
        CNT = np.minimum(CNT, N0-2)
        
        HH0 = - AP
        HH1 = np.minimum(D1[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :], INT)
        HH2 = np.maximum(D1[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :]-INT, 0.0)

        UQ = np.unique(CNT)
        QQ = np.zeros((len(UQ), N1), dtype=float)
        for n in range(N1):
            QQ[UQ==CNT[n], n] += 1.0

        UQ_Y = np.unique(ID_Y) if ID_Y.size > 0 else np.array([], dtype=int)
        QQ_Y = np.zeros((len(UQ_Y), len(ID_Y)), dtype=float)
        for n in range(len(ID_Y)):
            QQ_Y[UQ_Y==ID_Y[n], n] += 1.0
        
        UQ_C = np.unique(ID_C) if ID_C.size > 0 else np.array([], dtype=int)
        QQ_C = np.zeros((len(UQ_C), len(ID_C)), dtype=float)
        for n in range(len(ID_C)):
            QQ_C[UQ_C==ID_C[n], n] += 1.0

        UQ_A = np.unique(ID_A) if ID_A.size > 0 else np.array([], dtype=int)
        QQ_A = np.zeros((len(UQ_A), len(ID_A)), dtype=float)
        for n in range(len(ID_A)):
            QQ_A[UQ_A==ID_A[n], n] += 1.0
        
        RHO = np.sqrt(float(self.cir_rho)**INT)

        N0 = np.asarray(self.inducing_depths).shape[0]

        SAM = np.ones((N0, num_samples, int(2*self.cir_alpha)), dtype=float)/np.sqrt(float(2*self.cir_alpha))
        SAM += rng.normal(0.0, 0.01, size=(N0, num_samples, int(2*self.cir_alpha)))

        SAM_BIAS = np.ones((num_samples,), dtype=float)
        SAM_BIAS += rng.normal(0.0, 0.1, size=(num_samples,))

        Mw = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)
        Vw = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)

        Mw_BIAS = np.zeros((num_samples,), dtype=float)
        Vw_BIAS = np.zeros((num_samples,), dtype=float)

        for rr in range(1, max_iters+1):
            SAM_T = np.transpose(SAM, (1, 0, 2))

            sqnorm = np.sum(SAM_T**2, axis=1, keepdims=True)
            sqnorm_T = np.transpose(sqnorm, (1, 0, 2))

            inner = np.zeros((num_samples, num_samples, int(2*self.cir_alpha)), dtype=float)
            for k in range(int(2*self.cir_alpha)):
                inner[:, :, k] = SAM_T[:, :, k]@SAM[:, :, k]
            # inner = np.einsum("ika,kja->ija", SAM_T, SAM)
            KK = sqnorm + sqnorm_T - 2.0*inner

            KK_BIAS = (SAM_BIAS[:, None]-SAM_BIAS[None, :])**2
            
            hh = np.median(np.sum(KK, axis=2)+KK_BIAS)/np.log(num_samples)
            # hh = max(hh, 1e-12)

            KK = np.exp(-KK/hh)

            sum_KK = np.sum(KK, axis=0)
            PK = (2.0/hh*SAM)*sum_KK[None, :, :]
            for k in range(int(2*self.cir_alpha)):
                PK[:, :, k] -= (2.0/hh*SAM[:, :, k])@KK[:, :, k]
            # PK = (2.0/hh*SAM)*sum_KK[None, :, :] - np.einsum("ika,kja->ija", 2.0/hh*SAM, KK)

            KK_BIAS = np.exp(-KK_BIAS/hh)
            PK_BIAS = (2.0/hh*SAM_BIAS)*np.sum(KK_BIAS, axis=0) - (2.0/hh*SAM_BIAS)@KK_BIAS

            # Initialization
            PDEV = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)
            PDEV_BIAS = np.zeros((num_samples,), dtype=float)

            # Transition Model
            SAM_G = np.sum(SAM**2, axis=2)  # (N0, nSamples)

            PDEV[:-1, :, :] -= (2.0*self.cir_beta/(1.0-RHO**2)*RHO*(RHO*SAM[:-1, :, :]-SAM[1:, :, :]))
            PDEV[1:, :, :] += (2.0*self.cir_beta/(1.0-RHO**2)*(RHO*SAM[:-1, :, :]-SAM[1:, :, :]))
            PDEV[0, :, :] -= 2.0*self.cir_beta*SAM[0, :, :]

            # Emission Model
            AA = HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :] + SAM_BIAS[None, :]**2 + self.minimum_age_thredhold

            AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
            AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
            AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

            # d18O:
            if AA_Y.size > 0:
                MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)
                PMU = interp1d(self.d18O_stack.A, self.d18O_stack.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                PSIG = interp1d(self.d18O_stack.A, self.d18O_stack.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)

                ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O
                ALPHA = ((2.0*self.a_d18O+1.0)*(ZZ*(SIG**2)*PMU+(ZZ**2)*(SIG**3)*PSIG)/ (2.0*self.b_d18O+(ZZ**2)*(SIG**2))-SIG*PSIG)

                ALPHA = QQ_Y@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_Y, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            # 14C:
            if AA_C.size > 0:
                MU = np.zeros_like(AA_C)
                SIG = np.zeros_like(AA_C)
                PMU = np.zeros_like(AA_C)
                PSIG = np.zeros_like(AA_C)

                for k in range(len(self.CALIB)):
                    calib_id = (YC[:, -1].astype(int)==(k+1))

                    if np.any(calib_id):
                        cal_k = self.CALIB[k]
                        MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])
                        PMU[calib_id, :] = interp1d(cal_k.A, cal_k.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        PSIG[calib_id, :] = interp1d(cal_k.A, cal_k.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])

                WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
                ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

                if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                    ALPHA = - ZZ/WW*PMU + SIG*((ZZ/WW)**2)*PSIG - SIG/(WW**2)*PSIG
                else:
                    ALPHA = - (0.5+self.a_14C)/(self.b_14C+0.5*(ZZ**2))*(ZZ/WW*PMU-SIG*((ZZ/WW)**2)*PSIG) - SIG/(WW**2)*PSIG

                ALPHA = QQ_C@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_C, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            # ETC:
            if AA_A.size > 0:
                ALPHA = - (AA_A-YA[:, 0:1])*(YA[:, 1:2]**(-2))

                ALPHA = QQ_A@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_A, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            PHI = PK.copy()
            for k in range(int(2*self.cir_alpha)):
                PHI[:, :, k] += PDEV[:, :, k]@KK[:, :, k]
            # PHI = np.einsum("ika,kja->ija", PDEV, KK) + PK
            PHI_BIAS = PDEV_BIAS@KK_BIAS + PK_BIAS

            PHI = PHI/num_samples
            PHI_BIAS = PHI_BIAS/num_samples

            # Update (Adam-like)
            Mw = beta1*Mw - (1.0-beta1)*PHI
            Vw = beta2*Vw + (1.0-beta2)*(PHI**2)

            Mw_BIAS = beta1*Mw_BIAS - (1.0-beta1)*PHI_BIAS
            Vw_BIAS = beta2*Vw_BIAS + (1.0-beta2)*(PHI_BIAS**2)

            step = eta*np.sqrt(1.0-beta2**rr)/(1.0-beta1**rr)

            SAM = SAM - step*Mw/(np.sqrt(Vw)+epsilon)
            SAM_BIAS = SAM_BIAS - step*Mw_BIAS/(np.sqrt(Vw_BIAS)+epsilon)

        # final outputs
        self.F = SAM*np.sqrt(float(self.inv_std_param))
        SAM_G = np.sum(SAM**2, axis=2)

        AGE = SAM_BIAS[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :]
        self.AGE = AGE
        self.BIAS = SAM_BIAS


    def HMC(self, max_iters, num_samples):

        rng = np.random.default_rng()

        stack_sig_inv = np.asarray(self.d18O_stack.SIG, dtype=float) ** (-1)

        eps = 1e-3
        M = 30
        THSD = 1e-2

        YY = self.d18O.to_numpy()
        YC = self.C14.to_numpy()
        YA = self.ETC.to_numpy()

        if YY.size > 0:
            ID_Y = YY[:, -1].astype(int)
            YY = YY[:, 1:-1]
        else:
            ID_Y = np.array([], dtype=int)
        
        if YC.size > 0:
            ID_C = YC[:, -1].astype(int)
            YC = YC[:, 1:-1]
        else:
            ID_C = np.array([], dtype=int)

        if YA.size > 0:
            ID_A = YA[:, -1].astype(int)
            YA = YA[:, 1:-1]
        else:
            ID_A = np.array([], dtype=int)

        D1 = np.asarray(self.depth, dtype=float) * float(self.inv_std_param)
        N1 = D1.shape[0]
        D1 = np.sort(D1)

        N0 = np.asarray(self.inducing_depths).shape[0]
        INT = float(self.inducing_interval) * float(self.inv_std_param)

        # AP = np.arange(0, INT, INT / num_samples)
        AP = np.linspace(0, INT, num_samples, endpoint=False)

        CNT = np.zeros((N1, ), dtype=int)
        HH = np.zeros((N1, N0), dtype=float)
        for n in range(N1):
            CNT[n] = int(np.floor((D1[n]-self.start_depth)/INT))
            HH[n, :CNT[n]] = INT
        CNT = np.minimum(CNT, N0-2)
        
        HH0 = - AP
        HH1 = np.minimum(D1[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :], INT)
        HH2 = np.maximum(D1[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :]-INT, 0.0)

        UQ = np.unique(CNT)
        QQ = np.zeros((len(UQ), N1), dtype=float)
        for n in range(N1):
            QQ[UQ==CNT[n], n] += 1.0

        UQ_Y = np.unique(ID_Y) if ID_Y.size > 0 else np.array([], dtype=int)
        QQ_Y = np.zeros((len(UQ_Y), len(ID_Y)), dtype=float)
        for n in range(len(ID_Y)):
            QQ_Y[UQ_Y==ID_Y[n], n] += 1.0
        
        UQ_C = np.unique(ID_C) if ID_C.size > 0 else np.array([], dtype=int)
        QQ_C = np.zeros((len(UQ_C), len(ID_C)), dtype=float)
        for n in range(len(ID_C)):
            QQ_C[UQ_C==ID_C[n], n] += 1.0

        UQ_A = np.unique(ID_A) if ID_A.size > 0 else np.array([], dtype=int)
        QQ_A = np.zeros((len(UQ_A), len(ID_A)), dtype=float)
        for n in range(len(ID_A)):
            QQ_A[UQ_A==ID_A[n], n] += 1.0

        RHO = np.sqrt(float(self.cir_rho)**INT)

        SAM = self.F/np.sqrt(float(self.inv_std_param))
        SAM_G = np.sum(SAM**2, axis=2)
        SAM_BIAS = self.BIAS.copy()

        if SAM.shape[1] != num_samples:
            rand_seed = rng.integers(0, SAM.shape[1], size=num_samples)
            SAM = SAM[:, rand_seed, :]
            SAM_G = np.sum(SAM**2, axis=2)
            SAM_BIAS = SAM_BIAS[rand_seed]

        AA = SAM_BIAS[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :]
        AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
        AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
        AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

        PHI = rng.normal(0.0, 1.0, size=(N0, num_samples, int(2*self.cir_alpha)))
        PHI_BIAS = rng.normal(0.0, 1.0, size=(num_samples,))

        LOGLIK_old = np.sum(np.sum(-0.5*(PHI**2)-0.5*np.log(2.0*np.pi), axis=2), axis=0)
        LOGLIK_old += - 0.5*(PHI_BIAS**2) - 0.5*np.log(2.0*np.pi)
        LOGLIK_old += np.sum(-self.cir_beta*(SAM[-1, :, :]**2)-self.cir_beta/(1.0-RHO**2)*np.sum((SAM[:-1, :, :]-RHO*SAM[1:, :, :])**2, axis=0), axis=1)

        # d18O
        if AA_Y.size > 0:
            MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
            SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)

            ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O

            if not (np.isinf(self.a_d18O) and np.isinf(self.b_d18O)):
                LOGLIK_old += np.sum(-(self.a_d18O+0.5)*np.log(1.0+(ZZ**2)*(SIG**2)/(2.0*self.b_d18O))+np.log(SIG)-np.log(self.scale_d18O),axis=0)

        # 14C
        if AA_C.size > 0:
            MU = np.zeros_like(AA_C)
            SIG = np.zeros_like(AA_C)

            for k in range(len(self.CALIB)):
                calib_id = (YC[:, -1].astype(int)==(k+1))
                if np.any(calib_id):
                    cal_k = self.CALIB[k]

                    MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                    SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])

            WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
            ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

            if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                LOGLIK_old += np.sum(-0.5*(ZZ**2)-np.log(WW), axis=0)
            else:
                LOGLIK_old += np.sum(-(0.5+self.a_14C)*np.log(1.0+(ZZ**2)/(2.0*self.b_14C))-np.log(WW), axis=0)

        # ETC
        if AA_A.size > 0:
            LOGLIK_old += np.sum(-0.5*((AA_A-YA[:, 0:1])**2)*(YA[:, 1:2]**(-2)), axis=0)

        rand_seed_log = np.log(rng.random((max_iters, num_samples)))
        APP_RATE = np.zeros((max_iters,), dtype=float)
        

        # main loop:
        for rr in range(max_iters):
            SAM_G = np.sum(SAM**2, axis=2)

            AA = SAM_BIAS[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :]
            AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
            AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
            AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

            PDEV = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)
            PDEV_BIAS = np.zeros((num_samples,), dtype=float)

            # transition model
            PDEV[:-1, :, :] -= (2.0*self.cir_beta/(1.0-RHO**2)*RHO*(RHO*SAM[:-1, :, :]-SAM[1:, :, :]))
            PDEV[1:, :, :] += (2.0*self.cir_beta/(1.0-RHO**2)*(RHO*SAM[:-1, :, :]-SAM[1:, :, :]))
            PDEV[0, :, :] -= 2.0*self.cir_beta*SAM[0, :, :]

            # d18O
            if AA_Y.size > 0:
                MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)
                PMU = interp1d(self.d18O_stack.A, self.d18O_stack.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                PSIG = interp1d(self.d18O_stack.A, self.d18O_stack.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)

                ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O
                ALPHA = (2.0*self.a_d18O+1.0)*(ZZ*(SIG**2)*PMU+(ZZ**2)*(SIG**3)*PSIG)/(2.0*self.b_d18O+(ZZ**2)*(SIG**2)) - SIG*PSIG

                ALPHA = QQ_Y@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_Y, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            # 14C
            if AA_C.size > 0:
                MU = np.zeros_like(AA_C)
                SIG = np.zeros_like(AA_C)
                PMU = np.zeros_like(AA_C)
                PSIG = np.zeros_like(AA_C)

                for k in range(len(self.CALIB)):
                    calib_id = (YC[:, -1].astype(int)==(k+1))

                    if np.any(calib_id):
                        cal_k = self.CALIB[k]
                        MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])
                        PMU[calib_id, :] = interp1d(cal_k.A, cal_k.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        PSIG[calib_id, :] = interp1d(cal_k.A, cal_k.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])

                WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
                ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

                if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                    ALPHA = - ZZ/WW*PMU + SIG*((ZZ/WW)**2)*PSIG - SIG/(WW**2)*PSIG
                else:
                    ALPHA = - (0.5+self.a_14C)/(self.b_14C+0.5*(ZZ**2))*(ZZ/WW*PMU-SIG*((ZZ/WW)**2)*PSIG) - SIG/(WW**2)*PSIG

                ALPHA = QQ_C@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_C, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            # ETC:
            if AA_A.size > 0:
                ALPHA = - (AA_A-YA[:, 0:1])*(YA[:, 1:2]**(-2))

                ALPHA = QQ_A@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_A, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS)

            PHI_new = PHI + 0.5*eps*PDEV
            PHI_BIAS_new = PHI_BIAS + 0.5*eps*PDEV_BIAS

            SAM_new = SAM.copy()
            SAM_BIAS_new = SAM_BIAS.copy()

            for _ in range(M-1):
                SAM_new = SAM_new + eps*PHI_new
                SAM_BIAS_new = SAM_BIAS_new + eps*PHI_BIAS_new
                SAM_G_new = np.sum(SAM_new**2, axis=2)

                AA = SAM_BIAS_new[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G_new + HH0*SAM_G_new[0, :][None, :] + HH1*SAM_G_new[CNT, :] + HH2*SAM_G_new[CNT+1, :]
                AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
                AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
                AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

                PDEV = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)
                PDEV_BIAS = np.zeros((num_samples,), dtype=float)

                # transition model
                PDEV[:-1, :, :] -= (2.0*self.cir_beta/(1.0-RHO**2)*RHO*(RHO*SAM_new[:-1, :, :]-SAM_new[1:, :, :]))
                PDEV[1:, :, :] += (2.0*self.cir_beta/(1.0-RHO**2)*(RHO*SAM_new[:-1, :, :]-SAM_new[1:, :, :]))
                PDEV[0, :, :] -= 2.0*self.cir_beta*SAM_new[0, :, :]

                # d18O
                if AA_Y.size > 0:
                    MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                    SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)
                    PMU = interp1d(self.d18O_stack.A, self.d18O_stack.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                    PSIG = interp1d(self.d18O_stack.A, self.d18O_stack.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)

                    ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O
                    ALPHA = (2.0*self.a_d18O+1.0)*(ZZ*(SIG**2)*PMU+(ZZ**2)*(SIG**3)*PSIG)/(2.0*self.b_d18O+(ZZ**2)*(SIG**2)) - SIG*PSIG

                    ALPHA = QQ_Y@ALPHA

                    DELTA = np.zeros((N1, num_samples), dtype=float)
                    DELTA[UQ_Y, :] = ALPHA

                    PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                    PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                    PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                    PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                    PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

                # 14C
                if AA_C.size > 0:
                    MU = np.zeros_like(AA_C)
                    SIG = np.zeros_like(AA_C)
                    PMU = np.zeros_like(AA_C)
                    PSIG = np.zeros_like(AA_C)

                    for k in range(len(self.CALIB)):
                        calib_id = (YC[:, -1].astype(int)==(k+1))

                        if np.any(calib_id):
                            cal_k = self.CALIB[k]
                            MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                            SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])
                            PMU[calib_id, :] = interp1d(cal_k.A, cal_k.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                            PSIG[calib_id, :] = interp1d(cal_k.A, cal_k.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])

                    WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
                    ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

                    if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                        ALPHA = - ZZ/WW*PMU + SIG*((ZZ/WW)**2)*PSIG - SIG/(WW**2)*PSIG
                    else:
                        ALPHA = - (0.5+self.a_14C)/(self.b_14C+0.5*(ZZ**2))*(ZZ/WW*PMU-SIG*((ZZ/WW)**2)*PSIG) - SIG/(WW**2)*PSIG

                    ALPHA = QQ_C@ALPHA

                    DELTA = np.zeros((N1, num_samples), dtype=float)
                    DELTA[UQ_C, :] = ALPHA

                    PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                    PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                    PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                    PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                    PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

                # ETC:
                if AA_A.size > 0:
                    ALPHA = - (AA_A-YA[:, 0:1])*(YA[:, 1:2]**(-2))

                    ALPHA = QQ_A@ALPHA

                    DELTA = np.zeros((N1, num_samples), dtype=float)
                    DELTA[UQ_A, :] = ALPHA

                    PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                    PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                    PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                    PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                    PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

                PHI_new = PHI_new + eps*PDEV
                PHI_BIAS_new = PHI_BIAS + eps*PDEV_BIAS

            SAM_new = SAM_new + eps*PHI_new
            SAM_BIAS_new = SAM_BIAS_new + eps*PHI_BIAS_new
            SAM_G_new = np.sum(SAM_new**2, axis=2)

            AA = SAM_BIAS_new[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G_new + HH0*SAM_G_new[0, :][None, :] + HH1*SAM_G_new[CNT, :] + HH2*SAM_G_new[CNT+1, :]
            AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
            AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
            AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

            PDEV = np.zeros((N0, num_samples, int(2*self.cir_alpha)), dtype=float)
            PDEV_BIAS = np.zeros((num_samples,), dtype=float)

            # transition model
            PDEV[:-1, :, :] -= (2.0*self.cir_beta/(1.0-RHO**2)*RHO*(RHO*SAM_new[:-1, :, :]-SAM_new[1:, :, :]))
            PDEV[1:, :, :] += (2.0*self.cir_beta/(1.0-RHO**2)*(RHO*SAM_new[:-1, :, :]-SAM_new[1:, :, :]))
            PDEV[0, :, :] -= 2.0*self.cir_beta*SAM_new[0, :, :]

            # d18O
            if AA_Y.size > 0:
                MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)
                PMU = interp1d(self.d18O_stack.A, self.d18O_stack.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                PSIG = interp1d(self.d18O_stack.A, self.d18O_stack.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)

                ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O
                ALPHA = (2.0*self.a_d18O+1.0)*(ZZ*(SIG**2)*PMU+(ZZ**2)*(SIG**3)*PSIG)/(2.0*self.b_d18O+(ZZ**2)*(SIG**2)) - SIG*PSIG

                ALPHA = QQ_Y@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_Y, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

            # 14C
            if AA_C.size > 0:
                MU = np.zeros_like(AA_C)
                SIG = np.zeros_like(AA_C)
                PMU = np.zeros_like(AA_C)
                PSIG = np.zeros_like(AA_C)

                for k in range(len(self.CALIB)):
                    calib_id = (YC[:, -1].astype(int)==(k+1))

                    if np.any(calib_id):
                        cal_k = self.CALIB[k]
                        MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])
                        PMU[calib_id, :] = interp1d(cal_k.A, cal_k.PMU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        PSIG[calib_id, :] = interp1d(cal_k.A, cal_k.PSIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])

                WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
                ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

                if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                    ALPHA = - ZZ/WW*PMU + SIG*((ZZ/WW)**2)*PSIG - SIG/(WW**2)*PSIG
                else:
                    ALPHA = - (0.5+self.a_14C)/(self.b_14C+0.5*(ZZ**2))*(ZZ/WW*PMU-SIG*((ZZ/WW)**2)*PSIG) - SIG/(WW**2)*PSIG

                ALPHA = QQ_C@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_C, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

            # ETC:
            if AA_A.size > 0:
                ALPHA = - (AA_A-YA[:, 0:1])*(YA[:, 1:2]**(-2))

                ALPHA = QQ_A@ALPHA

                DELTA = np.zeros((N1, num_samples), dtype=float)
                DELTA[UQ_A, :] = ALPHA

                PDEV += (HH.T@DELTA)[:, :, None]*(2.0*SAM_new)
                PDEV[0, :, :] += (HH0*np.sum(DELTA, axis=0))[:, None]*(2.0*SAM_new[0, :, :])
                PDEV[UQ, :, :] += (QQ@(HH1*DELTA))[:, :, None]*(2.0*SAM_new[UQ, :, :])
                PDEV[UQ+1, :, :] += (QQ@(HH2*DELTA))[:, :, None]*(2.0*SAM_new[UQ+1, :, :])

                PDEV_BIAS += np.sum(DELTA, axis=0)*(2.0*SAM_BIAS_new)

            PHI_new = PHI_new + 0.5*eps*PDEV
            PHI_BIAS_new = PHI_BIAS_new + 0.5*eps*PDEV_BIAS

            AA = SAM_BIAS_new[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G_new + HH0*SAM_G_new[0, :][None, :] + HH1*SAM_G_new[CNT, :] + HH2*SAM_G_new[CNT+1, :]
            AA_Y = AA[ID_Y, :] if ID_Y.size > 0 else np.empty((0, num_samples))
            AA_C = AA[ID_C, :] if ID_C.size > 0 else np.empty((0, num_samples))
            AA_A = AA[ID_A, :] if ID_A.size > 0 else np.empty((0, num_samples))

            LOGLIK_new = np.sum(np.sum(-0.5*(PHI_new**2)-0.5*np.log(2.0*np.pi), axis=2), axis=0)
            LOGLIK_new += - 0.5*(PHI_BIAS_new**2) - 0.5*np.log(2.0*np.pi)
            LOGLIK_new += np.sum(-self.cir_beta*(SAM_new[-1, :, :]**2)-self.cir_beta/(1.0-RHO**2)*np.sum((SAM_new[:-1, :, :]-RHO*SAM_new[1:, :, :])**2, axis=0), axis=1)

            # d18O
            if AA_Y.size > 0:
                MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_Y)
                SIG = interp1d(self.d18O_stack.A, stack_sig_inv, kind="linear", axis=0, bounds_error=False, fill_value=THSD, assume_sorted=True)(AA_Y)

                ZZ = (YY-self.scale_d18O*MU-self.shift_d18O)/self.scale_d18O

                if not (np.isinf(self.a_d18O) and np.isinf(self.b_d18O)):
                    LOGLIK_new += np.sum(-(self.a_d18O+0.5)*np.log(1.0+(ZZ**2)*(SIG**2)/(2.0*self.b_d18O))+np.log(SIG)-np.log(self.scale_d18O),axis=0)

            # 14C
            if AA_C.size > 0:
                MU = np.zeros_like(AA_C)
                SIG = np.zeros_like(AA_C)

                for k in range(len(self.CALIB)):
                    calib_id = (YC[:, -1].astype(int)==(k+1))
                    if np.any(calib_id):
                        cal_k = self.CALIB[k]

                        MU[calib_id, :] = interp1d(cal_k.A, cal_k.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AA_C[calib_id, :])
                        SIG[calib_id, :] = interp1d(cal_k.A, cal_k.SIG, kind="linear", axis=0, bounds_error=False, fill_value=1.0/THSD, assume_sorted=True)(AA_C[calib_id, :])

                WW = np.sqrt(SIG**2+YC[:, 1][:, None]**2+YC[:, 3][:, None]**2)
                ZZ = (MU+YC[:, 2][:, None]-YC[:, 0][:, None])/WW

                if np.isinf(self.a_14C) and np.isinf(self.b_14C):
                    LOGLIK_new += np.sum(-0.5*(ZZ**2)-np.log(WW), axis=0)
                else:
                    LOGLIK_new += np.sum(-(0.5+self.a_14C)*np.log(1.0+(ZZ**2)/(2.0*self.b_14C))-np.log(WW), axis=0)

            # ETC
            if AA_A.size > 0:
                LOGLIK_new += np.sum(-0.5*((AA_A-YA[:, 0:1])**2)*(YA[:, 1:2]**(-2)), axis=0)

            accept_id = (~np.isnan(LOGLIK_new)) & ((LOGLIK_new - LOGLIK_old) > rand_seed_log[rr, :])

            SAM[:, accept_id, :] = SAM_new[:, accept_id, :]
            SAM_BIAS[accept_id] = SAM_BIAS_new[accept_id]

            PHI[:, accept_id, :] = PHI_new[:, accept_id, :]
            PHI_BIAS[accept_id] = PHI_BIAS_new[accept_id]
            LOGLIK_old[accept_id] = LOGLIK_new[accept_id]

            LOGLIK_old = LOGLIK_old - np.sum(np.sum(-0.5 * PHI ** 2 - 0.5 * np.log(2.0 * np.pi), axis=2), axis=0)
            LOGLIK_old = LOGLIK_old + 0.5 * PHI_BIAS ** 2 + 0.5 * np.log(2.0 * np.pi)

            PHI = rng.normal(0.0, 1.0, size=(N0, num_samples, int(2*self.cir_alpha)))
            PHI_BIAS = rng.normal(0.0, 1.0, size=(num_samples,))

            LOGLIK_old = LOGLIK_old + np.sum(np.sum(-0.5 * PHI ** 2 - 0.5 * np.log(2.0 * np.pi), axis=2), axis=0)
            LOGLIK_old = LOGLIK_old - 0.5 * PHI_BIAS ** 2 - 0.5 * np.log(2.0 * np.pi)

            APP_RATE[rr] = 100.0 * np.sum(accept_id) / num_samples

            if APP_RATE[rr] < 90.0:
                eps = eps / 1.01
            elif APP_RATE[rr] > 95.0:
                eps = eps * 1.01

        self.F = SAM*np.sqrt(float(self.inv_std_param))
        SAM_G = np.sum(SAM**2, axis=2)
        self.AGE = SAM_BIAS[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :]
        self.BIAS = SAM_BIAS
        self.acceptance_rate = APP_RATE

        self.D = self.inducing_depths[:-1] - 0.5*self.inducing_interval
        DD = self.D*float(self.inv_std_param)

        CNT = np.zeros((N0-1, ), dtype=int)
        HH = np.zeros((N0-1, N0), dtype=float)
        for n in range(N0-1):
            CNT[n] = int(np.floor((DD[n]-self.start_depth)/INT))
            HH[n, :CNT[n]] = INT
        CNT = np.minimum(CNT, N0-2)
        
        HH0 = - AP
        HH1 = np.minimum(DD[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :], INT)
        HH2 = np.maximum(DD[:, None]-self.start_depth-CNT[:, None]*INT+AP[None, :]-INT, 0.0)

        self.AGE_D = SAM_BIAS[None, :]**2 + self.minimum_age_thredhold + HH@SAM_G + HH0*SAM_G[0, :][None, :] + HH1*SAM_G[CNT, :] + HH2*SAM_G[CNT+1, :]


    def learnParam(self, islearn_d18O_scale, islearn_d18O_shift, islearn_std_param):
        if islearn_std_param == True:
            self.inv_std_param = np.median((self.AGE[-1, :]-self.AGE[0, :])/(self.depth[-1]-self.depth[0]))

        if (islearn_d18O_shift == True) or (islearn_d18O_scale == True):
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            gamma = 1e-3

            AGE = np.asarray(self.AGE, dtype=float)
            YY = self.d18O.to_numpy()

            if YY.size > 0:
                ID_Y = YY[:, -1].astype(int)
                YY = YY[:, 1:-1]
            else:
                ID_Y = np.array([], dtype=int)

            if ID_Y.size > 0:
                Mw = np.zeros((2,), dtype=float)
                Vw = np.zeros((2,), dtype=float)

                AGE = AGE[ID_Y, :]

                ID = (AGE >= np.min(self.d18O_stack.A)) & (AGE <= np.max(self.d18O_stack.A))

                AGE_valid = AGE[ID]

                YY_expand = np.tile(YY, (1, ID.shape[1]))
                YY_valid = YY_expand[ID]

                MU = interp1d(self.d18O_stack.A, self.d18O_stack.MU, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AGE_valid)
                SIG = interp1d(self.d18O_stack.A, self.d18O_stack.SIG, kind="linear", axis=0, bounds_error=False, fill_value="extrapolate", assume_sorted=True)(AGE_valid)

                CC = float(self.scale_d18O)
                HH = float(self.shift_d18O)

                for r in range(1, 10001):
                    PDEV = np.zeros((2,), dtype=float)

                    QQ = (2.0*self.a_d18O+1.0)*(YY_valid-CC*MU-HH)/(2.0*self.b_d18O*(CC**2)*(SIG**2)+(YY_valid-CC*MU-HH)**2)

                    n_particles = AGE.shape[1]

                    PDEV[0] = np.sum(QQ*(YY_valid-HH)/CC-1.0/CC)/n_particles
                    PDEV[1] = np.sum(QQ)/n_particles

                    Mw = beta1*Mw - (1.0-beta1)*PDEV
                    Vw = beta2*Vw + (1.0-beta2)*(PDEV**2)

                    step = gamma*np.sqrt(1.0-beta2**r)/(1.0-beta1**r)

                    CC = CC - step*Mw[0]/(np.sqrt(Vw[0])+epsilon)
                    HH = HH - step*Mw[1]/(np.sqrt(Vw[1])+epsilon)

                if islearn_d18O_scale == True:
                    self.scale_d18O = CC

                if islearn_d18O_shift == True:
                    self.shift_d18O = HH        

    def saveResults(self, output_folder_path):
        outdir = Path(output_folder_path)
        outdir.mkdir(parents=True, exist_ok=True)


        scalar_file = outdir / "parameters.txt"

        scalar_items = {
            "cir_alpha": self.cir_alpha,
            "cir_beta": self.cir_beta,
            "cir_rho": self.cir_rho,
            "std_param": 1.0/self.inv_std_param,
            "scale_d18O": self.scale_d18O,
            "shift_d18O": self.shift_d18O,
            "a_14C": self.a_14C,
            "b_14C": self.b_14C,
            "a_d18O": self.a_d18O,
            "b_d18O": self.b_d18O,
            "start_depth": self.start_depth,
            "minimum_age_thredhold": self.minimum_age_thredhold,
            "num_samples": self.AGE.shape[1],
            "num_inducing_depths": self.inducing_depths.shape[0],
        }

        with open(scalar_file, "w", encoding="utf-8") as f:
            for key, val in scalar_items.items():
                if val is not None:
                    f.write(f"{key}: {val}\n")
            

        depth = np.asarray(self.depth, dtype=float).reshape(-1)
        age = np.asarray(self.AGE, dtype=float)

        age_median = np.median(age, axis=1)
        age_mean = np.mean(age, axis=1)

        age_q16 = np.quantile(age, 0.16, axis=1)
        age_q84 = np.quantile(age, 0.84, axis=1)

        age_q025 = np.quantile(age, 0.025, axis=1)
        age_q975 = np.quantile(age, 0.975, axis=1)

        summary_df = pd.DataFrame({
            "depth": depth,
            "age_median": age_median,
            "age_mean": age_mean,
            "age_ci68_lower": age_q16,
            "age_ci68_upper": age_q84,
            "age_ci95_lower": age_q025,
            "age_ci95_upper": age_q975,
        })

        summary_file = outdir / "age_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

        age_file = outdir / "age_samples.txt"
        np.savetxt(age_file, age, fmt="%.4f", delimiter="\t")


        depth = np.asarray(self.D, dtype=float).reshape(-1)
        age = np.asarray(self.AGE_D, dtype=float)

        age_median = np.median(age, axis=1)
        age_mean = np.mean(age, axis=1)

        age_q16 = np.quantile(age, 0.16, axis=1)
        age_q84 = np.quantile(age, 0.84, axis=1)

        age_q025 = np.quantile(age, 0.025, axis=1)
        age_q975 = np.quantile(age, 0.975, axis=1)

        summary_df = pd.DataFrame({
            "depth": depth,
            "age_median": age_median,
            "age_mean": age_mean,
            "age_ci68_lower": age_q16,
            "age_ci68_upper": age_q84,
            "age_ci95_lower": age_q025,
            "age_ci95_upper": age_q975,
        })

        summary_file = outdir / "age_inducing_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

        age_file = outdir / "age_inducing_samples.txt"
        np.savetxt(age_file, age, fmt="%.4f", delimiter="\t")


        ACC_RATE = 1.0/np.sum(self.F**2, axis=2)
        AP = np.linspace(0, float(self.inducing_interval), self.F.shape[1], endpoint=False)
        D0 = self.inducing_depths[:, None] - AP[None, :]

        age_file = outdir / "acc_rate_inducing_depths.txt"
        np.savetxt(age_file, D0, fmt="%.4f", delimiter="\t")

        age_file = outdir / "acc_rate_inducing.txt"
        np.savetxt(age_file, ACC_RATE, fmt="%.4f", delimiter="\t")


        YY = self.d18O.to_numpy()
        YC = self.C14.to_numpy()
        YA = self.ETC.to_numpy()

        if YY.size > 0:
            ID_Y = YY[:, -1].astype(int)

            depth = np.asarray(YY[:, 0], dtype=float).reshape(-1)
            age = self.AGE[ID_Y, :]
            
            age_median = np.median(age, axis=1)
            age_mean = np.mean(age, axis=1)

            age_q16 = np.quantile(age, 0.16, axis=1)
            age_q84 = np.quantile(age, 0.84, axis=1)

            age_q025 = np.quantile(age, 0.025, axis=1)
            age_q975 = np.quantile(age, 0.975, axis=1)

            summary_df = pd.DataFrame({
                "depth": depth,
                "d18O": YY[:, 1],
                "age_median": age_median,
                "age_mean": age_mean,
                "age_ci68_lower": age_q16,
                "age_ci68_upper": age_q84,
                "age_ci95_lower": age_q025,
                "age_ci95_upper": age_q975,
            })

            summary_file = outdir / "d18O_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        
        if YC.size > 0:
            ID_C = YC[:, -1].astype(int)

            depth = np.asarray(YC[:, 0], dtype=float).reshape(-1)
            age = self.AGE[ID_C, :]

            age_median = np.median(age, axis=1)
            age_mean = np.mean(age, axis=1)

            age_q16 = np.quantile(age, 0.16, axis=1)
            age_q84 = np.quantile(age, 0.84, axis=1)

            age_q025 = np.quantile(age, 0.025, axis=1)
            age_q975 = np.quantile(age, 0.975, axis=1)

            summary_df = pd.DataFrame({
                "depth": depth,
                "age": YC[:, 1],
                "error": YC[:, 2],
                "dR": YC[:, 3],
                "dSTD": YC[:, 4],
                "cc": YC[:, 5],
                "age_median": age_median,
                "age_mean": age_mean,
                "age_ci68_lower": age_q16,
                "age_ci68_upper": age_q84,
                "age_ci95_lower": age_q025,
                "age_ci95_upper": age_q975,
            })

            summary_file = outdir / "_14C_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

        if YA.size > 0:
            ID_A = YA[:, -1].astype(int)

            depth = np.asarray(YA[:, 0], dtype=float).reshape(-1)
            age = self.AGE[ID_A, :]

            age_median = np.median(age, axis=1)
            age_mean = np.mean(age, axis=1)

            age_q16 = np.quantile(age, 0.16, axis=1)
            age_q84 = np.quantile(age, 0.84, axis=1)

            age_q025 = np.quantile(age, 0.025, axis=1)
            age_q975 = np.quantile(age, 0.975, axis=1)

            summary_df = pd.DataFrame({
                "depth": depth,
                "age": YA[:, 1],
                "unct": YA[:, 2],
                "mode": YA[:, 3],
                "age_median": age_median,
                "age_mean": age_mean,
                "age_ci68_lower": age_q16,
                "age_ci68_upper": age_q84,
                "age_ci95_lower": age_q025,
                "age_ci95_upper": age_q975,
            })

            summary_file = outdir / "ETC_summary.csv"
            summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

