#ver 4.1 27.02.23
#EQUIL, Entrainments

import numpy as np
import pandas as pd

class Stage:
    """Attributes:
        Oin:                  Organic phase mass inlet flow.
        Ain:                  Aqueous phase mass inlet flow.
        yin:                  List of organic phase inlet concentrations. [%water_org, %H3PO4_org, ...]
        xin:                  List of aqueous phase inlet concentrations. [%solvent_aq, %H3PO4_aq, ...]
        Oout:                 Organic phase mass outlet flow.
        Aout:                 Aqueous phase mass outlet flow.
        yout_tag:             List of organic phase equilibrium concentrations.
        yout:                 List of organic phase outlet concentrations.
        xout:                 List of aqueous phase outlet concentrations.
        runs:                 Number of iterations.
        eff:                  Stage efficiency (0.0-1.0).
        errors:               Numpy array of errors.
        conv:                 Numpy array of convergence.
        EQUIL:                Object class for calculating equilibrium concentrations.
        entrainment_perc:     Percentage of entrainments [org_in, aq_in, org_out, aq_out]. Additionally to the main stream: Oin 100 kg with 5% entrainment means 100 kg organic and 5 kg aqueous -- 105 overall.
        entrainment_comp_in:  Composition of inlet entrainments in [aqueous phase in Oin, organic phase in Ain] --> [[%solvent_aq, %H3PO4_aq, ...], [%water_org, %H3PO4_org, ...], ]. Outlet compositions are calculated: yout, xout.
        entrainment_out:      List of entrainment out data: [entrainment_perc[2:4], yout, xout].
        
        Functions:
        output():       Print output summary.
        roundoutput():  Print rounded output summary."""
    
    def __init__(self, Oin, Ain, yin, xin, EQUIL, eff=1., entrainment_perc=0, entrainment_comp_in=0, convergence=1E-3):
        
        Oin_clear = float(Oin)
        Ain_clear = float(Ain)
        self.yin = np.array(yin, dtype=float) # [%water_org, %H3PO4_org, ...]
        self.xin = np.array(xin, dtype=float) # [%solvent_aq, %H3PO4_aq, ...]
        self.EQUIL = EQUIL
        self.eff = eff
        self.entrainment_perc = entrainment_perc
        self.entrainment_comp_in = entrainment_comp_in
        self.check_inputs()
        Oin = Oin_clear + Ain_clear*self.entrainment_perc[1]/100.
        Ain = Ain_clear + Oin_clear*self.entrainment_perc[0]/100.
        a = Oin_clear*self.yin/100. + Ain_clear*self.entrainment_perc[1]/100*self.entrainment_comp_in[1]/100. # [%water_org, %H3PO4_org, %SO4_org, ...]
        b = Ain_clear*self.xin/100. + Oin_clear*self.entrainment_perc[0]/100*self.entrainment_comp_in[0]/100. # [%solvent_aq, %H3PO4_aq, ...]

        #k=0
        c_k=(a+b)/2
        d=a+b-c_k
        alfa=0 # d[0]=alfa*Aout
        A=np.array([[1,1,0],[-1,0,1],[1,0,-alfa]]) # d[0],Oout,Aout
        B=np.array([[Oin+sum(c_k)-sum(a)+b[0]],
                    [Ain-sum(b)+a[0]-c_k[0]+sum(d)-d[0]],
                    [0]])
        C=np.dot(np.linalg.inv(A),B)
        d[0],Oout,Aout = C[0,0],C[1,0],C[2,0]
        xout=d/Aout*100 #
        out_data = self.EQUIL([xout])
        xout=out_data.xout[0]
        yout_tag=out_data.yout[0]
        yout=self.yin+self.eff*(yout_tag-self.yin)
        d=Aout*xout/100
        alfa=d[0]/Aout
        e_k=c_k-Oout*yout/100
        errors=np.array([e_k/(Oout*yout/100+1E-7)]) #

        #k=1
        c_k1=Oout*yout/100
        conv=[np.ones(len(c_k1))]
        d=a+b-c_k1
        A=np.array([[1,1,0],[-1,0,1],[1,0,-alfa]]) # d[0],Oout,Aout
        B=np.array([[Oin+sum(c_k1)-sum(a)+b[0]],
                    [Ain-sum(b)+a[0]-c_k1[0]+sum(d)-d[0]],
                    [0]])
        C=np.dot(np.linalg.inv(A),B)
        d[0],Oout,Aout = C[0,0],C[1,0],C[2,0]
        xout=d/Aout*100
        out_data = self.EQUIL([xout])
        xout=out_data.xout[0]
        yout_tag=out_data.yout[0]
        yout=self.yin+self.eff*(yout_tag-self.yin)
        d=Aout*xout/100
        alfa=d[0]/Aout
        e_k1=c_k1-Oout*yout/100
        errors=np.concatenate((errors, [e_k1/(Oout*yout/100+1E-7)]), axis=0)

        #k=2
        c_k2=c_k1-e_k1/((e_k1-e_k+1E-7)/(c_k1-c_k+1E-7))
        conv=np.concatenate((conv, [(c_k2-c_k1)/(c_k2+1E-7)]), axis=0)
        d=a+b-c_k2
        A=np.array([[1,1,0],[-1,0,1],[1,0,-alfa]]) # d[0],Oout,Aout
        B=np.array([[Oin+sum(c_k2)-sum(a)+b[0]],
                    [Ain-sum(b)+a[0]-c_k2[0]+sum(d)-d[0]],
                    [0]])
        C=np.dot(np.linalg.inv(A),B)
        d[0],Oout,Aout = C[0,0],C[1,0],C[2,0]
        xout=d/Aout*100
        out_data = self.EQUIL([xout])
        xout=out_data.xout[0]
        yout_tag=out_data.yout[0]
        yout=self.yin+self.eff*(yout_tag-self.yin)
        d=Aout*xout/100
        alfa=d[0]/Aout
        e_k2=c_k2-Oout*yout/100
        errors=np.concatenate((errors, [e_k2/(Oout*yout/100+1E-7)]), axis=0)

        #k>2
        i=0
        while max(abs(conv[-1]))>convergence:
            c_k, e_k=c_k1, e_k1
            c_k1, e_k1=c_k2, e_k2
            c_k2=c_k1-e_k1/((e_k1-e_k+1E-7)/(c_k1-c_k+1E-7))
            conv=np.concatenate((conv, [(c_k2-c_k1)/(c_k2+1E-7)]), axis=0)
            d=a+b-c_k2
            A=np.array([[1,1,0],[-1,0,1],[1,0,-alfa]]) # d[0],Oout,Aout
            B=np.array([[Oin+sum(c_k2)-sum(a)+b[0]],
                        [Ain-sum(b)+a[0]-c_k2[0]+sum(d)-d[0]],
                        [0]])
            C=np.dot(np.linalg.inv(A),B)
            d[0],Oout,Aout = C[0,0],C[1,0],C[2,0]
            Oout=Oin+sum(c_k2)-sum(a)+b[0]-d[0]
            Aout=Ain+sum(d)-sum(b)+a[0]-c_k2[0]
            xout=d/Aout*100
            out_data = self.EQUIL([xout])
            xout=out_data.xout[0]
            yout_tag=out_data.yout[0]
            yout=self.yin+self.eff*(yout_tag-self.yin)
            d=Aout*xout/100
            alfa=d[0]/Aout
            e_k2=c_k2-Oout*yout/100
            errors=np.concatenate((errors, [e_k2/(Oout*yout/100+1E-7)]), axis=0)
            i+=1
            if i==200:
                break

        self.errors=errors
        self.conv=conv
        self.runs=i
        self.Oin=Oin
        self.Ain=Ain
        Oout_clean=Oout
        Aout_clean=Aout
        A=np.array([[1,self.entrainment_perc[3]/100],[self.entrainment_perc[2]/100,1]])
        B=np.array([[Oout_clean],[Aout_clean]])
        C=np.dot(np.linalg.inv(A),B)
        self.Oout, self.Aout = C[0,0], C[1,0]
        self.xout=xout
        self.yout_tag=yout_tag
        self.yout=yout 
        self.entrainment_out=[self.entrainment_perc[2:4], self.yout, self.xout]
        # self.K=self.xout/self.yout

    def check_inputs(self):
        if np.isscalar(self.eff):
            self.eff = np.ones(len(self.yin)) * self.eff
            self.eff[0] = 1. # water and solvent
        else:
            if len(self.eff) != len(self.yin):
                print ('Length of efficiency array is not correct.')
            self.eff = np.ones(len(self.yin))* self.eff
            self.eff[0] = 1.

        if np.isscalar(self.entrainment_perc):
            self.entrainment_perc = np.ones(4) * self.entrainment_perc
        else:
            if len(self.entrainment_perc) != 4:
                print ('List of entrainment percents should consist of 4 elements [org_in, aq_in, org_out, aq_out] or single number.')
            self.entrainment_perc = np.ones(4) * self.entrainment_perc

        if np.isscalar(self.entrainment_comp_in):
            self.entrainment_comp_in = np.ones([2,len(self.yin)]) * self.entrainment_comp_in
        else:
            if len(self.entrainment_comp_in[0]) != len(self.yin):
                print ('Array of entrainment compositions is not correct.')
            self.entrainment_comp_in = np.ones(len(self.yin)) * self.entrainment_comp_in

    def output(self):
        print ("Oout=", self.Oout)
        print ("yout=", self.yout)
        print ("yout_tag=", self.yout_tag, " %water_org_equilib, %solubles_equilib")
        print ("Aout=", self.Aout)
        print ("xout=", self.xout)
        print ("equilibrium efficiency=", self.eff)
        print ("error=", self.errors[-1])
        print ("convergence=", self.conv[-1])
        # print ("K=", self.K)

    def roundoutput(self):
        print ("Oout=", round(self.Oout,1))
        print ("yout=", [round(num,3) for num in self.yout], " %water_org, %solubles")
        print ("yout_tag=", [round(num,3) for num in self.yout_tag], " %water_org_equilib, %solubles_equilib")
        print ("Aout=", round(self.Aout,2))
        print ("xout=", [round(num,3) for num in self.xout], " %solvent_aq, %solubles")
        print ("equilibrium efficiency=", self.eff)
        print ("error=", [round(num,7) for num in self.errors[-1]])
        print ("convergence=", [round(num,7) for num in self.conv[-1]])
        # print ("K=", [round(num,5) for num in self.K])

class Battery:
    """Attributes:
        stages_num:                       Number of stages.
        stages:                           Array of Stage objects (0 -- stage_num-1). Organic inlet to stages[0] -- Stage 1.
        Oin/Ain/Oin/Oout:                 Organic/aqueous phase mass inlet/outlet flow.
        Oin/Ain/Oin/Oout_list:            List of organic/aqueous phase mass inlet/outlet flows for stages.
        yin/xin/yout_tag/yout/xout_list:  List of lists of organic/aqueous phase inlet/outlet concentrations: [stage, component]
        runs:                             Number of iterations.
        eff_list:                         List of stage efficiencies (0.0-1.0).
        errors:                           Numpy array of errors.
        EQUIL:                            Object class for calculating equilibrium concentrations.
        K:                                List of distribute ratios for solubles. %aq/%org
        entrainment_perc_in:              Entrainment percent for inlets [org_in, aq_in].
        entrainment_perc_out:             Entrainment percent data should consist of 2D array (stages_num, 2)  or 1D array [org_out, aq_out] or single number.
        entrainment_perc_list:            Entrainment percent data arranged for stages.
        entrainment_comp_in:              Composition of inlet entrainments in [aqueous phase in Oin, organic phase in Ain] --> [[%solvent_aq, %H3PO4_aq, ...], [%water_org, %H3PO4_org, ...]]. Outlet compositions are calculated: yout, xout.
        entrainment_comp_list:            List of entrainment compositions [[%solvent_aq, %H3PO4_aq, ...],[%water_org, %H3PO4_org, ...], ... stage_num].
        entrainment_comp_flat:            List of entrainment compositions arranged for presentation in table.
        """

    def __init__(self, stages_num, Oin, Ain, yin, xin, EQUIL, eff=1., entrainment_perc_in=0, entrainment_comp_in=0, entrainment_perc_out=0, convergence=1E-4):
        self.stages_num = stages_num
        self.Oin_list = np.ones(stages_num)*Oin
        self.Ain_list = np.ones(stages_num)*Ain
        yin=np.array(yin, dtype=float) # [%water_org, %H3PO4_org, ...]
        xin=np.array(xin, dtype=float) # [%solvent_aq, %H3PO4_aq, ...]
        self.yin_list = np.array([yin for i in range(stages_num)])
        self.xin_list = np.array([xin for i in range(stages_num)])
        self.EQUIL = EQUIL
        self.stages = [[] for i in range(stages_num)] # List of stages
        self.Oout_list = np.zeros(stages_num)
        self.Aout_list = np.zeros(stages_num)
        self.yout_tag_list = np.array([yin for i in range(stages_num)])
        self.yout_list = np.array([yin for i in range(stages_num)])
        self.xout_list = np.array([xin for i in range(stages_num)])
        self.eff = eff
        self.entrainment_perc_in = entrainment_perc_in
        self.entrainment_comp_in = entrainment_comp_in
        self.entrainment_perc_out = entrainment_perc_out
        self.check_inputs()
        entrainment_comp_list = np.array([self.entrainment_comp_in for i in range(stages_num)])*1.

        # k=0
        for i in range(stages_num):
            self.stages[i] = Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff[i], 
                                   entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_tag_list[i] = self.stages[i].yout_tag
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout
        self.errors=np.array([(self.xout_list[1]-self.xin_list[0])/(self.xout_list[1]+1E-7)])

        # k=1
        for i in range(1, stages_num):
            self.Oin_list[i] = self.Oout_list[i-1]
            self.yin_list[i] = self.yout_list[i-1]
            entrainment_comp_list[i-1][1] = self.yout_list[i]
        for i in range(0, stages_num-1):
            self.Ain_list[i] = self.Aout_list[i+1]
            self.xin_list[i] = self.xout_list[i+1]
            entrainment_comp_list[i+1][0] = self.xout_list[i]
        for i in range(stages_num):
            self.stages[i] = Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff[i], 
                                   entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_tag_list[i] = self.stages[i].yout_tag
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout          
        self.errors=np.concatenate((self.errors,[(self.xout_list[1]-self.xin_list[0])/(self.xout_list[1]+1E-5)]), axis=0)

        # k>1
        convergence = convergence
        j=0
        while max(abs(self.errors[-1]))>convergence or max(abs(self.errors[-2]))>convergence:
        # for j in range(20):
            for i in range(1, stages_num):
                self.Oin_list[i] = self.Oout_list[i-1]
                self.yin_list[i] = self.yout_list[i-1]
                entrainment_comp_list[i-1][1] = self.yout_list[i]
            for i in range(0, stages_num-1):
                self.Ain_list[i] = self.Aout_list[i+1]
                self.xin_list[i] = self.xout_list[i+1] 
                entrainment_comp_list[i+1][0] = self.xout_list[i]  
            for i in range(stages_num):
                self.stages[i] = Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff[i], 
                                       entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
                self.Oout_list[i] = self.stages[i].Oout
                self.Aout_list[i] = self.stages[i].Aout
                self.yout_tag_list[i] = self.stages[i].yout_tag
                self.yout_list[i] = self.stages[i].yout
                self.xout_list[i] = self.stages[i].xout          
            self.errors=np.concatenate((self.errors,[(self.xout_list[1]-self.xin_list[0])/(self.xout_list[1]+1E-7)]), axis=0)
            j+=1
            if j==5000:
                print (j, 'iterations!')
                break

        self.K = self.xout_list/(self.yout_tag_list+1E-7)
        self.runs = j
        self.Oin, self.Ain, self.yin, self.xin = Oin, Ain, yin, xin
        self.Oout, self.Aout, self.yout_tag, self.yout, self.xout, = self.Oout_list[-1], self.Aout_list[0], self.yout_tag_list[-1], self.yout_list[-1], self.xout_list[0], 
        self.entrainment_comp_list = entrainment_comp_list
        entrainment_comp_flat = []
        for stage in range(stages_num):
            entrainment_comp_flat.append([i for item in entrainment_comp_list[stage] for i in item])
        self.entrainment_comp_flat = np.array(entrainment_comp_flat)

    def check_inputs(self):
        # --------
        eff, stages_num, yin_len = self.eff, self.stages_num, len(self.yin_list[0])
        if np.isscalar(eff):                                             # scalar
            eff_out = np.ones([stages_num, yin_len]) * eff
        else:
            eff = np.array(eff)
            eff_out = np.ones([stages_num, yin_len])
            if np.isscalar(eff[0]):                                      # vector for elements [water, P2O5, ...] same for all stages
                a = np.array(list(map(lambda x:eff[:yin_len], eff_out))) # 2D array (stage_num, yin_len) with rows of eff
                eff_out[:a.shape[0], :a.shape[1]]=a                      # for the case if eff is shorter than yin_len
            else:                   
                if eff.shape[1] == 1:                                    # vector for stages same for all elements  
                    b = eff.T[0]
                    a = np.array(list(map(lambda x:b[:stages_num], eff_out.T))) 
                    eff_out.T[:a.shape[0], :a.shape[1]]=a 
                if eff.shape[1] > 1:                                     # list of lists
                    eff_out[:eff.shape[0], :eff.shape[1]]=eff            # for the case if eff is shorter than yin_len
        self.eff = eff_out
        self.eff[:,0]=1
        # ----------
        epi = self.entrainment_perc_in
        if np.isscalar(epi):
            epi = np.ones(2) * epi
        else:
            epi = np.array(epi, dtype=float)

        epo = self.entrainment_perc_out
        if np.isscalar(epo):
            epo_out = np.ones([stages_num, 2]) * epo
        else:
            epo = np.array(epo, dtype=float)
            epo_out = np.ones([stages_num, 2])
            if np.isscalar(epo[0]): # same [org_out, aq_out] for all stages 
                if len(epo) != 2:
                    print ('Entrainment percent data should consist of 2D array (stages_num, 2)  or 1D array [org_out, aq_out] or single number.')
                epo_out = epo_out * epo
            else:
                if epo.shape != (stages_num,2):
                    print ('Entrainment percent data should consist of 2D array (stages_num, 2)  or 1D array [org_out, aq_out] or single number.')
                epo_out = epo
        
        ep1 = np.concatenate([[epi], epo_out, [epi]], axis=0)
        ep2 = [[ep1[i,0],ep1[i+2,1]] for i in range(stages_num)]
        self.entrainment_perc_list = np.concatenate([ep2,epo_out], axis=1)
        # ------------
        if np.isscalar(self.entrainment_comp_in):
            self.entrainment_comp_in = np.ones([2, len(self.yin_list[0])]) * self.entrainment_comp_in
        else:
            if len(self.entrainment_comp_in[0]) != len(self.yin_list[0]):
                print ('Array of entrainment compositions is not correct.')

class BatteryTable:
    """
    """
    def __init__(self, batt):
        len1=len(batt.xin_list[0])
        out=np.concatenate((
            [batt.Oin_list], 
            batt.yin_list.T,             
            [batt.Ain_list], 
            batt.xin_list.T, 
            [batt.Oout_list], 
            batt.yout_list.T, 
            [batt.Aout_list], 
            batt.xout_list.T, 
            ),axis=0)
        Name = (
            ['Org in, ton/hr']+
            ['y'+str(a+1)+' in, %' for a in range(len1)]+            
            ['Aq in, ton/hr']+
            ['x'+str(a+1)+' in, %' for a in range(len1)]+
            ['Org out, ton/hr']+
            ['y'+str(a+1)+' out, %' for a in range(len1)]+        
            ['Aq out, ton/hr']+
            ['x'+str(a+1)+' out, %' for a in range(len1)]
            )
        columns = ['Stage '+str(a) for a in range(1,batt.stages_num+1)]
        self.data=pd.DataFrame(out, columns=columns, index=Name)
        display (self.data)

class BatteryTableFull:
    """
    """
    def __init__(self, batt):
        len1=len(batt.xin_list[0])
        out=np.concatenate((
            [batt.Oin_list], 
            batt.yin_list.T,             
            [batt.Ain_list], 
            batt.xin_list.T, 
            [batt.Oout_list], 
            batt.yout_list.T, 
            [batt.Aout_list], 
            batt.xout_list.T, 
            batt.K.T[1:],
            batt.entrainment_perc_list.T,
            np.concatenate([[batt.Oin_list],[batt.Ain_list],[batt.Oout_list],[batt.Aout_list]], axis=0)*batt.entrainment_perc_list.T/100,
            batt.entrainment_comp_flat.T,
            ),axis=0)
        Name = (
            ['Org in, ton/hr']+
            ['y'+str(a+1)+' in, %' for a in range(len1)]+            
            ['Aq in, ton/hr']+
            ['x'+str(a+1)+' in, %' for a in range(len1)]+
            ['Org out, ton/hr']+
            ['y'+str(a+1)+' out, %' for a in range(len1)]+        
            ['Aq out, ton/hr']+
            ['x'+str(a+1)+' out, %' for a in range(len1)]+
            ['K'+str(a+1)+' [aq/org]' for a in range(1,len1)]+
            ['aq in org_in, %']+['org in aq_in, %']+['aq in org_out, %']+['org in aq_out, %']+
            ['aq in org_in, ton/hr']+['org in aq_in, ton/hr']+['aq in org_out, ton/hr']+['org in aq_out, ton/hr']+
            ['aq contrain in Oin - x'+str(a+1)+'%' for a in range(len1)]+
            ['org contrain in Ain - y'+str(a+1)+'%' for a in range(len1)]
            )
        columns = ['Stage '+str(a) for a in range(1,batt.stages_num+1)]
        self.data=pd.DataFrame(out, columns=columns, index=Name)
        display (self.data)