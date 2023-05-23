#ver 1.0 29.9.22

import numpy as np

class Stage:
    """Attributes:
        K:              List of distribute ratios. 
        Ks:             List of lists of distribute ratios. Each list is an intercept and a slope for aqueous concentration (0.0-1.0) for each component.
        Oin:            Organic phase mass inlet flow.
        Ain:            Aqueous phase mass inlet flow.
        yin:            List of organic phase inlet concentrations.
        xin:            List of aqueous phase inlet concentrations.
        Oout:           Organic phase mass outlet flow.
        Aout:           Aqueous phase mass outlet flow.
        yout:           List of organic phase outlet concentrations.
        xout:           List of aqueous phase outlet concentrations.
        runs:           Number of iterations.
        errors:         Numpy array of errors.
        err:            Numpy array of convergence.
        
        Functions:
        output():       Print output summary.
        roundoutput():  Print rounded output summary."""

    def __init__(self, Oin, Ain, yin, xin, Ks, convergence=0.001):
        self.Oin = np.array(Oin, dtype=float)
        self.Ain = np.array(Ain, dtype=float)
        self.yin = np.array(yin, dtype=float)
        self.xin = np.array(xin, dtype=float)
        a = self.Oin*self.yin/100
        b = self.Ain*self.xin/100
        self.Ks = np.array(Ks, dtype=float)
        
        #k=0
        c_k=a
        d=a+b-c_k
        delta_to_organic=sum(c_k)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k=self.get_K(c_k)
        # Ks_k=Kvalues(c_k).Ks
        e_k=c_k*Ks_k*Aout-d*Oout
        errors=np.array([e_k/((d+1e-5)*Oout)])

        #k=1
        c_k1=d*Oout/Ks_k/Aout
        conv=[np.ones(len(c_k1))]
        # conv=np.array([(c_k1-c_k)/(c_k1+1e-6)]) # too high
        d=a+b-c_k1
        delta_to_organic=sum(c_k1)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k1=self.get_K(c_k1)
        e_k1=c_k1*Ks_k1*Aout-d*Oout
        errors=np.concatenate((errors, [e_k1/(d*Oout)]), axis=0)

        #k=2
        c_k2=c_k1-e_k1/((e_k1-e_k)/(c_k1-c_k))
        conv=np.concatenate((conv, [(c_k2-c_k1)/c_k2]), axis=0)
        d=a+b-c_k2
        delta_to_organic=sum(c_k2)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k2=self.get_K(c_k2)
        e_k2=c_k2*Ks_k2*Aout-d*Oout
        errors=np.concatenate((errors, [e_k2/(d*Oout)]), axis=0)

        #k>2
        i=0
        while max(abs(conv[-1]))>convergence:
            c_k, e_k=c_k1, e_k1
            c_k1, e_k1=c_k2, e_k2
            c_k2=c_k1-e_k1/((e_k1-e_k)/(c_k1-c_k))
            conv=np.concatenate((conv, [(c_k2-c_k1)/c_k2]), axis=0)
            d=a+b-c_k2
            delta_to_organic=sum(c_k2)-sum(a)
            Oout=Oin+delta_to_organic
            Aout=Ain-delta_to_organic            
            Ks_k2=self.get_K(c_k2)
            e_k2=c_k2*Ks_k2*Aout-d*Oout
            errors=np.concatenate((errors, [e_k2/(d*Oout)]), axis=0)
            i+=1
            if i==200:
                break

        self.errors=errors
        self.conv=conv
        self.runs=i
        self.Aout=Aout
        self.Oout=Oout
        self.xout=d/Aout*100
        self.yout=c_k2/Oout*100
        self.K=self.xout/self.yout

    def get_K(self, x):
        K = [K[0]+c*K[1] for c,K in zip(x,self.Ks)]
        return K 

    def output(self):
        print ("Oout=", self.Oout)
        print ("yout=", self.yout)
        print ("Aout=", self.Aout)
        print ("xout=", self.xout)
        print ("error=", self.errors[-1])
        print ("convergence=", self.conv[-1])
        print ("K=", self.K)

    def roundoutput(self):
        print ("Oout=", round(self.Oout,1))
        print ("yout=", [round(num,3) for num in self.yout])
        print ("Aout=", round(self.Aout,2))
        print ("xout=", [round(num,3) for num in self.xout])
        print ("error=", [round(num,7) for num in self.errors[-1]])
        print ("convergence=", [round(num,7) for num in self.conv[-1]])
        print ("K=", [round(num,5) for num in self.K])

class Kvalues:
    def __init__(self, c_list):
        k_solvent=[0.0003,0] #intercept, slope
        k_water=[18.6,0]
        k_phosphate=[3.25,0]
        k_sulfate=[1.04,0]
        K_list= [k_solvent, k_water, k_phosphate, k_sulfate]
        self.Ks = [a[0] for a in K_list]
        # self.Ks = [a[0]+a[1]*b for a,b in zip(K_list,c_list)]