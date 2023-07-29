#ver 2.1 19.10.22

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
        eff:		Stage efficiency (0.0-1.0).
        errors:         Numpy array of errors.
        err:            Numpy array of convergence.
        Kclass:         Object class for calculating K values. Kclass(x).K returns list of Ks.
        
        Functions:
        output():       Print output summary.
        roundoutput():  Print rounded output summary."""
    
    def __init__(self, Oin, Ain, yin, xin, Kclass, eff=1, convergence=1E-3):
        self.Oin = float(Oin)
        self.Ain = float(Ain)
        self.yin = np.array(yin, dtype=float)
        self.xin = np.array(xin, dtype=float)
        a = self.Oin*self.yin/100
        b = self.Ain*self.xin/100
        self.Kclass = Kclass
        eff = float(eff)
        self.eff = eff
        
        #k=0
        c_k=(a+b)/2
        d=a+b-c_k
        delta_to_organic=sum(c_k)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k=self.get_K(d/Aout*100)
        e_k=c_k*Ks_k*Aout-d*Oout*eff
        errors=np.array([e_k/((d+1e-5)*Oout)])

        #k=1
        c_k1=d*Oout/Ks_k/Aout*eff
        conv=[np.ones(len(c_k1))]
        # conv=np.array([(c_k1-c_k)/(c_k1+1e-6)]) # too high
        d=a+b-c_k1
        delta_to_organic=sum(c_k1)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k1=self.get_K(d/Aout*100)
        e_k1=c_k1*Ks_k1*Aout-d*Oout*eff
        errors=np.concatenate((errors, [e_k1/(d*Oout)]), axis=0)

        #k=2
        c_k2=c_k1-e_k1/((e_k1-e_k)/(c_k1-c_k))
        conv=np.concatenate((conv, [(c_k2-c_k1)/c_k2]), axis=0)
        d=a+b-c_k2
        delta_to_organic=sum(c_k2)-sum(a)
        Oout=Oin+delta_to_organic
        Aout=Ain-delta_to_organic
        Ks_k2=self.get_K(d/Aout*100)
        e_k2=c_k2*Ks_k2*Aout-d*Oout*eff
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
            Ks_k2=self.get_K(d/Aout*100)
            e_k2=c_k2*Ks_k2*Aout-d*Oout*eff
            errors=np.concatenate((errors, [e_k2/(d*Oout)]), axis=0)
            i+=1
            if i==200:
                break

        self.errors=errors
        self.conv=conv
        self.runs=i
        self.Oout=Oin+delta_to_organic
        self.Aout=Ain-delta_to_organic
        self.xout=d/Aout*100
        self.yout=c_k2/Oout*100   
        self.K=self.xout/self.yout

    def get_K(self, x):
        return self.Kclass(x).K 

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

class Battery:
    """Attributes:
	
	  """
    def __init__(self, stages_num, Oin, Ain, yin, xin, Kclass, eff=1., convergence=1E-4):
        self.stages_num = stages_num
        self.Oin_list = np.ones(stages_num)*Oin
        self.Ain_list = np.ones(stages_num)*Ain
        yin=np.array(yin, dtype=float)
        xin=np.array(xin, dtype=float)        
        self.yin_list = np.array([yin for i in range(stages_num)])
        self.xin_list = np.array([xin for i in range(stages_num)])
        self.Kclass = Kclass
        self.stages = [[] for i in range(stages_num)] # List of stages
        self.Oout_list = np.zeros(stages_num)
        self.Aout_list = np.zeros(stages_num)
        self.yout_list = np.array([yin for i in range(stages_num)])
        self.xout_list = np.array([xin for i in range(stages_num)])
        self.eff = float(eff)

        # k=0
        for i in range(stages_num):
            self.stages[i] = LLC.Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], Kclass=self.Kclass, eff=self.eff)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout
        self.errors=np.array([(self.xout_list[1]-self.xin_list[0])/self.xout_list[1]])

        # k=1
        for i in range(1, stages_num):
            self.Oin_list[i] = self.Oout_list[i-1]
            self.yin_list[i] = self.yout_list[i-1]
        for i in range(0, stages_num-1):
            self.Ain_list[i] = self.Aout_list[i+1]
            self.xin_list[i] = self.xout_list[i+1]
        for i in range(stages_num):
            self.stages[i] = LLC.Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], Kclass=Kclass, eff=self.eff)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout          
        self.errors=np.concatenate((self.errors,[(self.xout_list[1]-self.xin_list[0])/self.xout_list[1]]), axis=0)

        # k>1
        convergence = convergence
        j=0
        while max(abs(self.errors[-1]))>convergence or max(abs(self.errors[-2]))>convergence:
        # for j in range(20):
            for i in range(1, stages_num):
                self.Oin_list[i] = self.Oout_list[i-1]
                self.yin_list[i] = self.yout_list[i-1]
            for i in range(0, stages_num-1):
                self.Ain_list[i] = self.Aout_list[i+1]
                self.xin_list[i] = self.xout_list[i+1]      
            for i in range(stages_num):
                self.stages[i] = LLC.Stage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], Kclass=Kclass, eff=self.eff)
                self.Oout_list[i] = self.stages[i].Oout
                self.Aout_list[i] = self.stages[i].Aout
                self.yout_list[i] = self.stages[i].yout
                self.xout_list[i] = self.stages[i].xout          
            self.errors=np.concatenate((self.errors,[(self.xout_list[1]-self.xin_list[0])/self.xout_list[1]]), axis=0)
            j+=1
            if j==5000:
                print (j, 'iterations!')
                break

        self.Ks = self.xout_list/self.yout_list
        self.iters = j