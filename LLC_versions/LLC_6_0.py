#ver 6.0 5.7.26
#NewStage
#CounterCurrentBattery

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

class Stage:
    """Ступень жидкостной экстракции с учетом энтрейнмента и эффективности."""

    def __init__(self, Oin, Ain, yin, xin, EQUIL, eff=1.0, 
                 entrainment_perc=0, entrainment_comp_in=0, 
                 convergence=1E-4, solver_type='custom'):
        
        # Исходные расходы "чистых" фаз
        self.Oin_clear = float(Oin)
        self.Ain_clear = float(Ain)
        self.yin = np.array(yin, dtype=float)
        self.xin = np.array(xin, dtype=float)
        self.EQUIL = EQUIL
        self.convergence_threshold = convergence

        # 1. Валидация и расширение параметров (eff, entrainment)
        self._check_and_expand_inputs(eff, entrainment_perc, entrainment_comp_in)

        # 2. Расчет входящих потоков с учетом энтрейнмента (физические массы компонентов)
        # mass_org_in/mass_aq_in — это векторы масс каждого компонента [кг] во входящих потоках
        ent = self.entrainment_perc / 100.
        self.Oin_total = self.Oin_clear + self.Ain_clear * ent[1]
        self.Ain_total = self.Ain_clear + self.Oin_clear * ent[0]

        self.mass_org_in = (self.Oin_clear * self.yin / 100. + 
                            self.Ain_clear * ent[1] * self.entrainment_comp_in[1] / 100.)
        self.mass_aq_in = (self.Ain_clear * self.xin / 100. + 
                           self.Oin_clear * ent[0] * self.entrainment_comp_in[0] / 100.)
        
        self.total_mass_vector = self.mass_org_in + self.mass_aq_in

        # 3. Выбор и запуск солвера
        if solver_type == 'custom':
            self._solve_custom()
        else:
            self._solve_scipy()

        # 4. Коррекция выходящих потоков на энтрейнмент (решение СЛАУ для чистых фаз)
        self._final_correction()

    def _check_and_expand_inputs(self, eff, ent_p, ent_c):
        n = len(self.yin)
        # Эффективность
        self.eff = np.ones(n) * eff if np.isscalar(eff) else np.array(eff)
        self.eff[0] = 1.0 # Вода и растворитель
        
        # % энтрейнмента [org_in, aq_in, org_out, aq_out]
        self.entrainment_perc = np.ones(4) * ent_p if np.isscalar(ent_p) else np.array(ent_p)
        
        # Составы энтрейнмента
        if np.isscalar(ent_c):
            self.entrainment_comp_in = np.ones((2, n)) * ent_c
        else:
            self.entrainment_comp_in = np.array(ent_c)

    def _iteration_step(self, guess_mass_org_out, alfa, force_explicit=False):
        """Один шаг пересчета баланса и равновесия."""
        # mass_aq_out — вектор масс компонентов в водной фазе (баланс)
        mass_aq_out = self.total_mass_vector - guess_mass_org_out

        # Решаем СЛАУ для определения общего расхода фаз и массы первого компонента (воды/растворителя)
        # Уравнения:
        # 1) Oout_total + mass_aq_out[0] = B0 (баланс масс)
        # 2) Aout_total - mass_aq_out[0] = B1 (баланс масс)
        # 3) mass_aq_out[0] = alfa * Aout_total (равновесное допущение для воды)
        A_matrix = np.array([[1.0, 1.0, 0.0],
                             [-1.0, 0.0, 1.0],
                             [1.0, 0.0, -alfa]])
        
        B0 = self.Oin_total + np.sum(guess_mass_org_out) - np.sum(self.mass_org_in) + self.mass_aq_in[0]
        B1 = self.Ain_total - np.sum(self.mass_aq_in) + self.mass_org_in[0] - guess_mass_org_out[0] + np.sum(mass_aq_out) - mass_aq_out[0]
        
        sol = np.linalg.solve(A_matrix, [B0, B1, 0.0])
        mass_aq_out[0], Oout_total, Aout_total = sol[0], sol[1], sol[2]

        if force_explicit:
            Aout_total = abs(self.Ain_total + np.sum(mass_aq_out) - np.sum(self.mass_aq_in) + self.mass_org_in[0] - guess_mass_org_out[0])

        xout_perc = mass_aq_out / Aout_total * 100.0
        
        # Равновесный расчет
        equil_data = self.EQUIL([xout_perc])
        xout_perc = equil_data.xout[0]
        yout_equil_perc = equil_data.yout[0]
        
        # Учет эффективности ступени
        yout_actual_perc = self.yin + self.eff * (yout_equil_perc - self.yin)
        
        # Обновляем альфу и ошибку для следующего шага
        new_mass_aq_out = Aout_total * xout_perc / 100.0
        new_alfa = new_mass_aq_out[0] / Aout_total
        
        # Ошибка баланса по органической фазе
        error = guess_mass_org_out - Oout_total * yout_actual_perc / 100.0
        
        return Oout_total, Aout_total, xout_perc, yout_equil_perc, yout_actual_perc, new_alfa, error

    def _solve_custom(self):
        """Твой оригинальный алгоритм секущих."""
        # k=0: начальное приближение (50/50 распределение)
        curr_guess = self.total_mass_vector / 2.0
        alfa = 0.0
        Oout, Aout, xout, y_eq, y_act, alfa, err_k0 = self._iteration_step(curr_guess, alfa)
        
        # k=1: уточнение
        prev_guess, prev_error = curr_guess, err_k0
        curr_guess = Oout * y_act / 100.0
        Oout, Aout, xout, y_eq, y_act, alfa, err_k1 = self._iteration_step(curr_guess, alfa)
        
        self.errors = [err_k0 / (Oout * y_act / 100.0 + 1E-7)]
        self.conv = [np.ones(len(self.yin))]

        # Iterations (k >= 2)
        for i in range(200):
            # Метод секущих для обновления вектора масс органики
            denom = (err_k1 - prev_error + 1E-7) / (curr_guess - prev_guess + 1E-7)
            next_guess = curr_guess - err_k1 / denom
            
            self.conv = np.vstack([self.conv, (next_guess - curr_guess) / (next_guess + 1E-7)])
            
            # Шаг расчета
            prev_guess, prev_error = curr_guess, err_k1
            curr_guess = next_guess
            Oout, Aout, xout, y_eq, y_act, alfa, err_k1 = self._iteration_step(curr_guess, alfa, force_explicit=True)
            
            self.errors = np.vstack([self.errors, err_k1 / (Oout * y_act / 100.0 + 1E-7)])
            self.runs = i + 1
            
            if np.max(np.abs(self.conv[-1])) < self.convergence_threshold:
                break
        
        self._save_results(Oout, Aout, xout, y_eq, y_act)

    def _solve_scipy(self):
        """Альтернативный мощный солвер."""
        def objective(mass_org_vec):
            # Scipy минимизирует эту функцию (ошибку баланса)
            _, _, _, _, _, _, err = self._iteration_step(mass_org_vec, 0.0, True)
            return err

        initial_guess = self.total_mass_vector / 2.0
        sol = fsolve(objective, initial_guess)
        
        # Финальный прогон с найденным решением для сохранения переменных
        res = self._iteration_step(sol, 0.0, True)
        self.runs = -1 # fsolve не возвращает число итераций так просто
        self.errors = [np.zeros(len(self.yin))]
        self.conv = [np.zeros(len(self.yin))]
        self._save_results(*res[:5])

    def _save_results(self, Oout_total, Aout_total, xout, yout_tag, yout):
        self._Oout_raw = Oout_total
        self._Aout_raw = Aout_total
        self.xout = xout
        self.yout_tag = yout_tag
        self.yout = yout

    def _final_correction(self):
        # Коррекция на выходящий энтрейнмент (чистые фазы)
        ent = self.entrainment_perc[2:4] / 100.0
        M = np.array([[1.0, ent[1]], [ent[0], 1.0]])
        B = np.array([self._Oout_raw, self._Aout_raw])
        clean_flows = np.linalg.solve(M, B)
        self.Oout, self.Aout = clean_flows[0], clean_flows[1]
        self.entrainment_out = [self.entrainment_perc[2:4], self.yout, self.xout]

    def output(self):
        print(f"Oout (clean) = {self.Oout:.2f}")
        print(f"yout = {self.yout}")
        print(f"Aout (clean) = {self.Aout:.2f}")
        print(f"xout = {self.xout}")
        print(f"Runs: {self.runs}")

    def roundoutput(self):
        print(f"Oout = {round(self.Oout, 1)}")
        print(f"yout = {[round(n, 3) for n in self.yout]}")
        print(f"Aout = {round(self.Aout, 2)}")
        print(f"xout = {[round(n, 3) for n in self.xout]}")

###################################################################
###################################################################


class OldStage:
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
    
    def __init__(self, Oin, Ain, yin, xin, EQUIL, eff=1., entrainment_perc=0, entrainment_comp_in=0, convergence=1E-4):
        
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
            Aout=abs(Aout)
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


###################################################################
###################################################################


class CounterCurrentBattery:
    """
    A class for simulating a multi-stage counter-current battery extraction process.

    Attributes:
        stages_num (int): Number of stages in the battery.
        stages (list): A list representing stages. Currently, it is initialized as an empty list and not actively used in the calculations.
        Oin_list (numpy.ndarray): List of organic phase inlet flow rates for each stage.
        Ain_list (numpy.ndarray): List of aqueous phase inlet flow rates for each stage.
        Oout_list (numpy.ndarray): List of organic phase outlet flow rates for each stage.
        Aout_list (numpy.ndarray): List of aqueous phase outlet flow rates for each stage.
        yin_list (numpy.ndarray): List of organic phase inlet concentrations for each stage. Shape: (stages_num, num_components_organic).
        xin_list (numpy.ndarray): List of aqueous phase inlet concentrations for each stage. Shape: (stages_num, num_components_aqueous).
        yout_tag_list (numpy.ndarray): List of organic phase outlet concentrations (tagged for equilibrium calculation) for each stage. Shape: (stages_num, num_components_organic).
        yout_list (numpy.ndarray): List of organic phase outlet concentrations for each stage. Shape: (stages_num, num_components_organic).
        xout_list (numpy.ndarray): List of aqueous phase outlet concentrations for each stage. Shape: (stages_num, num_components_aqueous).
        runs (int): Number of iterations performed for battery convergence.
        eff_list (numpy.ndarray): List of stage efficiencies for each component in each stage. Shape: (stages_num, num_components_organic).
        EQUIL (object): Object for calculating equilibrium concentrations. Must have a method to calculate equilibrium.
        K (numpy.ndarray): List of distribution ratios (K values) for each component at the final stage, aqueous phase to organic phase ratio (aq/org). Shape: (stages_num, num_components_organic).
        entrainment_perc_in (float or array-like): Entrainment percentage for inlet streams [organic in Oin, aqueous in Ain].
        entrainment_perc_out (float or array-like): Entrainment percentage for outlet streams. Can be a scalar (same for all stages and outlets), a 1D array [org_out, aq_out] (same for all stages), or a 2D array (stages_num, 2).
        entrainment_perc_list (numpy.ndarray): Entrainment percentage data structured as a list for each stage. Shape: (stages_num, 4).
        entrainment_comp_in (array-like): Composition of entrainment in the inlet streams. Specified as a list of lists: [[components in aqueous entrainment of Oin], [components in organic entrainment of Ain]].
        entrainment_comp_list (numpy.ndarray): Entrainment compositions for each stage. Shape: (stages_num, 2, num_components).
        entrainment_comp_flat (numpy.ndarray): Flattened list of entrainment compositions, suitable for tabular representation. Shape: (stages_num, 2 * num_components).
    """

    def __init__(self, stages_num, Oin, Ain, yin, xin, EQUIL, method='successive', damping=1, eff=1., max_iter=100, entrainment_perc_in=0, entrainment_comp_in=0, entrainment_perc_out=0, battery_convergence=1e-3, stage_convergence=1E-2, convergence=None, **kwargs):
        """
        Initializes the BatteryConv class.

        Parameters:
            stages_num (int): Number of stages in the battery.
            Oin (float): Organic phase inlet flow rate to the first stage.
            Ain (float): Aqueous phase inlet flow rate to the last stage.
            yin (list): Organic phase inlet concentrations to the first stage. e.g., [%water_org, %H3PO4_org, ...].
            xin (list): Aqueous phase inlet concentrations to the last stage. e.g., [%solvent_aq, %H3PO4_aq, ...].
            EQUIL (object): Equilibrium calculation class instance.
            method (str, optional): Iteration method for solving the battery ('successive', 'secant', 'damp_secant', 'newton', 'anderson', 'broyden'). Defaults to 'successive'.
            damping (float, optional): Damping factor for iterative methods (0.0 to 1.0). Defaults to 1.
            eff (float or array-like, optional): Stage efficiency (0.0 to 1.0). Can be a scalar (same for all stages and components), a vector for elements (same for all stages), a vector for stages (same for all elements) or a 2D array (stage and elements). Defaults to 1.0.
            max_iter (int, optional): Maximum number of iterations for battery convergence. Defaults to 100.
            entrainment_perc_in (float or array-like, optional): Entrainment percentage in inlet streams [organic_in, aq_in]. Defaults to 0.
            entrainment_comp_in (array-like, optional): Composition of inlet entrainments [[%solvent_aq, %H3PO4_aq, ...], [%water_org, %H3PO4_org, ...]]. Defaults to 0.
            entrainment_perc_out (float or array-like, optional): Entrainment percentage in outlet streams. Defaults to 0.
            battery_convergence (float, optional): Convergence tolerance for battery iterations. Defaults to 1e-3.
            stage_convergence (float, optional): Convergence tolerance for individual stage iterations within Stage class. Defaults to 1E-2.
            convergence (float, optional):  Deprecated parameter for backward compatibility; if provided, it sets both battery_convergence and stage_convergence. Defaults to None.
        """
        self.stages_num = stages_num
        self.EQUIL = EQUIL
        self.eff = eff
        self.entrainment_perc_in = entrainment_perc_in
        self.entrainment_comp_in = entrainment_comp_in
        self.entrainment_perc_out = entrainment_perc_out

        if convergence is not None: # old verions of Battery
            battery_convergence = convergence
            stage_convergence = convergence

        self.initialize_arrays(Oin, Ain, yin, xin)
        self.check_inputs()

        if self.stages_num==1:
            self.solve_battery(stage_convergence)
            self.runs=0
            self.finalize()
            return

        # Initial run for error calculation
        self.solve_battery(stage_convergence) # Iterates over stages and runs Stage for each one
        self.update_outs()
        self.update_ins()
        self.update_errors()

        if method=='successive':
            self.successive_substitution(max_iter, damping, battery_convergence, stage_convergence)
        if method=='secant':
            self.secant(max_iter, damping, battery_convergence, stage_convergence)
        if method=='damp_secant':
            self.damp_secant(max_iter, damping, battery_convergence, stage_convergence)
        if method=='newton':
            self.newton_raphson(max_iter, battery_convergence, stage_convergence)
        if method=='anderson':
            self.anderson_acceleration(max_iter, battery_convergence, stage_convergence)
        if method=='broyden':
            self.broyden(max_iter, battery_convergence, stage_convergence)

        self.finalize()

    def newton_raphson(self, max_iter, battery_convergence, stage_convergence, step=0.1):
        """
        Solves the battery system using the Newton-Raphson numerical method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.
            step (float, optional): Step size for numerical Jacobian approximation. Defaults to 0.1.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j = 0
        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
            # Compute numerical Jacobian
            current_state = self.outs_array.iloc[-1].copy()
            jacobian = np.zeros((len(current_state), len(current_state)))
            for i in range(len(current_state)):
                perturbed = current_state.copy()
                perturbed[i] *= (1 + step)

                # Evaluate perturbed state
                self.update_stage_connections_from_array(perturbed)
                self.solve_battery(stage_convergence)
                self.update_outs()

                # Compute derivative column
                jacobian[:, i] = (self.errors_array.iloc[-1] - self.errors_array.iloc[-2]) / (step * current_state[i])

            # Solve system using damped Newton step
            try:
                delta = np.linalg.solve(jacobian, -self.errors_array.iloc[-1])
                new_ins = current_state + 0.5 * delta  # Use 0.5 damping factor
            except np.linalg.LinAlgError:
                print("Jacobian is singular, switching to steepest descent step")
                grad = np.dot(jacobian.T, self.errors_array.iloc[-1])
                new_ins = current_state - 0.1 * grad

            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence)
            self.update_outs()
            self.update_ins()
            self.update_errors()

            if j == max_iter:
                print(f'Failed to converge after {max_iter} iterations!')
                break
            j += 1
        self.runs = j

    def anderson_acceleration(self, max_iter, battery_convergence, stage_convergence, m=5):
        """
        Solves the battery system using the Anderson acceleration method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.
            m (int, optional): History size for Anderson acceleration. Defaults to 5.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j = 0
        history_F = []  # History of residuals
        history_X = []  # History of iterates

        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
            current_X = self.outs_array.iloc[-1]
            current_F = self.errors_array.iloc[-1]

            # Update histories
            history_F.append(current_F)
            history_X.append(current_X)
            if len(history_F) > m:
                history_F.pop(0)
                history_X.pop(0)

            if len(history_F) > 1:
                # Construct matrices for least squares problem
                F_mat = np.column_stack([f for f in history_F])
                diff_F = F_mat[:, 1:] - F_mat[:, :-1]

                try:
                    # Solve least squares problem
                    alpha = np.linalg.lstsq(diff_F.T @ diff_F,
                                            -F_mat[:, -1] @ diff_F,
                                            rcond=None)[0]

                    # Compute new iterate
                    theta = np.concatenate([alpha, [1 - alpha.sum()]])
                    new_ins = sum(t * x for t, x in zip(theta, history_X))
                except np.linalg.LinAlgError:
                    # Fallback to simple iteration if least squares fails
                    new_ins = current_X - 0.5 * current_F
            else:
                # Not enough history for acceleration
                new_ins = current_X - 0.5 * current_F

            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence)
            self.update_outs()
            self.update_ins()
            self.update_errors()

            if j == max_iter:
                print(f'Failed to converge after {max_iter} iterations!')
                break
            j += 1
        self.runs = j

    def broyden(self, max_iter, battery_convergence, stage_convergence):
        """
        Solves the battery system using Broyden's quasi-Newton method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j = 0
        # Initial Jacobian approximation (use identity matrix)
        B = np.eye(len(self.outs_array.iloc[-1]))
        x_old = self.outs_array.iloc[-1]
        f_old = self.errors_array.iloc[-1]

        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
            try:
                # Solve system using current Jacobian approximation
                delta = np.linalg.solve(B, -f_old)
                new_ins = x_old + 0.5 * delta  # Use damping
            except np.linalg.LinAlgError:
                # Fallback to steepest descent if matrix is singular
                new_ins = x_old - 0.1 * f_old

            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence)
            self.update_outs()
            self.update_ins()
            self.update_errors()

            # Update Broyden matrix
            x_new = self.outs_array.iloc[-1]
            f_new = self.errors_array.iloc[-1]

            s = x_new - x_old
            y = f_new - f_old

            # Sherman-Morrison update
            B = B + np.outer(y - B @ s, s) / (s @ s)

            x_old = x_new
            f_old = f_new

            if j == max_iter:
                print(f'Failed to converge after {max_iter} iterations!')
                break
            j += 1
        self.runs = j

    def damp_secant(self, max_iter, damping, battery_convergence, stage_convergence):
        """
        Solves the battery system using the damped secant method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            damping (float): Damping factor (0.0 to 1.0).
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j = 0

        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
            # Calculate secant update with damping
            update = self.errors_array.iloc[-1] * (
                self.outs_array.iloc[-1] - self.outs_array.iloc[-2]
            ) / (self.errors_array.iloc[-1] - self.errors_array.iloc[-2] + 1e-10)

            # Limit maximum update size to prevent overshooting
            max_update = np.abs(self.outs_array.iloc[-1]) * 0.2  # Max 20% change
            update = np.clip(update, -max_update, max_update)

            # Apply damping factor
            new_ins = self.outs_array.iloc[-1] - damping * update

            # Ensure physical constraints
            new_ins = np.maximum(new_ins, 0)  # Keep concentrations non-negative

            # Update system state
            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence)
            self.update_outs()
            self.update_ins()
            self.update_errors()

            # Adaptive damping - reduce if error increases
            if j > 0 and abs(self.errors_array.iloc[-1].any()) > abs(self.errors_array.iloc[-2].any()):
                damping *= 0.8  # Reduce damping by 20%
            else:
                damping = min(damping * 1.1, 1.0)  # Slowly increase damping up to 1.0
            if j == max_iter:
                print(f'Failed to converge after {max_iter} iterations!')
                break
            j += 1
        self.runs = j

    def secant(self, max_iter, damping, battery_convergence, stage_convergence):
        """
        Solves the battery system using the secant method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            damping (float): Damping factor (0.0 to 1.0).
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j=0
        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
        # for j in range(20):
            new_ins = self.outs_array.iloc[-1] - damping * self.errors_array.iloc[-1] * (self.outs_array.iloc[-1] - self.outs_array.iloc[-2]) / (self.errors_array.iloc[-1] - self.errors_array.iloc[-2] + 1e-10)
            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence) # Iterates over stages and runs Stage for each one
            self.update_outs()
            self.update_ins()
            self.update_errors()
            if j==max_iter:
                print (max_iter, 'iterations!')
                break
            j+=1
        self.runs = j

    def successive_substitution(self, max_iter, damping, battery_convergence, stage_convergence):
        """
        Solves the battery system using the successive substitution method.

        Parameters:
            max_iter (int): Maximum number of iterations.
            damping (float): Damping factor (0.0 to 1.0).
            battery_convergence (float): Convergence tolerance for the battery.
            stage_convergence (float): Convergence tolerance for each stage.

        Outputs:
            None: Updates the BatteryConv object attributes in place after convergence or reaching max_iter.
        """
        j=0
        while abs(self.errors_array.iloc[-1]).max() > battery_convergence:
        # for j in range(20):
            new_ins = damping * self.outs_array.iloc[-1] + (1 - damping) * self.outs_array.iloc[-2]
            self.update_stage_connections_from_array(new_ins)
            self.solve_battery(stage_convergence) # Iterates over stages and runs Stage for each one
            self.update_outs()
            self.update_ins()
            self.update_errors()
            if j==max_iter:
                print (max_iter, 'iterations!')
                break
            j+=1
        self.runs = j

    def initialize_arrays(self, Oin, Ain, yin, xin):
        """
        Initializes arrays for flows, compositions, and stages.

        Parameters:
            Oin (float): Organic phase inlet flow rate to the first stage.
            Ain (float): Aqueous phase inlet flow rate to the last stage.
            yin (list): Organic phase inlet concentrations to the first stage.
            xin (list): Aqueous phase inlet concentrations to the last stage.

        Outputs:
            None: Initializes various arrays as attributes of the BatteryConv object.
        """
        self.Oin_list = np.ones(self.stages_num) * Oin
        self.Ain_list = np.ones(self.stages_num) * Ain
        yin = np.array(yin, dtype=float) # [%water_org, %H3PO4_org, ...]
        xin = np.array(xin, dtype=float) # [%solvent_aq, %H3PO4_aq, ...]
        self.yin_list = np.array([yin for i in range(self.stages_num)])
        self.xin_list = np.array([xin for i in range(self.stages_num)])
        self.Oout_list = np.zeros(self.stages_num)
        self.Aout_list = np.zeros(self.stages_num)
        self.yout_tag_list = np.zeros((self.stages_num, len(yin)))
        self.yout_list = np.zeros((self.stages_num, len(yin)))
        self.xout_list = np.zeros((self.stages_num, len(xin)))
        self.stages = [[] for i in range(self.stages_num)] # List of stages - currently not used
        self.outs_array = pd.DataFrame(np.zeros((1,(self.stages_num-1)*(len(xin)*2+2))))
        self.ins_array = pd.DataFrame(np.zeros((1,(self.stages_num-1)*(len(xin)*2+2))))
        self.errors_array = pd.DataFrame(np.zeros((1,(self.stages_num-1)*(len(xin)*2+2))))

    def check_inputs(self):
        """
        Checks and processes input parameters related to efficiency and entrainment.

        Parameters:
            None

        Outputs:
            None: Updates attributes eff_list, entrainment_perc_list, and entrainment_comp_list in place.
        """
        # -------- Efficiency processing --------
        eff, stages_num, yin_len = self.eff, self.stages_num, len(self.yin_list[0])
        if np.isscalar(eff):                                                     # scalar
            eff_out = np.ones([stages_num, yin_len]) * eff
        else:
            eff = np.array(eff)
            eff_out = np.ones([stages_num, yin_len])
            if np.isscalar(eff[0]):                                              # vector for elements [water, P2O5, ...] same for all stages
                a = np.array(list(map(lambda x:eff[:yin_len], eff_out))) # 2D array (stage_num, yin_len) with rows of eff
                eff_out[:a.shape[0], :a.shape[1]]=a                             # for the case if eff is shorter than yin_len
            else:
                if eff.shape[1] == 1:                                            # vector for stages same for all elements
                    b = eff.T[0]
                    a = np.array(list(map(lambda x:b[:stages_num], eff_out.T)))
                    eff_out.T[:a.shape[0], :a.shape[1]]=a
                if eff.shape[1] > 1:                                             # list of lists
                    eff_out[:eff.shape[0], :eff.shape[1]]=eff                   # for the case if eff is shorter than yin_len
        self.eff_list = eff_out
        self.eff_list[:,0]=1 # Setting efficiency for water to 100% - assuming water is not extracted.

        # -------- Entrainment percentage processing --------
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

        # -------- Entrainment composition processing --------
        if np.isscalar(self.entrainment_comp_in):
            self.entrainment_comp_in = np.ones([2, len(self.yin_list[0])]) * self.entrainment_comp_in
        else:
            if len(self.entrainment_comp_in[0]) != len(self.yin_list[0]):
                print ('Array of entrainment compositions is not correct.')
        self.entrainment_comp_list = np.array([self.entrainment_comp_in for i in range(self.stages_num)])*1.


    def update_stage_connections_from_array(self, new_ins):
        """
        Updates organic and aqueous inlet flows and compositions for each stage based on the 'new_ins' array.

        Parameters:
            new_ins (pandas.Series): A pandas Series containing updated inlet flow rates and compositions.
                                      The structure is based on the outputs array structure.

        Outputs:
            None: Updates attributes Oin_list, yin_list, Ain_list, xin_list, entrainment_comp_list in place.
        """
        n = len(self.xin_list[0])
        half = int(len(new_ins)/2)
        # Organic phase connections (forward direction)
        for i in range(0, self.stages_num-1):
            self.Oin_list[i+1] = new_ins[i*(n+1)]
            self.yin_list[i+1] = new_ins[i*(n+1)+1:i*(n+1)+n+1].to_numpy()
        # Aqueous phase connections (backward direction)
        for i in range(0, self.stages_num-1):
            self.Ain_list[i] = new_ins[half+i*(n+1)]
            self.xin_list[i] = new_ins[half+i*(n+1)+1:half+i*(n+1)+n+1].to_numpy()
        # Entrainment composition update - assuming entrainment composition of next stage outlet is entrained in current stage inlet
        for i in range(0, self.stages_num-1):
            self.entrainment_comp_list[i][1] = self.yout_list[i+1] # Organic entrainment in Ain is from yout of stage i+1
            self.entrainment_comp_list[i+1][0] = self.xout_list[i] # Aqueous entrainment in Oin is from xout of stage i

    def update_outs(self):
        """
        Updates the outs_array DataFrame with the current outlet flows and compositions.
        The outs_array stores the outlet conditions that will be used as inlets for the next iteration.

        Parameters:
            None

        Outputs:
            None: Updates the outs_array attribute by appending a new row with current outlet conditions.
        """
        outs=[]
        # Organic phase outlets (forward direction)
        for i in range(1, self.stages_num): # Starting from stage 2, using outlet of stage 1 as inlet
            a=np.array([self.Oout_list[i-1]]) # Oout of previous stage
            b=self.yout_list[i-1]           # yout of previous stage
            outs.extend([a.item(), *b.tolist()])
        # Aqueous phase outlets (backward direction)
        for i in range(0, self.stages_num-1): # Up to stage stages_num-1, using outlet of stage stages_num as inlet to stages_num-1 and so on.
            c=np.array([self.Aout_list[i+1]]) # Aout of next stage
            d=self.xout_list[i+1]           # xout of next stage
            outs.extend([c.item(), *d.tolist()])
        outs = np.array(outs).reshape(1,-1)
        self.outs_array=pd.concat([self.outs_array, pd.DataFrame(outs)], axis=0, ignore_index=True)

    def update_ins(self):
        """
        Updates the ins_array DataFrame with the current inlet flows and compositions.
        The ins_array stores the inlet conditions for the current iteration, derived from the outlets of the previous iteration.

        Parameters:
            None

        Outputs:
            None: Updates the ins_array attribute by appending a new row with current inlet conditions.
        """
        ins=[]
        # Organic phase inlets (forward direction)
        for i in range(1, self.stages_num): # Starting from stage 2, using outlet of stage 1 as inlet
            a=np.array([self.Oin_list[i]]) # Oin of current stage
            b=self.yin_list[i]           # yin of current stage
            ins.extend([a.item(), *b.tolist()])
        # Aqueous phase inlets (backward direction)
        for i in range(0, self.stages_num-1): # Up to stage stages_num-1, using outlet of stage stages_num as inlet to stages_num-1 and so on.
            c=np.array([self.Ain_list[i]]) # Ain of current stage
            d=self.xin_list[i]           # xin of current stage
            ins.extend([c.item(), *d.tolist()])
        ins = np.array(ins).reshape(1,-1)
        self.ins_array=pd.concat([self.ins_array, pd.DataFrame(ins)], axis=0, ignore_index=True)

    def update_errors(self):
        """
        Calculates and updates the errors_array DataFrame with the relative errors between current outlets and inlets.
        Errors are calculated as (outs - ins) / (outs + 1e-7) to prevent division by zero.

        Parameters:
            None

        Outputs:
            None: Updates the errors_array attribute by appending a new row with calculated relative errors.
        """
        # new_error = self.outs_array.iloc[-1] - self.ins_array.iloc[-1] # Absolute error - original line, replaced with relative error
        new_error = (self.outs_array.iloc[-1] - self.ins_array.iloc[-1]) / (self.outs_array.iloc[-1] + 1e-7) # Relative error
        self.errors_array=pd.concat([self.errors_array, new_error.to_frame().T], ignore_index=True)

    def calculate_stage(self, stage_idx, flows_comps, stage_convergence):
        """
        Calculates the equilibrium for a single stage using the Stage class.

        Parameters:
            stage_idx (int): Index of the stage to calculate (0-indexed).
            flows_comps (list): List containing inlet flows and compositions for the stage: [Oin, yin, Ain, xin].
                                Oin (float): Organic inlet flow rate.
                                yin (list): Organic inlet composition.
                                Ain (float): Aqueous inlet flow rate.
                                xin (list): Aqueous inlet composition.
            stage_convergence (float): Convergence tolerance for the Stage class.

        Outputs:
            list: Results from the stage calculation, including:
                  [Oout, yout, yout_tag, Aout, xout]
                Oout (float): Organic outlet flow rate.
                yout (numpy.ndarray): Organic outlet composition.
                yout_tag (numpy.ndarray): Organic outlet composition tagged for equilibrium.
                Aout (float): Aqueous outlet flow rate.
                xout (numpy.ndarray): Aqueous outlet composition.
        """
        Oin = flows_comps[0]
        yin = flows_comps[1]
        Ain = flows_comps[2]
        xin = flows_comps[3]
        stage = Stage(
            Oin=Oin,
            Ain=Ain,
            yin=yin,
            xin=xin,
            EQUIL=self.EQUIL,
            eff=self.eff_list[stage_idx],
            convergence=stage_convergence,
            entrainment_perc=self.entrainment_perc_list[stage_idx],
            entrainment_comp_in=self.entrainment_comp_list[stage_idx]
        )
        return [stage.Oout, stage.yout, stage.yout_tag, stage.Aout, stage.xout]

    def solve_battery(self, stage_convergence):
        """
        Solves all stages in the battery iteratively. For each stage, it calculates the equilibrium using calculate_stage method.

        Parameters:
            stage_convergence (float): Convergence tolerance for individual stages (passed to calculate_stage).

        Outputs:
            None: Updates attributes Oout_list, yout_list, yout_tag_list, Aout_list, xout_list with results from each stage.
        """
        for i in range(self.stages_num):
            flows_comps = [self.Oin_list[i],self.yin_list[i],self.Ain_list[i],self.xin_list[i]]
            results = self.calculate_stage(i, flows_comps, stage_convergence)
            self.Oout_list[i] = results[0]
            self.yout_list[i] = results[1]
            self.yout_tag_list[i] = results[2]
            self.Aout_list[i] = results[3]
            self.xout_list[i] = results[4]

    def finalize(self):
        """
        Finalizes the battery calculation after convergence. Calculates distribution ratios (K), and sets final outlet flows and compositions.

        Parameters:
            None

        Outputs:
            None: Updates attributes K, Oout, Aout, yout_tag, yout, xout, entrainment_comp_flat in place.
        """
        self.K = self.xout_list/(self.yout_tag_list+1E-7)
        self.Oout, self.Aout, self.yout_tag, self.yout, self.xout, = self.Oout_list[-1], self.Aout_list[0], self.yout_tag_list[-1], self.yout_list[-1], self.xout_list[0],
        entrainment_comp_flat = []
        for stage in range(self.stages_num):
            entrainment_comp_flat.append([i for item in self.entrainment_comp_list[stage] for i in item])
        self.entrainment_comp_flat = np.array(entrainment_comp_flat)


###################################################################
###################################################################


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

    def __init__(self, stages_num, Oin, Ain, yin, xin, EQUIL, eff=1., max_iter=100, entrainment_perc_in=0, entrainment_comp_in=0, entrainment_perc_out=0, convergence=1E-4):
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
        self.errors=np.zeros([2,len(xin)]) # for stages_num=1

        # k=0
        for i in range(stages_num):
            self.stages[i] = OldStage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff_list[i], 
                                   entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_tag_list[i] = self.stages[i].yout_tag
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout
        if stages_num>1:
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
            self.stages[i] = OldStage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff_list[i], 
                                   entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
            self.Oout_list[i] = self.stages[i].Oout
            self.Aout_list[i] = self.stages[i].Aout
            self.yout_tag_list[i] = self.stages[i].yout_tag
            self.yout_list[i] = self.stages[i].yout
            self.xout_list[i] = self.stages[i].xout 
        if stages_num>1:   
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
                self.stages[i] = OldStage(self.Oin_list[i], self.Ain_list[i], self.yin_list[i], self.xin_list[i], EQUIL=self.EQUIL, eff=self.eff_list[i], 
                                       entrainment_perc=self.entrainment_perc_list[i], entrainment_comp_in=entrainment_comp_list[i], convergence=convergence)
                self.Oout_list[i] = self.stages[i].Oout
                self.Aout_list[i] = self.stages[i].Aout
                self.yout_tag_list[i] = self.stages[i].yout_tag
                self.yout_list[i] = self.stages[i].yout
                self.xout_list[i] = self.stages[i].xout          
            self.errors=np.concatenate((self.errors,[(self.xout_list[1]-self.xin_list[0])/(self.xout_list[1]+1E-7)]), axis=0)
            j+=1
            if j==max_iter:
                print (max_iter, 'iterations!')
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
        self.eff_list = eff_out
        self.eff_list[:,0]=1
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


#####################################################################
#####################################################################


class BatteryTable:
    """
    """
    def __init__(self, batt, show=True, index_names=[]):
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
        if len(index_names)>0:
            Name=index_names
        self.data=pd.DataFrame(out, columns=columns, index=Name)
        if show:
            display (self.data)


#####################################################################
#####################################################################


class BatteryTableFull:
    """
    """
    def __init__(self, batt, show=True, index_names=[]):
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
        if len(index_names)>0:
            Name=index_names
        self.data=pd.DataFrame(out, columns=columns, index=Name)
        if show:
            display (self.data)
