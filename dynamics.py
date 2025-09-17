import numpy as np
import casadi as cs

class BicycleModel:
    """Implements bicycle-model dynamics of a small scale car as detailed in 

    "Cataffo, Vittorio & Silano, Giuseppe & Iannelli, Luigi & Puig, Vicenç & Glielmo, Luigi. 
    (2022). A Nonlinear Model Predictive Control Strategy for Autonomous Racing of Scale Vehicles. 
    10.1109/SMC53654.2022.9945279.".

    Implements methods for computing both continuous time and discrete time dynamics. 
    Supports CasADi for compatibility with OpEn. Use the method dynamics_cs for this. 
    """
    def __init__(self, Ts):
        """Constructor

        :param Ts: Sampling time for discretised system dynamics.
        """
        self.Ts = Ts
        self.car_width = 0.25
        self.car_length = 0.4
        self.m = 5.692  # mass of the car in kg
        self.Jz = .204  # MOI of car along z-axis
        self.lf = .178  # distance b/w Cg and front wheel axel
        self.lr = .147  # distance b/t Cg and rear wheel axel

        # parameters for simplified Pacejkas's magic formula
        self.Bf = 9.242  # front tire stiffness factor
        self.Br = 17.716  # rear tire stiffness factor
        self.Cf = .085  # front tire shape factor
        self.Cr = .133  # rear tire shape factor
        self.Df = 134.585  # front tire peak factor
        self.Dr = 159.919  # rear tire peak factor

        # empirical parameters for fitting model's response to drivetrain characteristics.
        self.Cm1 = 20
        self.Cm2 = 6.92E-7
        self.Cm3 = 3.99
        self.Cm4 = .67

        self.brake_coeff_of_kinetic_fric = 0.1

        self.xt = None  # variable to store vehicle's x global x position

    def dynamics_generic_dt(self, z, u, lib=cs):
        """Compute and return the next state of the discretised bicycle-model dynamical system.

        :param z: current state of the system as a list [xt, yt, psi_t, vxt, vyt, omega_t].
        :param u: current input to the system as a list [pwm_t, delta_t, brake].
        :param lib: select library to use for computing the next state, between {NumPy, CasADi}.
        :return: returns the list of computed next system states.
        """
        xt = z[0]  # current x position of the vehicle
        yt = z[1]  # current y position of the vehicle
        psi_t = z[2]  # current heading of the vehicle
        vxt = z[3]  # current x-velocity of the vehicle
        vyt = z[4]  # current y-velocity of the vehicle
        omega_t = z[5]  # current angular velocity of the vehicle along the z-axis

        pwm_t = u[0]  # current pwm (acceleration) input to the vehicle
        delta_t = u[1]  # current steering angle applied to the vehicle
        brake = u[2]  # current brake force applied to the vehicle

        # code to compute the dynamics 
        if lib == np:
            alpha_f = -lib.arctan2(omega_t*self.lf + vyt, vxt) + delta_t
            alpha_r = lib.arctan2(omega_t*self.lr - vyt, vxt)
            Ffy = self.Df*lib.sin(self.Cf*lib.arctan2(self.Bf*alpha_f, 1))
            Fry = self.Dr*lib.sin(self.Cr*lib.arctan2(self.Br*alpha_r, 1))
        else:
            alpha_f = -lib.atan2(omega_t*self.lf + vyt, vxt) + delta_t
            alpha_r = lib.atan2(omega_t*self.lr - vyt, vxt)
            Ffy = self.Df*lib.sin(self.Cf*lib.atan2(self.Bf*alpha_f, 1))
            Fry = self.Dr*lib.sin(self.Cr*lib.atan2(self.Br*alpha_r, 1))

        Frx = (self.Cm1 - self.Cm2*vxt)*pwm_t - self.Cm3 - self.Cm4*vxt**2
        Ffx = Frx

        xt_ = xt + self.Ts*(vxt*lib.cos(psi_t) - vyt*lib.sin(psi_t))
        yt_ = yt + self.Ts*(vxt*lib.sin(psi_t) + vyt*lib.cos(psi_t))
        psi_t_ = psi_t + self.Ts*omega_t

        Fx_brake = self.brake_coeff_of_kinetic_fric * brake

        vxt_ = vxt + self.Ts*(Frx - Fx_brake - Ffy*lib.sin(delta_t) + (Ffx - Fx_brake)*lib.cos(delta_t) + self.m*vyt*omega_t)/self.m  # - self.Ts*self.brake_coeff_of_kinetic_fric * vxt * brake
        vyt_ = vyt + self.Ts*(Fry + Ffy*lib.cos(delta_t) + (Ffx - Fx_brake)*lib.sin(delta_t) - self.m*vxt*omega_t)/self.m  # - self.Ts*self.brake_coeff_of_kinetic_fric * vyt * brake
        omega_t_ = omega_t + self.Ts*(self.lf*Ffy*lib.cos(delta_t) + self.lf*(Ffx - Fx_brake)*lib.sin(delta_t) - self.lr*Fry)/self.Jz

        # return the computed next states
        return [xt_, yt_, psi_t_, vxt_, vyt_, omega_t_]

    def dynamics_generic_ct(self, z, u, lib=cs): 
        """Compute and return the next state of the continuos-time bicycle-model dynamical system.

        :param z: current state of the system as a list [xt, yt, psi_t, vxt, vyt, omega_t].
        :param u: current input to the system as a list [pwm_t, delta_t, brake].
        :param lib: select library to use for computing the next state, between {NumPy, CasADi}.
        :return: returns the list of computed next system states.
        """
        xt = z[0]  # current x position of the vehicle
        yt = z[1]  # current y position of the vehicle
        psi_t = z[2]  # current heading of the vehicle
        vxt = z[3]  # current x-velocity of the vehicle
        vyt = z[4]  # current y-velocity of the vehicle
        omega_t = z[5]  # current angular velocity of the vehicle along the z-axis

        pwm_t = u[0]  # current pwm (acceleration) input to the vehicle
        delta_t = u[1]  # current steering angle applied to the vehicle
        brake = u[2]  # current brake force applied to the vehicle

        # code to compute the dynamics 
        if lib == np:
            alpha_f = -lib.arctan2(omega_t*self.lf + vyt, vxt) + delta_t
            alpha_r = lib.arctan2(omega_t*self.lr - vyt, vxt)
            Ffy = self.Df*lib.sin(self.Cf*lib.arctan2(self.Bf*alpha_f, 1))
            Fry = self.Dr*lib.sin(self.Cr*lib.arctan2(self.Br*alpha_r, 1))
        else:
            alpha_f = -lib.atan2(omega_t*self.lf + vyt, vxt) + delta_t
            alpha_r = lib.atan2(omega_t*self.lr - vyt, vxt)
            Ffy = self.Df*lib.sin(self.Cf*lib.atan2(self.Bf*alpha_f, 1))
            Fry = self.Dr*lib.sin(self.Cr*lib.atan2(self.Br*alpha_r, 1))

        Frx = (self.Cm1 - self.Cm2*vxt)*pwm_t - self.Cm3 - self.Cm4*vxt**2
        Ffx = Frx

        xdot = vxt*lib.cos(psi_t) - vyt*lib.sin(psi_t)
        ydot = vxt*lib.sin(psi_t) + vyt*lib.cos(psi_t)
        psi_dot = omega_t

        Fx_brake = self.brake_coeff_of_kinetic_fric * brake

        vxdot = (Frx - Fx_brake - Ffy*lib.sin(delta_t) + (Ffx - Fx_brake)*lib.cos(delta_t) + self.m*vyt*omega_t)/self.m  # - self.brake_coeff_of_kinetic_fric * vxt * brake
        vydot = (Fry + Ffy*lib.cos(delta_t) + (Ffx - Fx_brake)*lib.sin(delta_t) - self.m*vxt*omega_t)/self.m  # - self.brake_coeff_of_kinetic_fric * vyt * brake
        omega_dot = (self.lf*Ffy*lib.cos(delta_t) + self.lf*(Ffx - Fx_brake)*lib.sin(delta_t) - self.lr*Fry)/self.Jz

        # return the computed next states
        return [xdot, ydot, psi_dot, vxdot, vydot, omega_dot]

    def dynamics_cs(self, z, u):
        """Compute and return the next state of the discretised bicycle-model dynamical system
        using CasADi methods.

        :param z: current state of the system as a list [xt, yt, psi_t, vxt, vyt, omega_t].
        :param u: current input to the system as a list [pwm_t, delta_t, brake].
        :param lib: select library to use for computing the next state, between {NumPy, CasADi}.
        :return: returns the list of computed next system states.
        """
        xt = z[0]  # current x position of the vehicle
        yt = z[1]  # current y position of the vehicle
        psi_t = z[2]  # current heading of the vehicle
        vxt = z[3]  # current x-velocity of the vehicle
        vyt = z[4]  # current y-velocity of the vehicle
        omega_t = z[5]  # current angular velocity of the vehicle along the z-axis

        pwm_t = u[0]  # current pwm (acceleration) input to the vehicle
        delta_t = u[1]  # current steering angle applied to the vehicle
        brake = u[2]  # current brake force applied to the vehicle
        
        # code to compute the dynamics 
        alpha_f = -cs.atan2(omega_t*self.lf + vyt, vxt) + delta_t
        alpha_r = cs.atan2(omega_t*self.lr - vyt, vxt)

        Ffy = self.Df*cs.sin(self.Cf*cs.atan2(self.Bf*alpha_f, 1))
        Fry = self.Dr*cs.sin(self.Cr*cs.atan2(self.Br*alpha_r, 1))
        Frx = (self.Cm1 - self.Cm2*vxt)*pwm_t - self.Cm3 - self.Cm4*vxt**2
        Ffx = Frx

        xt = xt + self.Ts*(vxt*cs.cos(psi_t) - vyt*cs.sin(psi_t))
        yt = yt + self.Ts*(vxt*cs.sin(psi_t) + vyt*cs.cos(psi_t))
        psi_t = psi_t + self.Ts*omega_t

        Fx_brake = self.brake_coeff_of_kinetic_fric * brake

        vxt = vxt + self.Ts*(Frx - Fx_brake - Ffy*cs.sin(delta_t) + (Ffx - Fx_brake)*cs.cos(delta_t) + self.m*vyt*omega_t)/self.m  # - self.Ts*self.brake_coeff_of_kinetic_fric * vxt * brake
        vyt = vyt + self.Ts*(Fry + Ffy*cs.cos(delta_t) + (Ffx - Fx_brake)*cs.sin(delta_t) - self.m*vxt*omega_t)/self.m  # - self.Ts*self.brake_coeff_of_kinetic_fric * vyt * brake
        omega_t = omega_t + self.Ts*(self.lf*Ffy*cs.cos(delta_t) + self.lf*(Ffx - Fx_brake)*cs.sin(delta_t) - self.lr*Fry)/self.Jz

        # return the computed next states
        return cs.vertcat(xt, yt, psi_t, vxt, vyt, omega_t)