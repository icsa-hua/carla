from Battery import battery 
from kalmanfilter import KalmanFilter as KF
from protocol import launch_experiment_protocol 
import matplotlib.pyplot as plt

import numpy as np 
import math as m 

def get_KF(R0,R1,C1,std_dev, time_step):
    #initial state (SoC is intentionally set to a wrong value)
    #x = [[SoC],[RC_voltage]]

    x = np.matrix([[0.5],[0.0]])

    exp_coeff = m.exp(-time_step/(C1*R1))

    #State transition matrix 
    F = np.matrix([[1,0],[0,exp_coeff]])

    #control-input model 
    B = np.matrix([[-time_step/(Q_tot * 3600)],[R1*(1-exp_coeff)]])

    #Variance from the std 
    var = std_dev**2

    #measurement noise 
    R = var 

    #State covariance 
    P = np.matrix([[var,0],[0,var]])
    
    #Process noise covariance matrix 
    Q = np.matrix([[var/50,0],[0,var/50]])

    def HJacobian(x):
        return np.matrix([[battery_simulation.OCV_model(x[0,0]),-1]])

    def Hx(x):
        return battery_simulation.OCV_model(x[0,0]) - x[1,0]
    
    return KF(x,P,F,B,Q,R,Hx,HJacobian)

def plot_everything(time, true_voltage, mes_voltage, true_SoC, estim_SoC, current):

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    # title, labels
    ax1.set_title('')    
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('voltage / V')
    ax2.set_xlabel('Time / s')
    ax2.set_ylabel('Soc')
    ax3.set_xlabel('Time / s')
    ax3.set_ylabel('Current / A')


    ax1.plot(time, true_voltage, label="True voltage")
    ax1.plot(time, mes_voltage, label="Mesured voltage")
    ax2.plot(time, true_SoC, label="True SoC")
    ax2.plot(time, estim_SoC, label="Estimated SoC")
    ax3.plot(time, current, label="Current")
    
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()


if __name__ == '__main__':

    #total Capacity 
    Q_tot = 3.2 
     
    #Thevenin Model Values 
    R0 = 2.4
    R1 = 2.3 
    C1 = 11.326
    R2 = 3.9
    C2 = 196816

    time_step = 10 
    battery_simulation = battery(Q_tot,R0,R1,C1,R2,C2) 

    #Discharged battery 
    battery_simulation.actual_capacity = 0
    
    #Measurement noise standard deviation 
    std_dev = 0.015 

    #Get configured KF 
    Kf = get_KF(R0,R1,C1,std_dev,time_step)

    time = [0] 
    true_SoC = [battery_simulation.state_of_charge]
    estim_SoC = [Kf.x[0,0]]
    true_voltage = [battery_simulation.voltage]
    mes_voltage = [battery_simulation.voltage + np.random.normal(0,0.1,1)[0]]
    current = [battery_simulation.current]

    def update_all(actual_current):
        battery_simulation.current = actual_current
        battery_simulation.update(time_step)

        time.append(time[-1] + time_step)
        current.append(actual_current)

        true_voltage.append(battery_simulation.voltage)
        mes_voltage.append(battery_simulation.voltage + np.random.normal(0,std_dev,1)[0])

        Kf.predict(u_control_input=actual_current)
        Kf.update(mes_voltage[-1] + R0 * actual_current)

        true_SoC.append(battery_simulation.state_of_charge)
        estim_SoC.append(Kf.x[0,0])

        return battery_simulation.voltage 
    
    launch_experiment_protocol(Q_tot,time_step,update_all)

    plot_everything(time,true_voltage,mes_voltage,true_SoC,estim_SoC,current)










