#Kalman Filter 

import numpy as np

class KalmanFilter():
    def __init__(self,initial_state_x, initial_covariance_P, transition_matrix_F,external_input_B, process_noise_Q, measurement_covariance_R, Hx, HJacobian):
        self._x = initial_state_x
        self._P = initial_covariance_P
        self._F = transition_matrix_F
        self._B = external_input_B
        self._Q = process_noise_Q
        self._R = measurement_covariance_R
        self._Hx = Hx 
        self._HJacobian = HJacobian

    def update(self,z):
        #In normal Kalman Filter 
        #Kalman Gain: K = P / (P + R)
        #Update state estimate: x = x_hat + K * (z- H * x_hat)
        #Update error covariance: P = (1-K) * P 
        
        P = self._P
        R = self._R 
        x = self._x
        # In Normal KF H = 1 measurement matrix for a direct measurement

        H = self._HJacobian(x)
        
        #Calculate Kalman gain in normal KF 
        # K = P / (P + R) 
        # x = x + K * (measurement - H * x)
        # P = (1 - K)*P

        #In extended KF
        S = H * P * H.T + R
        K = P * H.T * S.I 
        self._K = K
        hx = self._Hx(x)
        y = np.subtract(z,hx)
        self._x = x + K * y

        KH = K * H 
        I_KH = np.identity((KH).shape[1]) - KH
        self._P = I_KH * P * I_KH.T + K * R * K.T

    def predict (self, u_control_input = 0):
        #in normal KF 
        #Predicted state estimate: x_hat = A * x + B * u
        #Predicted error covariance: P = A * P * A.T + Q 
        '''
        A =1 
        B = 0
        u = control_input 
        Q = self._Q

        predicted_state = A * self._x + B * u
        predicted_covariance = A * self._P * A + Q 
        self._x = predicted_state 
        self._P = predicted_covariance
        '''

        #In EKF 
        self._x = self._F * self._x + self._B * u_control_input
        self._P = self._F * self._P * self._F.T + self._Q

    @property
    def x(self):
        return self._x


        


        
