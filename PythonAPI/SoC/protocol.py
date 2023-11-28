# Change in experiment; 

def launch_experiment_protocol(Q_tot, time_step, experiment_callback):
        charge_current_rate = 0.5 
        discharge_current_rate = 1 
        discharge_constants_stages_time = 20*60
        pulse_time = 60 
        total_pulse_time = 40*60

        high_cut_off_voltage = 4.2
        low_cut_off_voltage = 2.5

        #Charge CC 
        current = -charge_current_rate * Q_tot
        voltage = 0

        while voltage < high_cut_off_voltage:
                voltage = experiment_callback(current)


        #Charge CV 
        while current < -0.1: 
                if voltage > high_cut_off_voltage*1.001:
                        current += 0.01*Q_tot
                
                voltage = experiment_callback(current) 

        time = 0
        current = discharge_current_rate*Q_tot
        while time < discharge_constants_stages_time: 
                experiment_callback(current)
                time += time_step
        
        time = 0 
        while time < total_pulse_time:
            time_low = 0
            current = 0
            while time_low < pulse_time:
                    experiment_callback(current) 
                    time_low += time_step
            time_high = 0
            current = discharge_current_rate * Q_tot
            while time_high < pulse_time:
                    experiment_callback(current)
                    time_high += time_step
            time += time_low + time_high

        time = 0 
        current = discharge_current_rate * Q_tot
        while time < discharge_constants_stages_time:
                experiment_callback(current)
                time += time_step
