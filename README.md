## IS-17550-3-Std
#Ref Test Std calculation as per IS 17550

## Import all essential library

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")



## Import raw data
SxS_df = pd.read_excel(r'D:\HAIL_DATA-1\E drive Data\2022\New Energy Norms _2023\SxS_EnergyData.xlsx', sheet_name='Raw Data')
SxS_df.columns

## length of total data in Raw data
Power = SxS_df['Power']
Power =np.array(Power)
print('Length of data point in raw data: ', Power.shape)

plt.figure(figsize=(500, 100))
#plt_1 = plt.figure(figsize=(6, 3))

plt.title("Power Graph ")
plt.xlabel("Time ")
plt.ylabel("Power (W) ")
plt.plot(Power, color = 'r', linewidth = '10')
plt.show()
Freezer_Temp = SxS_df['Favg']
Fridge_Temp = SxS_df['Ravg']
Int_Energy = SxS_df ['Int_Power']


# We have to calculate first defrost in data
## Defrost calcuation in Raw data

Defrost_start = []
x=len(Power) 
for c in range (x-1):
 if (Power[c+1]-Power[c]) > 150:
        defrost = float(c)
        Defrost_start.append(defrost)

print(Defrost_start)

Defrost_end = []
for c in range (x-1):
 if (Power[c]-Power[c+1]) > 150:
        defrost = float(c)
        Defrost_end.append(defrost)

print(Defrost_end)

print('No of defrost in data :', len(Defrost_end))

## All paramter define/short for X period
#for n in range (x):
 #   if Power[n] > 150:
 #       print (n)
 # data for power in X period
aa = int(Defrost_start[0])
Power_x = Power[:aa,]

Power_x_period = Power_x
plt.figure(figsize=(500, 100))
plt.plot(Power_x, color = 'r', linewidth = '10')


## average power calculation function for each zone
def average_power (raw_data):
    f = len(raw_data)
    number_cycle =[]
    for e in range (f-1):
        if (raw_data[e+1]-raw_data[e]) > 30:
            cycle = int(e)
            number_cycle.append(cycle)
                  
    cycle_avg = []
    no_cycle =len(number_cycle)
    for t in range (no_cycle-1):
        a = int (number_cycle[t])
        b = int (number_cycle[t+1])
        cycle_a = np.mean (raw_data[a:b])
        cycle_avg.append(cycle_a)
    return cycle_avg ,number_cycle
  print('No of data length in X period ', len(Power_x))
  
  
 ## Data cleaning for on off cycle because too much flutuation in data

def cycle_time_data_clean (xxx):

    xxx = np.array (xxx)
    YY = []
    for i in range (len(xxx)):
        if xxx[i] > 100:
            YY.append(i)   
    xxx = np.delete(xxx, YY)
    
    YY = []
    for j in range (len(xxx)):
        if xxx[j] < 75 and xxx[j] >25:
            YY.append(j)   
    xxx = np.delete(xxx, YY)    
    
    
    return xxx
   
## On Off cycle calculation function
 
    def on_off_cycle (raw_data):
    q = len(raw_data)
    on_cycle =[]        
    for e in range (q-1):
        if (raw_data[e+1]-raw_data[e]) > 30:
            On_cycle = int(e)
            on_cycle.append(On_cycle)

    
    off_cycle =[]        
    for e in range (q-1):
        if (raw_data[e]-raw_data[e+1]) > 60:
            Of_cycle = int(e)
            off_cycle.append(Of_cycle)
            
        
    return on_cycle , off_cycle     
            
## Compressor Run percentage calculation in stable period use clean data           
def run_per (on , off):
    percentage = []
    x = len (on)
    for i in range (x-4):
        n = on[i] - off[i]
        f = off[i] - on[i+1]
        per = n / (n + f)
        percentage.append(per)
    return percentage
    
  Power_x = cycle_time_data_clean (Power_x)
  
  
## Cycle average value in period calculation  
cycle_average_value, no_cycle_x_period = average_power (Power_x)
length = len (cycle_average_value)
print('Number of cycle in X period: ', length )
print('Each cycle avergae: ', cycle_average_value)
#print(no_cycle_x_period)
#print(cycle_average_value)
print ('Cycle value start and end point: ',  no_cycle_x_period )



## Property calculation for each cycle points

def all_temp_data (no_cycle_x_period, req_data):
    data_store= []
    ln = len (no_cycle_x_period)
    for i in range (ln-1):
        a = no_cycle_x_period[i]
        b = no_cycle_x_period[i+1]
        xx = np.mean(req_data[a:b])
        data_store.append(xx)             
    return data_store
 
## Calculate all value in X period

Power_xxx = all_temp_data (no_cycle_x_period, Power_x)
print(Power_xxx)

Power_xxx = all_temp_data (no_cycle_x_period, Power_x)
print('Stable each cycle data ', Power_xxx)

power_avg_x =np.mean(Power_xxx[(length-8):(length-2)])
print('Average Power in X Period :', power_avg_x)

## Freezer Room average temperature in X period 
freezer_temp_x = all_temp_data (no_cycle_x_period, Freezer_Temp)
print('Stable each cycle data frezzer room ', freezer_temp_x)

freezer_avg_x =np.mean(freezer_temp_x[(length-8):(length-2)])
print('Average Freezer in X Period :', freezer_avg_x)
#print(Power_xxx)
#Freezer_avg_x =np.mean(no_cycle_x_period[(length-8):])
#print('Average Freezer Temperature  X Period :', Freezer_avg_x)

## Frride Room average temperature in X period 
fridge_temp_x = all_temp_data (no_cycle_x_period, Fridge_Temp)
print('Stable each cycle data fridge room ', fridge_temp_x)

fridge_avg_x =np.mean(fridge_temp_x[(length-8):(length-2)])
print('Average Fridge in X Period :', fridge_avg_x)

x_time_End = no_cycle_x_period[-1]
print ('No of cycle period last time = ', no_cycle_x_period[-1]) 
x_time_Start = no_cycle_x_period[-6]
print ('No of cycle period first time= ', no_cycle_x_period[-6]) 
x_Time = (x_time_End - x_time_Start ) /120 
print ('X Period in Hours  = ', x_Time      )

## Calculate all value in Y period
## data for power in Y period
ab = int(Defrost_end[0])
bb = int(Defrost_start[1])
Power_y = Power[ab:bb,]
print(Power_y)
print ('Y period data length: ',Power_y.shape )
Power_y = cycle_time_data_clean (Power_y)

cycle_average_value_y, no_cycle_y_period = average_power (Power_y)
length = len (cycle_average_value_y)
print('Number of cycle in X period: ', length )
print('Each cycle avergae: ', cycle_average_value_y)

print ('Cycle value start and end point: ',  no_cycle_y_period )


Power_yyy = all_temp_data (no_cycle_y_period, Power_y)
print(Power_yyy)

Power_yyy = all_temp_data (no_cycle_y_period, Power_y)
print('Stable each cycle data ', Power_yyy)

power_avg_y =np.mean(Power_yyy[(length-8):(length-2)])
print('Average Power in X Period :', power_avg_y)

# Freezer Room average temperature in Y period 
Freezer_Temp_y = Freezer_Temp[ab:bb,]
freezer_temp_y = all_temp_data (no_cycle_y_period, Freezer_Temp_y)
print('Stable each cycle data frezzer room ', freezer_temp_y)

freezer_avg_y =np.mean(freezer_temp_y[(length-8):(length-2)])
print('Average Freezer in Y Period :', freezer_avg_y)

# Fridge Room average temperature in X period 
Fridge_Temp_y=  Fridge_Temp[ab:bb,]
fridge_temp_y = all_temp_data (no_cycle_y_period, Fridge_Temp_y)
print('Stable each cycle data fridge room ', fridge_temp_y)

fridge_avg_y =np.mean(fridge_temp_y[(length-8):(length-2)])
print('Average Fridge in Y Period :', fridge_avg_y)

# Time Period calculated in Y period
#print ('No of cycle period last time = ', no_cycle_y_period[-1]) 
#print ('No of cycle period first time= ', no_cycle_y_period[-6]) 
#y_Time = (no_cycle_y_period[-1] - no_cycle_y_period[-6] ) /120 
#print ('X Period in Hours  = ', y_Time      )


y_time_End = no_cycle_y_period[-1]
print ('No of cycle period last time = ', no_cycle_y_period[-1]) 
y_time_Start = no_cycle_y_period[-6]
print ('No of cycle period first time= ', no_cycle_y_period[-6]) 
y_Time = (y_time_End - y_time_Start ) /120 
print ('X Period in Hours  = ', y_Time      )

## Now calculated Other paramter between X - Y Period

#Energy at end up  X Period						
#Energy at end up  Y Period						
#Time at end up  X Period						
#Time at end up  Y Period						
#Average Power at end up  X Period to end up Y period						
#Average Freezer Temperature at end up  X Period to end up Y period						
#Average Fridge Temperature at end up  X Period to end up Y period						
#Estimated Time at end up  X Period to end up Y period						
#Estimated  Energy at end up  X Period to end up Y period						


Int_Energy =np.array(Int_Energy)
plt.plot(Int_Energy)

X_time_End = x_time_End
X_time_Start = x_time_Start 
print ('Time at End up  X Period : ', x_time_End )
print ('Time at Start up  X Period : ', x_time_Start )

print ('Energy at End up  X Period : ', Int_Energy[x_time_End] )
print ('Energy at Start up  X Period : ', Int_Energy[x_time_Start] )

Y_time_End = y_time_End + aa
Y_time_Start = y_time_Start + aa
print ('Time at End up  y Period : ', Y_time_End )
print ('Time at Start up  y Period : ', Y_time_Start )

print ('Energy at End up  y Period : ', Int_Energy[Y_time_End] )
print ('Energy at Start up  y Period : ', Int_Energy[Y_time_Start] )


#Average Power at end up  X Period to end up Y period
Power_avg_xy = np.mean(Power[X_time_End:Y_time_End])
print('Average Power at end up  X Period to end up Y period; ', Power_avg_xy)


## Average Freezer Temperature at end up  X Period to end up Y period
Freezer_avg_xy = np.mean(Freezer_Temp[X_time_End:Y_time_End])

print('Average Freezer Temperature at end up  X Period to end up Y period; ', Freezer_avg_xy)

#Average Fridge Temperature at end up  X Period to end up Y period

Fridge_avg_xy = np.mean(Fridge_Temp[X_time_End:Y_time_End])

print('Average Fridge Temperature at end up  X Period to end up Y period; ', Fridge_avg_xy)


## Estimated Time at end up  X Period to end up Y period
print ('Time at End up  x Period : ', X_time_End )
print ('Time at End up  y Period : ', Y_time_End )

Est_time_xy =(Y_time_End - X_time_End ) /120
print ('Estimated Time at end up  X Period to end up Y period : ', Est_time_xy )

## Estimated  Energy at end up  X Period to end up Y period
Est_Int_Energy_xy = Int_Energy[Y_time_End] - Int_Energy[x_time_End]
print ('Estimated  Energy at end up  X Period to end up Y period : ', Est_Int_Energy_xy    )

#Energy at Start up D Period						
#Energy at end up D Period						
#Energy at Start up  F Period						
#Energy at end up  F Period						
#Time at Start up  D Period						
#Time at end up  D Period						
#Time at Start up  F Period						
#Time at end up  F Period						
#Average Power in  D Period						
#Average Power in  F Period						
#Average Freezer Temperature in  D Period						
#Average Fridge Temperature  in  F Period						
#Average Freezer Temperature in  D Period						
#Average Fridge Temperature  in  F Period						


## D period taken from before of 1st defrost
no_cycle_d_period = no_cycle_x_period[-9:]
print('data ponit taken for D period: ' , no_cycle_d_period  )

## f period taken from before of 1st defrost
no_cycle_f_period = no_cycle_y_period[2:11] 
no_cycle_f_period = [ x + ab for x in no_cycle_f_period]
print('data ponit taken for D period: ' , no_cycle_f_period  )


d_time_End = no_cycle_d_period[-1]
print ('No of cycle period last time = ', no_cycle_d_period[-1]) 
d_time_Start = no_cycle_d_period[-6]
print ('No of cycle period first time= ', no_cycle_y_period[-6]) 
d_Time = (d_time_End - d_time_Start ) /120 
print ('D Period in Hours  = ', d_Time      )


f_time_End = no_cycle_f_period[-1]
print ('No of cycle period last time = ', no_cycle_f_period[-1]) 
f_time_Start = no_cycle_f_period[-6]
print ('No of cycle period first time= ', no_cycle_f_period[-6]) 
f_Time = (f_time_End - f_time_Start ) /120 
print ('F Period in Hours  = ', f_Time      )

D_time_End = d_time_End
D_time_Start = d_time_Start

F_time_End = f_time_End
F_time_Start = f_time_Start

print ('Energy at End up of d Period : ', Int_Energy[D_time_End] )
print ('Energy at Start of  d Period : ', Int_Energy[D_time_Start] )
print ('Energy at End up of f Period : ', Int_Energy[F_time_End] )
print ('Energy at Start of  f Period : ', Int_Energy[F_time_Start] )


print ('Time at End of  D Period : ', D_time_End )
print ('Time at Start of  D Period : ', D_time_Start )
print ('Time at End of  F Period : ', F_time_End )
print ('Time at Start of  F Period : ', F_time_Start )

#Average Power in  D Period						
#Average Power in  F Period						
Power_avg_D = np.mean(Power[D_time_Start:D_time_End])
print('Average Power in  D Period; ', Power_avg_D)

Power_avg_F = np.mean(Power[F_time_Start:F_time_End])
print('Average Power in  F Period; ', Power_avg_F)


#Average Fridge Temperature in  D Period						
#Average Fridge Temperature  in  F Period						

Freezer_avg_D = np.mean(Freezer_Temp[D_time_Start:D_time_End])

print('Average Freezer Temperature in  D Period; ', Freezer_avg_D)

Freezer_avg_F = np.mean(Freezer_Temp[F_time_Start:F_time_End])

print('Average Freezer Temperature in  F Period; ', Freezer_avg_F)

#Average Freezer Temperature in  D Period						
#Average Freezer Temperature  in  F Period						

Fridge_avg_D = np.mean(Fridge_Temp[D_time_Start:D_time_End])

print('Average Freezer Temperature in  D Period; ', Fridge_avg_D)

Fridge_avg_F = np.mean(Fridge_Temp[F_time_Start:F_time_End])

print('Average Freezer Temperature in  F Period; ', Fridge_avg_F)


#Estimated  Energy at end up  D Period to end up F period						
#Estimated  Time at end up  D Period to end up F period						
#Estimated  Average Freezer at end up  D Period to end up F period						
#Estimated  Average Fridge at end up  D Period to end up F period						
#Estimated  Average Power at   D Period &  F period						
#Estimated Average Freezer Temperature at   D Period &  F period						
#Estimated Average Freezer Temperature at   D Period &  F period						
#Estimated  Time at   D Period Start to  F period end						


## Estimated  Energy at end up  D Period to end up F period

Est_Int_Energy_DF = Int_Energy[F_time_End] - Int_Energy[D_time_End]
print ('Estimated  Energy at end up  D Period to end up F period : ', Est_Int_Energy_df    )

Est_Int_Energy_DsFe = Int_Energy[F_time_End] - Int_Energy[D_time_Start]
print ('Estimated  Energy at Start of  D Period to end up F period : ', Est_Int_Energy_DsFe    )

## Estimated  Time at end up  D Period to end up F period


print ('Time at End up  F Period : ', F_time_End )
print ('Time at End up  D Period : ', D_time_End )

Est_time_DF =(F_time_End - D_time_End ) /120
print ('Estimated Time at end up  X Period to end up Y period : ', Est_time_df )


## Estimated  Average Freezer at end up  D Period to end up F period

Freezer_avg_DF = np.mean(Freezer_Temp[D_time_End:F_time_End])
print('Estimated  Average Freezer at end up  D Period to end up F period: ', Freezer_avg_DF)


## Estimated  Average Fridge at end up  D Period to end up F period

Fridge_avg_DF = np.mean(Fridge_Temp[D_time_End:F_time_End])

print('Average Freezer Temperature in  D Period; ', Fridge_avg_DF)

## Estimated  Average Power at   D Period &  F period
Power_avg_df = (Power_avg_D + Power_avg_F)/2

print('Estimated  Average Power at   D Period &  F period: ', Power_avg_df)

#Estimated Average Freezer Temperature at   D Period &  F period

Freezer_avg_df = (Freezer_avg_D + Freezer_avg_F) /2

print ('Estimated Average Freezer Temperature at   D Period &  F period :', Freezer_avg_df)

#Estimated Average Fridge Temperature at  D Period &  F period

Fridge_avg_df = ( Fridge_avg_D + Fridge_avg_F )/2

print ('Estimated Average Fridge Temperature at  D Period &  F period; ', Fridge_avg_df)

#Estimated  Time at   D Period Start to  F period end  




print ('Time at Sart of  D Period : ', D_time_Start )
print ('Time at End up  F Period : ', F_time_End )

Est_time_df =(F_time_End - D_time_Start ) /120
print ('Estimated  Time (hours) at   D Period Start to  F period end: ', Est_time_df )




#Accumulated temperature difference over time in compartment F						
#Accumulated temperature difference over time in compartment R						
#Additional energy consumed by the refrigerating appliance for defrost and recovery period						
						
#maximum possible defrost interval						
#minimum possible defrost interval						
#Defrost interval for an ambient temperature of 32 °C						
#Defrost interval for an ambient temperature of 32 °C (compressor control)						
						
#Steady state power for the selected defrost control cycle 						
						
#Energy in Wh over a period of 24hr						
#Energy in Wh over a period of year (unit)						
						
#steady state temperature in compartment F that occurs in the whole test period used for SS2 in degrees C;						
#steady state temperature in compartment R that occurs in the whole test period used for SS2 in degrees C;						
						
#Average temperature for the compartment F over a complete defrost control cycle						
#Average temperature for the compartment R over a complete defrost control cycle


## Accumulated temperature difference over time in compartment F (∆Thdf_F)
del_Thdf_F =  Est_time_df *(Freezer_avg_DF - Freezer_avg_df)

print ('Accumulated temperature difference over time in compartment F (∆Thdf_F) : ' , del_Thdf_F)

#Accumulated temperature difference over time in compartment R (∆Thdf_R)

del_Thdf_R =  Est_time_df *(Fridge_avg_DF - Fridge_avg_df)

print ('Accumulated temperature difference over time in compartment R (∆Thdf_R) : ' , del_Thdf_R)

## Additional energy consumed by the refrigerating appliance for defrost and recovery period (∆Edf)

del_Edf =    Est_Int_Energy_DsFe - (Power_avg_df *Est_time_df )  

print ('Additional energy consumed by the refrigerating appliance for defrost and recovery period (∆Edf) : ' , del_Edf)

#compressor run % calculated based on stable period (consider X  Period)
data_power_x = cycle_time_data_clean (Power_x_period)
on_time, off_time =  on_off_cycle (data_power_x)

#on_t = len(on_time)
#off_t =len(off_time)

#run_per_comp = run_per (on_time, off_time)
#comp_run_perct = np.mean (run_per_comp[10])
comp_run_perct = 0.82
print('compressor run % calculated based on stable period', comp_run_perct)


#maximum possible defrost interval ∆tdmax 
del_tdmax = 39/comp_run_perct

## minimum possible defrost interval (∆tdmini)
del_tdmin = 6/comp_run_perct

## Actual defrost interval in data 
del_tdact = (Defrost_start[1] -Defrost_start[0] )/120
print ('Actual defrost interval in data  (∆tdact) : ' , del_tdact )

## Defrost interval for an ambient temperature of 32 °C (∆tdf)
del_tdf = (del_tdmax*del_tdmin)/(0.2*(del_tdmax-del_tdmin)+del_tdmin)
#del_tdf = 52
print('Defrost interval for an ambient temperature of 32 °C (∆tdf) : ', del_tdf)



## Steady state power for the selected defrost control cycle (PSS2)
PSS2 =  (Est_Int_Energy_xy-del_Edf) / Est_time_xy

print ('Steady state power for the selected defrost control cycle (PSS2)' , PSS2)

## Energy in Wh over a period of 24hr
E_daily = PSS2*24 + (del_Edf*24)/del_tdf

print ('Energy in Wh over a period of 24hr (E_daily)' , E_daily)


## Energy in Wh over a period of year (unit)
E_daily_year = E_daily*0.365

print ('Energy in Wh over a period of year (unit) (E_daily_year)' , E_daily_year)

## steady state temperature in compartment F that occurs in the whole test period used for SS2 in degrees C;						
Tss2_F =  Freezer_avg_xy-   (del_Thdf_F/Est_time_xy)
print ('steady state temperature in compartment F that occurs in the whole test period used for SS2' , Tss2_F)
#(del_Thdf_F/Est_time_xy)

#steady state temperature in compartment R that occurs in the whole test period used for SS2 in degrees C;						
Tss2_R =  Fridge_avg_xy-   (del_Thdf_R/Est_time_xy)
print ('steady state temperature in compartment R that occurs in the whole test period used for SS2' , Tss2_R)

#T_favg	Average temperature for the compartment F over a complete defrost control cycle
T_favg = Tss2_F + (del_Thdf_F/del_tdf)
print ('Average temperature for the compartment F over a complete defrost control cycle :', T_favg)

#T_ravg	Average temperature for the compartment R over a complete defrost control cycle
T_ravg = Tss2_R + (del_Thdf_R/del_tdf)
print ('Average temperature for the compartment R over a complete defrost control cycle :', T_ravg)

print('X-Y period Energy ', Est_Int_Energy_xy)
print('X-Y period Time', Est_time_xy)
print ('for more detail check upper code')
H24_energy  = Est_Int_Energy_xy*24/Est_time_xy
print('Daily Wh energy comsumption based derost to defrost data, consider one defrost average data in 24hr', H24_energy)
print ('Calculated Daily Wh energy comsumption based on all factors', E_daily)

## Percentage increase in energy due to defrost factor
Percentage_increase_energy = (E_daily-H24_energy)*100 /H24_energy

print ('Percentage increase in energy due to defrost factor :',Percentage_increase_energy,'%' )

## Average Freezer and fridge Temperature increase 
print('Average Freezer Temperature at end up  X Period to end up Y period; ', Freezer_avg_xy)
print('Average Fridge Temperature at end up  X Period to end up Y period; ', Fridge_avg_xy)

## Calculated Average Freezer and fridge Temperature

print('Actual Freezer Temperature increase at end up  X Period to end up Y period; ', T_favg)
print('Actual Freezer Temperature increase at end up  X Period to end up Y period; ', T_ravg)

## Percentage change in F & R Temperature 

Percentage_increase_F_Temp = (Freezer_avg_xy-T_favg)*100 /Freezer_avg_xy
Percentage_increase_R_Temp = (T_ravg-Fridge_avg_xy)*100 /Fridge_avg_xy

print ('Percentage increase in Favg Temperature due to defrost factor :',Percentage_increase_F_Temp,'%' )
print ('Percentage increase in Favg Temperature due to defrost factor :',Percentage_increase_R_Temp,'%' )
















