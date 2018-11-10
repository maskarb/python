import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def size_of_state(Data_num, window_size):
    df=pd.DataFrame(Data_num)
    sos=[]
    for j in range(len(df.columns)):
        sos_temp=[]
        for i in df.index:
            A=list(df[j][i:i+window_size])
            A_1=[float(i) for i in A if i!=0 ]
            if len(A_1)==window_size:
                sos_temp.append(np.std(A_1,ddof=1))
        if len(sos_temp)==0:
            sos.append(0)
        else:
            sos.append(min(sos_temp)*2)
    return sos


def fisher(Data_num, Time, w_size, w_incr, sost):
    FI_final=[]
    k_init=[]
    for i in range(0,len(Data_num),w_incr):
        print(i)

        Data_win = Data_num[i:i+w_size]
        win_number=i

        if len(Data_win)==w_size:
            Bin=[]
            for m in range(len( Data_win )):
                Bin_temp=[]
                
                for n in range(len( Data_win )):
                    if m==n:
                        Bin_temp.append('I')
                    else:
                        Bin_temp_1=[]

                        for k in range(len(Data_win[n])):
                            if (abs(Data_win[m][k]-Data_win[n][k]))<=sost[k]:
                                Bin_temp_1.append(1)
                            else:
                                Bin_temp_1.append(0)
                                
                        Bin_temp.append(sum(Bin_temp_1))
                        
                Bin.append(Bin_temp)
            
            FI=[]
            for tl in range(1,101):
                tl1=len(sost)*float(tl)/100
                Bin_1=[]
                Bin_2=[]
                
                for j in range(len(Bin)):
                    if j not in Bin_2:
                       
                        Bin_1_temp=[j]
                        for i in range(len(Bin[j])):
                            if Bin[j][i]!='I' and Bin[j][i]>=tl1 and i not in Bin_2:
                                Bin_1_temp.append(i)
                                
                        Bin_1.append(Bin_1_temp)
                        Bin_2.extend(Bin_1_temp)
                    
                prob=[0]
                for i in Bin_1:
                    prob.append(float(len(i))/len(Bin_2))
                    
                prob.append(0)
                
                prob_q=[]
                for i in prob:
                    prob_q.append(np.sqrt(i))

                FI_temp=0
                for i in range(len(prob_q)-1):
                    FI_temp+=(prob_q[i]-prob_q[i+1])**2
                FI_temp=4*FI_temp    
                
                FI.append(FI_temp)
                
            for i in range(len(FI)):
                if FI[i]!=8.0:
                    k_init.append(FI.index(FI[i]))
                    break
            FI_final.append(FI)
    # return (k_init,FI_final)

    if len(k_init)==0:
        k_init.append(0)
    for i in range(0,len(FI_final)):
        FI_final[i].append(float(sum(FI_final[i][min(k_init):len(FI_final[i])]))/len(FI_final[i][min(k_init):len(FI_final[i])]))
        FI_final[i].append(Time[(i*w_incr+w_size)-1])
    
    df_FI=pd.DataFrame(FI_final)
    df_FI.to_csv("FI.csv",index=False,header=False)
    
    plt.plot(range(w_size,len(FI_final)+w_size), [i[-2] for i in FI_final ], 'b',label='FI')
    plt.ylim(0,8.5)
    plt.ylabel('Fisher Information')
    plt.xlabel('Time')
    plt.tight_layout()

    print("Fisher Done")
    return

def FI_smooth(f_name, step, step_win, xtick_step):
    out=open('FI.csv','r')
    data=csv.reader(out)
    Data=[]
        
    for row in data:
        Data.append(row)

    out.close()
    
    FI=[]
    time=[]
    
    for row in Data:
        FI.append(eval(row[-2]))
        time.append(row[-1])
        
    FI_smth=[]
    
    for i in range(step,len(FI)+step,step):
        for j in range(i-step,i):
            FI_smth.append(float(sum(FI[i-step:i]))/len(FI[i-step:i]))
            
    FI_smth=FI_smth[0:len(FI)]
            
    plt.plot(range(step_win,len(FI_smth)+step_win),FI_smth,'r',label='Smoothed')
    plt.xlabel('Time Step')
    plt.ylabel('Fisher Information')
    
    if xtick_step!='def':
        plt.xticks(range(step_win,len(FI_smth)+step_win,xtick_step),
                [time[i] for i in range(0,len(FI_smth),xtick_step)],rotation=75)
                
    else:
        plt.xticks(range(step_win,len(FI_smth)+step_win,3),
                [time[i] for i in range(0,len(FI_smth),3)],rotation=75)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(f_name+'_FI'+'.pdf')
    plt.savefig(f_name+'_FI'+'.png',dpi=1000)
    plt.close('all')
    
    for i in range(len(Data)):
        Data[i].append(FI_smth[i])
    
    data_final=[]
    for i in Data:
        data_temp=[i[-2],i[-3],i[-1]]
        data_final.append(data_temp)
    
    df_final=pd.DataFrame(data_final)
    df_final.columns=['Time_Step','FI','Smooth_FI']
    df_final.to_csv("%s_FI.csv"%(f_name))

    print("Smooth Done")
    return


def main():

    # location = "C:\\Users\\maska\\OneDrive\\Documents\\MASON\\20-jun-2018\\"
    # shift_list = [1.0]
    # number_runs = 1

    # for shift in shift_list:
    #     for run_num in range(0, number_runs):
    #         f_name = location+'reservoir-shift_{:.1f}-ts-{:d}_breakpoints.csv'.format(shift, run_num)

    f_name = 'python/rs1_ts1_bp.csv'
    df = pd.read_csv(f_name, header=0)
    Data_num = df[['storage']].values.tolist()

    smoothing = 6
    window_size = 4*12
    window_increment = 6

    Time = df['index']

    sost = size_of_state(Data_num, window_size)
    fisher(Data_num, Time, window_size, window_increment, sost)
    FI_smooth(f_name, smoothing, window_size, 'def')


if __name__ == '__main__':

    start = time.time()
    main()
    end = time.time()
    print(end - start)