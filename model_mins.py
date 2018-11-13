import os
import pandas as pd

def list_csv_files(search_dir):
    """This function accepts a string representing the path to a directory 
    that will be searched for files with the extension ".csv". This function 
    will return a list of strings representing all of the file names in the 
    specified directory that end with the ".csv" extension. If the specified 
    directory does not contain any .csv files, the function will return an 
    empty list."""
    fname = os.listdir(search_dir)
    files = []
    for i in range(len(fname)):
        if fname[i].endswith('.csv'):
            files.append(fname[i])
    return files

def find_mins_and_dates(df):
    count = 0
    dates = []
    for i in range(1, len(df)-1):
        if df["Smooth_FI"][i] < df["Smooth_FI"][i-1] and df["Smooth_FI"][i] < df["Smooth_FI"][i+1]:
            count += 1
            dates.append(df["Time_Step"][i].strftime('%m-%Y'))
    return [count, dates]

def main():
    fi_list = pd.DataFrame()
    df_csv = pd.DataFrame()
    directory = "C:\\Users\\maska\\research\\0.7_shift"
    csv_files = list_csv_files(directory) # get all of the csv files in the directory
    for file_ in csv_files:
        print(file_)
        try:
            df = pd.read_csv(directory+'\\'+file_, usecols=["Time_Step", "Smooth_FI"], parse_dates=["Time_Step"])
        except:
            pass
        else:
            count, dates = find_mins_and_dates(df) # count strict minimums and get corresponding dates as list
            fname_comps = file_.split("_") # split filename to get hwin, winspace, smooth
            df_temp = pd.Series([fname_comps[3], fname_comps[5], fname_comps[7], count, dates], name=file_) 
            # df_temp contains hwin, winspace, smooth, num fo mins, and correpsonding dates
            df_csv = df_csv.append(df_temp)
            # df_csv appends all runs to one df. is written to file below.

            # fi_list contains all of the FIs
            new_temp = pd.Series(list(df["Smooth_FI"]), index=df["Time_Step"], name=fname_comps[1])
            # make giant list containing all of the runs' FI
            fi_list = fi_list.join(new_temp, how='right')

    df_csv.to_csv(directory+'\\MATRIX.csv', header=["hwin", "winspace", "smoothing", "num_mins", "dates"], 
                            index_label="filename")
    fi_list.to_csv(directory+'\\combined_runs_fi.csv', index_label='date')
    


if __name__ == '__main__':
    main()