import csv
from ast import literal_eval

def get_variable_index(vars: list, headers: list):
    return [headers.index(var) for var in vars]

def read_csv_headers(filename):
    '''Read a csv file, return headers and data in lists'''
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        headers = next(data_csv_reader, None)
        raw_data = [row for row in data_csv_reader]
    return headers, raw_data

f_name = 'res-s-0.4-0.csv'

headers, Data = read_csv_headers(f_name)
var_list = ['storage', ]
index_list = get_variable_index(var_list, headers)

data_num = []
for i, row in enumerate(Data):
    data_num.append([literal_eval(row[i]) for i in index_list])


print(data_num)
