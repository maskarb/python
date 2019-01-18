import csv

from lookup_elev import els_stor

def read_csv_headers(filename):
    '''Read a csv file, return headers and data in lists'''
    with open(filename, 'r') as out:
        data_csv_reader = csv.reader(out)
        headers = next(data_csv_reader, None)
        raw_data = [row for row in data_csv_reader]
    return headers, raw_data

def get_variable_index(vars: list, headers: list):
    return [headers.index(var) for var in vars]

def main():
    headers, raw_data = read_csv_headers('fisher_analysis/historical_falls_lake.csv')
    ind = get_variable_index(['endElev'], headers)
    storage = [['storage']]
    for row in raw_data:
        elevation = float(row[ind[0]])
        storage.append([els_stor[elevation]])

    with open('historical_storage.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(storage)

if __name__ == '__main__':
    main()