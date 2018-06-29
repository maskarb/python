def drought_stage(month):
    return {
        1 : [40, 30, 25],
        2 : [50, 35, 25],
        3 : [65, 45, 30],
        4 : [85, 60, 35],
        5 : [75, 55, 35],
        6 : [65, 45, 30],
        7 : [55, 45, 25],
        8 : [50, 40, 25],
        9 : [45, 35, 25],
        10: [40, 30, 25],
        11: [35, 30, 25],
        12: [35, 30, 25],
    }[month]

def recission(month):
    return {
        1 : [60,  50, 45],
        2 : [70,  55, 45],
        3 : [85,  65, 50],
        4 : [100, 80, 55],
        5 : [95,  75, 55],
        6 : [85,  65, 50],
        7 : [75,  65, 45],
        8 : [70,  60, 45],
        9 : [65,  55, 45],
        10: [60,  50, 45],
        11: [55,  50, 45],
        12: [55,  50, 45],
    }
'''
stage1 = [False] * len(file_dict[filename])
stage2 = [False] * len(file_dict[filename])
stage3 = [False] * len(file_dict[filename])

for i in range(len(file_dict[filename])):
    if storage[i] <= drought_stage(months[i])[0]:
        stage1[i] = True
    if storage[i] <= drought_stage(months[i])[1]:
        stage2[i] = True
    if storage[i] <= drought_stage(months[i])[2]:
        stage3[i] = True
file_dict[filename]['stage1'] = stage1
file_dict[filename]['stage2'] = stage2
file_dict[filename]['stage3'] = stage3
print(file_dict[filename])
'''