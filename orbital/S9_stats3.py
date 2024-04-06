# %% Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Inputs


# Names for 0 thru 18/17
file_names_data = [
    '190612 Radial scans20190612_131606_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 0
    '190612 Radial scans20190612_131804_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 1
    '190612 Radial scans20190612_132024_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 2
    '190612 Radial scans20190612_132211_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 3
    '190612 Radial scans20190612_132925_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 4
    '190612 Radial scans20190612_133123_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 5
    '190612 Radial scans20190612_134206_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 6
    '190612 Radial scans20190612_134348_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 7
    '190612 Radial scans20190612_134630_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 8
    '190612 Radial scans20190612_134816_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 9
    '190612 Radial scans20190612_135826_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 10
    '190612 Radial scans20190612_140254_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 11
    '190612 Radial scans20190612_140610_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 12
    '190612 Radial scans20190612_140805_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 13
    '190612 Radial scans20190612_141724_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 14
    '190612 Radial scans20190612_141915_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 15
    '190612 Radial scans20190612_142218_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 16
    '190612 Radial scans20190612_142400_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 17
    '190612 Radial scans20190612_143612_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',  # 18
]

# Specify file path, sheet name
file_path1 = 'C:\\Users\\David\\Desktop\\Semester_9\\FPI_Lab\\Data_csv\\'
file_name_head = 'TVP File Template.xlsx'
file_path_head = file_path1 + file_name_head
file_sheet_head = 'Sheet1'

# Limits for Sweep Angle
limit_sweep = 22

# Remove probe angles
probes_to_remove = [6]


# Which plots
#which_plots = [1]
which_plots = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
#which_plots = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]




single_plot_num = 1
single_sweep_ang = 8





# %% Read Data

df_head = pd.read_excel(file_path_head, sheet_name=file_sheet_head, header=None)
col_head = df_head.values.tolist()[0]

probe_x = np.arange(-22,24,2)
probe_y_nested2 = []
probe_y_simple = [0]*len(probe_x)

single_sweep_num = np.where(probe_x == single_sweep_ang)[0][0]


single_list_short = []
single_list_long = []

# Gather and write Data as DataFrames, store into master list
df_list = [0] * len(file_names_data)
for i, plot_num in enumerate(which_plots):

    name = file_names_data[plot_num]

    file_path_data = file_path1 + name
    df_list[i] = pd.read_csv(file_path_data, lineterminator='\n', names=col_head)

    df_data = df_list[i]

    df_1 = df_data.copy()
    df_1.drop(df_data[df_data['Theta Angle (Degrees)'] >= limit_sweep].index, inplace=True)
    df_1.drop(df_data[df_data['Theta Angle (Degrees)'] <= -limit_sweep].index, inplace=True)

    angle_sweep = df_1[df_1.columns[0]].to_list()

    collector_bias = df_1[df_1.columns[24]]

    angle_probe = 22

    collector_list_nested = []
    angle_probe_nested = []
    angle_probe_list = []

    for j in range(1, 24):

        # Y (Probe angle) (2 deg of separation for all)
        angle_probe_nested.append([angle_probe] * len(angle_sweep))
        angle_probe_list.append(angle_probe)
        angle_probe -= 2

        # Z (amplitude)
        collector_list_raw = df_1[df_1.columns[j]]  # .reset_index()  # .to_list()
        #collector_list_raw = df_data[df_data.columns[j]]  # .reset_index()  # .to_list()

        collector_list = collector_bias - collector_list_raw
        # collector_list.reset_index()
        collector_list_nested.append(collector_list)


    collector_list_nested_reset = []
    for j, collector_series in enumerate(collector_list_nested):
        list_from_series = collector_series.to_list()
        collector_list_nested_reset.append(list_from_series)

    # Convert data to "3D" array
    for k, ang_probe in enumerate(angle_probe_nested):
        ang_p = ang_probe[0]
        for j, ang_s in enumerate(angle_sweep):
            col_val = collector_list_nested_reset[i][j]
            if k == 0 and j == 0:
                _3d_data = np.array([[ang_p, ang_s, col_val]])
            else:
                _3d_data = np.append(_3d_data, [[ang_p, ang_s, col_val]], axis=0)

    #probe_x = np.arange(-22,24,2)
    #probe_y =

    # Remove Collector Probes
    rows_to_remove = []

    for probe_val in probes_to_remove:
        for i, row in enumerate(_3d_data):
            if int(row[0]) == int(probe_val):
                rows_to_remove.append(i)  # Store rows to be deleted
    for row in reversed(rows_to_remove):
        _3d_data = np.delete(_3d_data, row, axis=0)  # Delete row

    swe_mat, pro_mat = np.meshgrid(angle_sweep, angle_probe_list)

    arr_in = [pro_mat, swe_mat]
    collector_array = np.array(collector_list_nested_reset)



    probe_y_nested = [0]*len(probe_x)

    # Loop through probe angle (-22 -> +22)
    for j, index in enumerate(probe_x):
        row_y = np.where(pro_mat == index)[0][0]
        test_list = []

        # Loop through each index
        for col in range(len(pro_mat[row_y])):
            val = collector_array[row_y][col]

            test_list.append(val)
            if probe_y_simple[j] == 0:
                probe_y_simple[j] = [val]
            else:
                probe_y_simple[j].append(val)


            if index == single_sweep_ang:
                single_list_long.append(val)
                if plot_num == single_plot_num:
                    single_list_short.append(val)


        probe_y_nested[j] = test_list


        #probe_y_nested.append(test_list)
        #print(test_list)

    probe_y_nested2.append(probe_y_nested)


    #probe_y = []
    #for indexes in probe_y

    """
    # iterate long
    for j in len(_3d_data):
        # iterate short
        for k in len(_3d_data[0]):
            value = np.where(a == -10)[0][0]

    """

# %% Grab lists from data
"""
mean_values = []
std_values = []
for i, item in enumerate(probe_y_simple):
    mean = np.mean(item)
    std = np.std(item)

    mean_values.append(mean)
    std_values.append(std)

    angle = probe_x[i]

    print(angle, mean, std)
"""

single_std_short = np.std(single_list_short)
single_std_long = np.std(single_list_long)



# %% Plotting

"""
fig1 = plt.figure(figsize=(10, 6))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.scatter(mean_values, std_values)
ax1.set(title='Mean vs. Standard Deviation', xlabel='Mean', ylabel='Standard Deviation')
ax1.grid(True)
"""


print('Plot Number', single_plot_num)
print('Sweep Angle', single_sweep_ang)

ax1_title = str('Single Dataset - Plot Number ' + str(single_plot_num))
#ax2_title = str('All Datasets' + single_sweep_ang)


fig1 = plt.figure(figsize=(10, 6))

ax1 = fig1.add_subplot(1,2,1)
ax1.hist(single_list_short)
ax1.set(title=ax1_title, xlabel=str('Collector Angle: ' + str(single_sweep_ang)), ylabel='Count')

ax2 = fig1.add_subplot(1,2,2)
ax2.hist(single_list_long)
ax2.set(title='All Datasets', xlabel=str('Collector Angle: ' + str(single_sweep_ang)), ylabel='Count')



# %%
plt.tight_layout()
plt.show()
