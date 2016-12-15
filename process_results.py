# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.rcdefaults()
plt.rc('font', **{'size': 12})
plt.style.use('ggplot')

tuned_params = True
hpc_runs = True
with_median = True
# directories with experiment results
rslt_path = './results/'
rslt_path += 'hpc_runs/' if hpc_runs else 'local_runs/'
rslt_path += 'tuned_params/' if tuned_params else 'fixed_params/'

# list of file names, each file has experiment data for multiple datasets
# and also multiple runs in case of non deterministic experiments
algorithms = {'files': ["LS", "FS", "RS", "RS Ensemble",
                        "RS Ensemble - Bagging",
                        "RS Ensemble - Boosting by Weighting"],
              'abbrev': ["YK", "FS", "RS", "EnRS", "EnRS\nBagging",
                         "EnRS\nBoosting"]}
# DataFrame and CSV column names
cols = ['Alg_Name', 'Dataset', 'TrainingTime', 'TestingTime',
        'TrainingAccuracy', 'TestingAccuracy', 'TrainSize', 'TestSize',
        'TSLen', 'MinLen', 'MaxLen', 'EnsembleSize']
alg_col = cols[0]
ds_col = cols[1]
acc_col = cols[5]
rt_col = cols[2]


# read the experimental data from CSV files
def read_csv_data():
    print("Starting reading results files...")
    # DataFrame for experimental results
    data_df = pd.DataFrame(columns=cols)
    temp_series = pd.Series(index=[alg_col])
    for algo in algorithms['files']:
        file_path = rslt_path + algo + '.csv'
        experiment_data = pd.read_csv(file_path, header=0)
        print(algo)
        temp_series.loc[alg_col] = algo
        temp_df = pd.DataFrame([temp_series]*len(experiment_data))
        experiment_data = temp_df.join(experiment_data)
        data_df = data_df.append(experiment_data, ignore_index=True)
        if algo == 'LS':
            # create duplicate entries for std_dev calculations
            data_df = data_df.append(experiment_data, ignore_index=True)
    print("Files loaded.")
    return data_df


# Plots the figures for all datasets
def plot_dataset_results():
    print("Starting plotting...")
    ds_category = {'ecg': ['CinC_ECG_Torso', 'ECG200', 'ECG5000',
                           'ECGFiveDays', 'TwoLeadECG',
                           'NonInvasiveFatalECGThorax1',
                           'NonInvasiveFatalECGThorax2'],
                   'image': ['Adiac', 'ArrowHead', 'BeetleFly', 'BirdChicken',
                             'DiatomSizeReduction', 'DistalPhalanxTW',
                             'DistalPhalanxOutlineAgeGroup',
                             'DistalPhalanxOutlineCorrect', 'FaceAll',
                             'FaceFour', 'FacesUCR', '50words', 'FISH',
                             'HandOutlines', 'Herring', 'MedicalImages',
                             'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxTW',
                             'MiddlePhalanxOutlineCorrect', 'OSULeaf',
                             'PhalangesOutlinesCorrect',
                             'ProximalPhalanxOutlineAgeGroup',
                             'ProximalPhalanxOutlineCorrect',
                             'ProximalPhalanxTW', 'ShapesAll', 'SwedishLeaf',
                             'Symbols', 'WordsSynonyms', 'yoga'],
                   'motion': ['Cricket_X', 'Cricket_Y', 'Cricket_Z',
                              'Gun_Point', 'Haptics', 'InlineSkate',
                              'ToeSegmentation1', 'ToeSegmentation2',
                              'UWaveGestureLibraryAll',
                              'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y',
                              'uWaveGestureLibrary_Z', 'Worms',
                              'WormsTwoClass'],
                   'sensor': ['Car', 'Earthquakes', 'FordA', 'FordB',
                              'ItalyPowerDemand', 'InsectWingbeatSound',
                              'Lighting2', 'Lighting7', 'Phoneme',
                              'MoteStrain', 'Plane', 'SonyAIBORobotSurface',
                              'Trace', 'SonyAIBORobotSurfaceII',
                              'StarlightCurves', 'Wafer'],
                   'spectra': ['Beef', 'Coffee', 'Ham', 'Meat', 'OliveOil',
                               'Strawberry', 'Wine'],
                   'synthetic': ['CBF', 'ChlorineConcentration', 'MALLAT',
                                 'ShapeletSim', 'synthetic_control',
                                 'TwoPatterns']}
    opacity = 0.65
    # err_cfg = {'ecolor': '0.3'}
    fig_size = (6.67, 3.5)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    # Plot each dataset
    for d_set in ds_list:
        print(d_set)
        group = data_df.groupby(ds_col, sort=False).get_group(d_set)
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=fig_size,
                                       sharey=True)
        acc_exact = group[group[alg_col] == 'LS'][acc_col].mean()
        rt_exact = group[group[alg_col] == 'LS'][rt_col].mean()
        # Set the x axis limits as (min-3, max+3)
        ax0.set_xlim(left=np.min(group[acc_col].values)-3,
                     right=np.max(group[acc_col].values)+3)
        # draw the baseline accuracy and runtime
        ax0.axvline(x=acc_exact, label=algorithms['abbrev'][0], linewidth=1.5,
                    color='k')
        ax1.set_xscale('log')
        ax1.axvline(x=1, label=algorithms['abbrev'][0], linewidth=1.5,
                    color='k')
        # draw boxplots for different algos
        acc, rt = [], []
        for algo in algorithms['files'][1:]:
            acc.append(group[group[alg_col] == algo][acc_col].values)
            rt.append(rt_exact/group[group[alg_col] == algo][rt_col].values)
        boxes1 = ax1.boxplot(rt, vert=False, whis='range', patch_artist=True)
        boxes0 = ax0.boxplot(acc, vert=False, whis='range', patch_artist=True,
                             labels=algorithms['abbrev'][1:])
        # change boxplot properties
        for boxes in [boxes0, boxes1]:
            # change color and linewidth of the whiskers
            for whisker in boxes['whiskers']:
                whisker.set(linewidth=0)
            # change color and linewidth of the caps
            for cap in boxes['caps']:
                cap.set(color='#000000', linewidth=1.5)
            # change linewidth of medians
            for med in boxes['medians']:
                med.set(linewidth=1)
            # change box face colors and set opacity
            for box, clr in zip(boxes['boxes'], colors):
                box.set_facecolor(clr)
                box.set_alpha(opacity)
        for axis in [ax0, ax1]:
            axis.tick_params(axis='both',     # changes apply to both axes
                             which='both',    # both ticks are affected
                             direction='in')  # ticks move inside the plot
        plt.tight_layout()
        # find the category of the current dataset
        for key, val in ds_category.items():
            if d_set in val:
                break
        # create figure paths
        path = rslt_path + 'plots/'
        category_path = path + key + '/'
        # if directories don't exist, create them
        if not os.path.exists(path) or not os.path.exists(category_path):
            os.makedirs(path)
            os.makedirs(category_path)
        # save the figure
        plt.savefig(path + d_set + '.png', dpi=150, format='png')
        plt.savefig(category_path + d_set + '.png', dpi=150, format='png')
        plt.close()
    print("Finished plotting.")


# create result tables
def create_table(df, table_type='accuracy'):
    '''
    df: DataFrame with aggregated results
    table_type: string, possible values 'accuracy'/'time'
    '''
    print("Creating " + table_type + "table...")
    sub = 9 if not with_median else 0

    col = acc_col if table_type == 'accuracy' else rt_col
    # cols[2] == TrainingTime
    # cols[5] == 'TestingAccuracy

    values = []
    for algo in algorithms['files']:
        t = []
        group = df.groupby(alg_col, sort=False).get_group(algo)
        for ds in ds_list:
            t.append(group[group[ds_col] == ds][col].mean())
        values = values + [pd.Series(t)]

    values = tuple(((1/vals_col if table_type == 'accuracy' else vals_col)
                    for vals_col in values))

    values = np.vstack(values).T
    values = values.astype(float)
    for i in range(len(values)):
        values[i] = stats.rankdata(values[i])
    wins = (values == 1).astype(int)  # wins = wins*1
    wins = wins.sum(axis=0)
    avgranks = values.sum(axis=0)/len(values)

    path = rslt_path + 'tables/'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + table_type + '.csv', 'w') as csv, \
            open(path + table_type + '.txt', 'w') as dis, \
            open(path + table_type + '.tex', 'w') as tex:
        output_types = ['csv', 'tex', 'dis']

        sep = {'csv': ",", 'dis': " | ", 'tex': " & "}
        fill = {'csv': "", 'dis': " ", 'tex': r" $\pm$"}  # u"\u00B1"
        ds_name_field = {'csv': "{0}", 'dis': "{0: <28}", 'tex': "{0: <28}"}
        data_field = {'csv': "{0}{1}", 'dis': "", 'tex': ""}
        if table_type == 'accuracy':
            dis_border_mul = 21 - sub
            tex_border_mul = 26 - sub
            data_field['dis'] = "{0}{1: ^" + str(dis_border_mul) + "}"
            data_field['tex'] = "{0}{1: <" + str(tex_border_mul) + "}"
        else:
            dis_border_mul = 11
            tex_border_mul = 11
            data_field['dis'] = '{0}{1: ^' + str(dis_border_mul) + '}'
            data_field['tex'] = '{0}{1: <' + str(dis_border_mul) + '}'

        row = {'csv': '', 'tex': '', 'dis': ''}

        for f_typ in output_types:
            row[f_typ] = ds_name_field[f_typ].format(ds_col)

        border = "_"*28

        for algo in algorithms['abbrev']:
            border += "___" + "_"*dis_border_mul
            for f_typ in output_types:
                row[f_typ] += data_field[f_typ].format(sep[f_typ],
                                                       algo.replace('\n', '-'))

        csv.write(row['csv'] + '\n')
        tex.write(row['tex'] + r'\\' + '\n')
        dis.write(row['dis'] + '\n' + border + '\n')
        print(row['dis'] + '\n' + border + '\n', end='')

        if table_type == 'accuracy':
            data_field['csv'] = '{0}{1:.2f}'
            data_field['tex'] = '{0}{1:6.2f}{2}{3:5.2f}' + \
                                (' ({4:6.2f})' if with_median else '')
            data_field['dis'] = '{0}{1:6.2f}{2}{3:5.2f}' + \
                                (' ({4:6.2f})' if with_median else '')
        else:
            data_field['csv'] = '{0}{1:.2f}'
            data_field['tex'] = '{0}{1:' + str(dis_border_mul) + '.1f}'
            data_field['dis'] = '{0}{1:' + str(tex_border_mul) + '.1f}'

        for d_set in ds_list:
            for f_typ in output_types:
                row[f_typ] = ds_name_field[f_typ].format(d_set)

            group = df.groupby(ds_col, sort=False).get_group(d_set)

            for algo in algorithms['files']:
                mean = group[group[alg_col] == algo][col].mean()
                stdv = group[group[alg_col] == algo][col].std()
                medn = group[group[alg_col] == algo][col].median()
                for f_typ in output_types:
                    row[f_typ] += data_field[f_typ].format(sep[f_typ], mean,
                                                           fill[f_typ], stdv,
                                                           medn)
            csv.write(row['csv'] + '\n')
            tex.write(row['tex'] + r'\\' + '\n')
            dis.write(row['dis'] + '\n')
            print(row['dis'] + '\n', end='')

        if table_type == 'accuracy':
            data_field['tex'] = '{0}{1: <' + str(tex_border_mul) + '.2f}'
            data_field['dis'] = '{0}{1: ^' + str(dis_border_mul) + '.2f}'
        else:
            data_field['tex'] = '{0}{1: <' + str(dis_border_mul) + '.2f}'
            data_field['dis'] = '{0}{1: ^' + str(tex_border_mul) + '.2f}'
        row['tex'] = ds_name_field['tex'].format('Average ranks:')
        row['dis'] = ds_name_field['dis'].format('Average ranks:')
        for r in avgranks:
            row['tex'] += data_field['tex'].format(sep['tex'], r)
            row['dis'] += data_field['dis'].format(sep['dis'], r)
        print(border + '\n' + row['dis'] + '\n', end='')
        tex.write(r'\\' + '\n\hline\n' + row['tex'] + r'\\' + '\n')
        dis.write(border + '\n' + row['dis'] + '\n' + border + '\n')

        if table_type == 'accuracy':
            data_field['tex'] = '{0}{1: <' + str(tex_border_mul) + 'd}'
            data_field['dis'] = '{0}{1: ^' + str(dis_border_mul) + 'd}'
        else:
            data_field['tex'] = '{0}{1:<' + str(dis_border_mul) + 'd}'
            data_field['dis'] = '{0}{1:^' + str(tex_border_mul) + 'd}'
        row['tex'] = ds_name_field['tex'].format('Wins:')
        row['dis'] = ds_name_field['dis'].format('Wins:')
        for w in wins:
            row['tex'] += data_field['tex'].format(sep['tex'], w)
            row['dis'] += data_field['dis'].format(sep['dis'], w)
        print(border + '\n' + row['dis'] + '\n', end='')
        tex.write(row['tex'] + '\n')
        dis.write(row['dis'] + '\n')
    print("Tables saved.")
# =============================================================================
if __name__ == '__main__':
    # DataFrame for experimental results
#    data_df = read_csv_data()
    # create a list of evaluated datasets
#    ds_list = data_df[ds_col].unique().tolist()
#    plot_dataset_results()
    create_table(data_df, table_type='accuracy')
    create_table(data_df, table_type='time')
