from fpgrowth_py import fpgrowth
import numpy as np


def get_non_fp_indices(fp2indices_dict, x_train, x_test):
    ''' get indices that do not have any FPs '''
    fp_indices_train, fp_indices_test = set(), set()
    all_indices_train, all_indices_test = set(list(range(len(x_train)))), set(list(range(len(x_test))))

    fp2indices_dict_train = fp2indices_dict['train']
    fp2indices_dict_test = fp2indices_dict['test']

    for fp in fp2indices_dict_train:
        for idx in fp2indices_dict_train[fp]:
            fp_indices_train.add(idx)

    for fp in fp2indices_dict_test:
        for idx in fp2indices_dict_test[fp]:
            fp_indices_test.add(idx)

    non_fp_indices_train = all_indices_train.difference(fp_indices_train)
    non_fp_indices_test  = all_indices_test.difference(fp_indices_test)

    return list(non_fp_indices_train), list(non_fp_indices_test)


def get_fp_indices_raw(fps, df):
    col_types = df.dtypes
    indices_who_has_fp = []
    for index, row in df.iterrows():
        add_index = True
        for curr_fp in fps:
            col_name = curr_fp.split('=')[0]
            if col_types[col_name] == np.float64:
                col_val = float(curr_fp.split('=')[1])
            else:
                col_val = int(curr_fp.split('=')[1])

            if row[col_name] == col_val:
                pass
            else:
                add_index = False
                break

        if add_index:
            indices_who_has_fp.append(index)

    return indices_who_has_fp


def get_fp_indices(fps, cols_meta, df):
    ''' get indices of the rows that contain the frequent pattern fp '''

    def get_bounds(col, lower, upper):
        main_dict = cols_meta[col]
        percentiles = list(main_dict.keys())
        percentiles_pairs = list(zip(percentiles, percentiles[1:]))
        for pair in percentiles_pairs:
            if main_dict[pair[0]] == float(lower) and main_dict[pair[1]] == float(upper):
                return [pair[0], pair[1]]

    col_names, lower_bounds, upper_bounds = [], [], []
    # fps may be a frequent pattern that has more that one element
    for fp in fps:
        col = fp.split('<')[1]
        lower = fp.split('<')[0]
        upper = fp.split('<')[2]

        lb, ub = get_bounds(col, lower, upper)

        # add to the list of current fps
        col_names.append(col)
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    indices_who_has_fp = []
    for index, row in df.iterrows():
        add_index = True
        for i, col in enumerate(col_names):
            valuec = row[col]  # current value
            lbc, ubc = lower_bounds[i], upper_bounds[i]  # lower bound current, upper bound current
            if lbc == '75th':
                if cols_meta[col][lbc] <= valuec <= cols_meta[col][ubc]:
                    pass
                else:
                    add_index = False
                    break
            else:
                if cols_meta[col][lbc] <= valuec < cols_meta[col][ubc]:
                    pass
                else:
                    add_index = False
                    break
        if add_index:
            indices_who_has_fp.append(index)

    # return df[df.index.isin(indices_who_has_fp)]
    return indices_who_has_fp


def identify_frequent_patterns(df, target_variable, supp_fp):
    # inputs needed
    itemSetList = []
    df_cols = list(df.columns)
    df_cols.remove(target_variable)

    #  get the 25th, 50th, and 75th quartiles of each column
    cols_meta = {}

    for col in df_cols:
        cols_meta[col] = {
            'min': df[col].min(),
            '25th': df[col].quantile(0.25),
            '50th': df[col].quantile(0.50),
            '75th': df[col].quantile(0.75),
            'max': df[col].max()
        }

        keys = list(cols_meta[col].keys())
        values = list(cols_meta[col].values())
        keys_to_delete = []
        for i in range(len(values)-1):
            if values[i] == values[i+1]:
                keys_to_delete.append(keys[i])

        if keys_to_delete:
            for k in keys_to_delete:
                del cols_meta[col][k]

    # use these quantiles for categorizing data
    for index, row in df.iterrows():
        curr_items = []
        for col in df_cols:
            percentiles = list(cols_meta[col].keys())
            percentiles_pairs = list(zip(percentiles, percentiles[1:]))
            for pair in percentiles_pairs:
                if pair[1] != 'max':
                    if cols_meta[col][pair[0]] <= row[col] < cols_meta[col][pair[1]]:
                        curr_items.append('{}<{}<{}'.format(cols_meta[col][pair[0]], col, cols_meta[col][pair[1]]))
                        break
                else:
                    curr_items.append(
                        '{}<{}<{}'.format(cols_meta[col][pair[0]], col, cols_meta[col][pair[1]]))

        itemSetList.append(curr_items)

    # get the frequent patterns -- list of sets
    freqItemSet, rules = fpgrowth(itemSetList, minSupRatio=supp_fp, minConf=supp_fp)
    if freqItemSet:
        print('Frequent patterns: ')
        for fp in freqItemSet:
            print(fp)
    return freqItemSet, cols_meta