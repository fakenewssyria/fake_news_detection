# Feature Selection ([feature_selection_code.py](https://github.com/fakenewssyria/fake_news_detection/blob/master/feature_selection/feature_selection_code.py))
This Repository contains the necessary codes for applying feature selection.

Modify lines 17/18 in order to control the number of features to be selected in each feature selection method


## Feature Selection Methods:

  * ```drop_zero_std()```: drops columns that have 0 standard deviation. (Actually will not drop but show the 
    columns that must be dropped)

  * ```drop_low_var()```: drops columns that have low variance. (Actually will not drop but show the 
    columns that must be dropped)

  * ```drop_high_correlation()```: drops columns that have high correlation. (Actually will not drop but show the 
    columns that must be dropped)

   * ```feature_importance(xg_boost=True, extra_trees=False)```: Applies feature importance to the data.
        * ```xg_boost=True, extra_trees=False```: will perform feature importance using XG Boost only
        * ```xg_boost=False, extra_trees=True```: will perform feature importance using Extra Trees only
        * ```xg_boost=True, extra_trees=True```: will perform feature importance using both XG Boost and Extra Trees
        * ```xg_boost=False, extra_trees=False```: Nothing will happen. Avoid this if you want to use feature selection.
        * **Default Behavior:** i.e. if we do: feature_importance() it will do only XG Boost. As: xg_boost=False, extra_trees=True

   * ```univariate()```: Applies Univariate Feature Selection with NUM_FEATURES_UNIVARIATE being selected (specified in line 17 in feature_selection_code.py). Raises Vlaue Error in this is greater than the total number of input features.

   * ```rfe()```:  Applies Recursive Feature Elimination with NUM_FEATURES_RFE being selected (specified in line 18 in feature_selection_code.py). Raises Vlaue Error in this is greater than the total number of input features.
   
## Scaling
  * ```scale=True, scale_input=True, scale_output=True```: Will scale both the input and the output columns.
  * ```scale=True, scale_input=True, scale_output=False```: Will scale only the input columns.
  * ```scale=True, scale_input=False, scale_output=True```: Will scale only the output column.
  * ```scale=True, scale_input=False, scale_output=False```: Will not scale any columns, although scale=True, but either scale_input or scale_output must be True
  * ```scale=False, scale_input=True, scale_output=True```: Will not scale any columns, although both scale_input=True and scale_output=True, but scale must be True as well in order to perform any scaling exercise.
  * ```scale=False, scale_input=True, scale_output=False```: Will not scale any columns, although scale_input=True, but scale must be True as well in order to perform any scaling exercise.
  * ```scale=False, scale_input=False, scale_output=True```: Will not scale any columns, although scale_output=True, but scale must be True as well in order to perform any scaling exercise.
  * ```scale=False, scale_input=False, scale_output=False```:  Will not scale any columns. 

## Indexing Input Columns in order to Scale
If ```scale=True and scale_input=True```:
  * ```input_zscore=(start_index_1, end_index_1)```: will apply Z-score scaling for the input columns starting at index start_index_1 and ending at end_index_1 (eclusive). By defualt, None. If None, no z-score scaling to any of the input columns is applied.
  * ```input_minmax=(start_index_2, end_index_2)```: will apply min-max scaling for the input columns starting at index start_index_2 and ending at end_index_2 (eclusive). By defualt, None. If None, no min-max scaling to any of the input columns is applied.
  * ```input_box=(start_index_3, end_index_3)```: will apply box-cox transformation for the input columns starting at index start_index_3 and ending at end_index_3 (eclusive). By defualt, None. If None, no box-cox transformation to any of the input columns is applied.
  * ```input_log=(start_index_4, end_index_4)```: will apply log transformation for the input columns starting at index start_index_4 and ending at end_index_4 (eclusive). By defualt, None. If None, no log transformation to any of the input columns is applied.


## Specifying Scaling Type to the Output Column
If ```scale=True and scale_output=True```:
  * ```output_zscore```: Boolean, by default, False. If True, Z-score scaling for the output column will be applied.
  * ```output_minmax```: Boolean, by default, False. If True, min-max scaling for the output column will be applied.
  * ```output_box```: Boolean, by default, False. If True, box-cox transformation for the output column will be applied.
  * ```output_log```: Boolean, by default, False. If True, log transformation for the output column will be applied.

Note: Either one of the above must be True, and all others must be False because we have to apply only one kind of scaling for the output column.

## Saving Feature Selection Plots
   * ```output_folder```: the path to the output folder that will be holding several modeling plots. If the path specified does not exist, it will be created dynamically at runtime.

## Regression vs. Classification   

If Regression, then specify ```regression=True```. If Classification, then specify ```regression=False```
    
 
## Columns

  * ```cols_drop```: list containing the names of the columns the user wants to drop from the data. By default, None. If None, no columns will be dropped from the data.
  * ```target_variable```: name of the column holding the target variable (this will be the output column)


## Raises
ValueError
  * If ```NUM_FEATURES_UNIVARIATE``` is greater than the total number of input features.
  * If ```NUM_FEATURES_RFE``` is greater than the total number of input features.
