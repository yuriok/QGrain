# How to fit the samples

By this, please make sure you have loaded the grain size distribution (GSD) data correctly.

If everything goes well, you will see the interface like this. By default, QGrain has fitted the first sample automatically.

![App Appearance With Data Loaded](../figures/app_appearance_with_data_loaded.png)

If you are confused to some widgets, you can hover on it to see the tips.

## Workflow

The workflow of fitting samples is that:

1. Try fit one typic sample untill you are satisfied.

    You can adjust the component number and watch the chart of fitting result to find a proper value.

    If it can not return a correct result, you can check the **Ovserve Iteration** option to find the reason. The progress of fitting will be displayed by **Loss Canvas** and **Distribution Canvas**. Also, you can drag the lines to test whether the component number is proper.

    If it can return a proper result by giving the expected mean values, you can adjust the algorithm settings to refine the performance to let it can get the proper result automatically.

2. Test other samples with the component number.
3. If the component number are suitable for all samples, use auto fit to process them all.
4. If some results are not correct, cancel the fitting and return the step 1. If the incorrect results are not too many, you can fit and record manually.
5. Click the **Save** option of **File** menu to save the fitting results to file.
