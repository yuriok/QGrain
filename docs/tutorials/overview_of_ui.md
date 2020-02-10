# Overview of UI

## The layout of app

QGrain consists of some docks which are movable, scalable, divisible, floatable, and closable. By default, QGrain puts the docks which have canvas in the top left corner, and all these docks are tabified. In order to control this app conveniently, the dock of `Control Panel` is in the bottom left corner alone. On the right, there are the docks to display the raw data and recorded results.

![App Appearance With Data Loaded](../figures/app_appearance_with_data_loaded.png)

If you are not satisfied with this default layout. You can adjust it as you please, the customized layout will be stored after you closed this app. If you want to reset the layout to default, you can click the **Reset** option of **Docks** menu. By double clicking the title or clicking the *Reset* button (top right corner of the dock) to float the dock. Drag the floating window above the other docks to tabify of split it. If you want to change the sizes of docks, you can drag the separator to adjust them. By clicking the *Close* button to close the needless docks. If you want to display a dock that has been closed before, you can click the **Docks** menu and select the corresponding option to realize it.

## Docks

### PCA Panel

The dock to do principal component analysis (PCA) (see the [wiki page](https://en.wikipedia.org/wiki/Principal_component_analysis) for more details). It contains one canvas to show the result and some widgets bellow to control the algorithm.

![Appearance of PCA Panel](../figures/pca_panel.png)

### Loss Canvas

The dock to dynamically display the process of loss (i.e. the difference between observation and fitting result) changing.

Note: Only take effect when the **Observe Iteration** option of **Control Panel** is checked.

![Appearance of Loss Canvas](../figures/loss_canvas.png)

### Distribution Cavas

The dock to dynamically display the distribution of raw data and fitting result of current sample.

![Appearance of Distribution Canvas](../figures/distribution_canvas.png)

For the dock which contains canvas, you can follow the following mouse interaction to adjust the figure.

* **Left button**: Interacts with items in the scene (select/move objects, etc). If there are no movable objects under the mouse cursor, then dragging with the left button will pan the scene instead.
* **Right button drag**: Scales the scene. Dragging left/right scales horizontally; dragging up/down scales vertically (although some scenes will have their x/y scales locked together). If there are x/y axes fisible in the scene, then right-dragging over the axis will _only_ affect that axis.
* **Right button click**: Clicking the right button in most cases will show a context menu with a variety of options depending on the object(s) under the mouse cursor.
* **Middle button (or wheel) drag**: Dragging the mouse with the wheel pressed down will always pan the scene (this is useful in instances where panning with the left button is prevented by other objects in the scene).
* **Wheel spin**: Zooms the scene in and out.

Moreover, you can check the `temp` folder of QGrain to get the files of canvas.
Note: If the size of `temp` folder is larger than 1 GB, it will be cleared when the app is initialized.

### Control Panel

The dock to control the fitting behaviours.

![Appearance of Control Panel](../figures/control_panel.png)

#### Tips

* Click the raido buttons of **Distribution Type** to switch the distribution function.
* Click the **+**/**-** button to add/reduce the component number you guess.
* **Observe Iteration**: Whether to display the iteration procedure.
* **Inherit Parameters**: Whether to inherit the parameters of last fitting. It will improve the accuracy and efficiency when the samples are continuous.
* **Auto Fit**: Whether to automaticlly fit after the sample data changed.
* **Auto Record**: Whether to automaticlly record the fitting result after fitting finished.
* Click the **Previous** button to back to the previous sample.
* Click the **Next** button to jump to the next sample.
* Click the **Auto Run Orderly** button to run the program automatically. The samples from current to the end will be processed one by one.
* Click the **Cancel** button to cancel the fitting progress.
* Click the **Try Fit** button to fit the current sample.
* Click the **Record** button to record the current fitting result.\nNote: It will record the LAST SUCCESS fitting result, NOT CURRENT SAMPLE.
* Click the **Multi Cores Fitting** button to fit all samples. It will utilize all cores of cpu to accelerate calculation.
* Move the lines in **Distribution Canvas** dock to set the expected mean values of each component, if it can not return a proper result and you make sure the component is correct.

### Raw Data Table

The dock to show the GSD data of samples.

By clicking the row of sample, you can set this sample as current sample and process it.

![Appearance of Raw Data Table](../figures/raw_data_table.png)

### Recorded Data Table

The dock to show the recorded fitting results.

![Appearance of Recorded Data Table](../figures/recorded_data_table.png)
