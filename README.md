Please refer to Report.pdf for description of project.

-------------------------------------------------------------------------------------
Code description:

Only the 'Daily use of a WF credit card' tab (daily-creditcard___detailCategory.csv)
of the data file is used.
Every customer's sequential purchase history has 4 features:
Date (time)
Des1 (categorical data)
Des2 (categorical data)
Payment (numerical data)

Des1, Des2 and Payment can be used as features for the data to build a Recurrent Neural Network model.
It is advisable to have larger datasets to better train a deep learning model.
Due to small dataset available, we limit ourselves to train only the Des2 feature.

'''ReadData.py
Reads sequential card usage data from
the 'daily-creditcard___detailCategory.csv' file.

'''rnn.py
Contains code for the Recurrent Neural Network model implementation.

--------------------------------------------------------------------------------------
Instructions to run the code: (Note: They are specific to LINUX)

1. Install python libraries 'numpy' , 'matplotlib' and 'tensorflow' through terminal using following commands:
-pip install numpy
-pip install matplotlib
-pip install tensorflow

(Make sure the above libraries are successfully installed)

2. Run the rnn.py file using command:
-python rnn.py

3. Enter masked_id of customer whose purchase you want to predict in the terminal prompt.


------------------------------------------------------------------------------------------
Results:

The output is the current, shifted and predicted values for current batch data.
Note: The predicts are not very good because the dataset is very small and also it is synthetic data with no purchase
patterns.

*IMPORTANT*
We also generate the loss computed for every batch run. The algorithm is successful in the prediction task as the error
values generally shows a decreasing trend after every run. (Visible in the plot generated).

