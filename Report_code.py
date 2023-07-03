import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import csv as csv
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


## import the data
## transform the data so that there is a column for date and column for Total, booked, Claims and column for GDP 
df = pd.read_csv('data.csv') #use pandas to read the csv
listOfDates= list(df.loc[:,"Q3 2001":"Q3 2022"].columns)
ndf = df.melt(id_vars=["Claims and deposits","Country of non-resident"],value_vars=listOfDates,var_name='QuarterDate',value_name='Indicator Value')
qs = ndf['QuarterDate'].str.replace(r'(Q\d) (\d+)', r'\2-\1')
ndf['date'] = pd.PeriodIndex(qs, freq='Q').to_timestamp()
keyRows= ["Total, booked, Claims", "Total, booked, Deposits" ,"GDP"]
tempDf= ndf.loc[ndf['Country of non-resident'].isin(keyRows)]
tempDf = tempDf.pivot(index='date', columns='Country of non-resident', values='Indicator Value').reset_index()
tempDf = tempDf.replace(',', '', regex=True)

dates = tempDf.loc[ : ,'date']
error_list = []
error_small_list = []
error_date_list = []
variance_list = []
variance_small_list = []
variance_date_list = []
for i in range(0,10000):
        
    ## create the multi linear regression model
    #mask the data
    mk = np.random.rand(len(dates)) < 0.8 ##80% of the data no bias
    train = tempDf[mk] #80% of the data
    test = tempDf[~mk]

    #mask the data using dates
    mask_1 = (dates >= '2001-01-01') & (dates <= '2007-12-31')
    mask_2 = (dates >= '2009-01-01') & (dates <= '2011-12-31')
    mask_3 = (dates >= '2013-01-01') & (dates <= '2019-12-31')
    mask = mask_1 | mask_2 | mask_3 #create a mask for the data 

    train_date = tempDf[mask] #15% of the data
    test_date = tempDf[~mask]

    #mask the data using random sample with 15% of the data
    smallmask = np.random.rand(len(dates)) < 0.15
    train_small = tempDf[smallmask] #15% of the data
    test_small = tempDf[~smallmask]

    #make x and y
    x_rand = np.asanyarray(train[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_rand = np.asanyarray(train[['GDP']]).astype(float)

    x_date = np.asanyarray(train[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_date = np.asanyarray(train[['GDP']]).astype(float)

    x_small = np.asanyarray(train[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_small = np.asanyarray(train[['GDP']]).astype(float)


    #train the model
    regr = LinearRegression()
    regr.fit (x_rand, y_rand)
    # The coefficients
    print ('random sample Coefficients: ', regr.coef_)
    print ('random sample Intercept: ',regr.intercept_)
    #predict the model
    y_hat_rand= regr.predict(test[['Total, booked, Claims','Total, booked, Deposits']]).astype(float) #predict the model for the random sample of 80% of the data
    x_rand_test = np.asanyarray(test[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_rand_test = np.asanyarray(test[['GDP']]).astype(float)

    y_hat_date = regr.predict(test_date[['Total, booked, Claims','Total, booked, Deposits']]).astype(float) #predict the model for the random sample for certain dates
    x_date_test = np.asanyarray(test_date[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_date_test = np.asanyarray(test_date[['GDP']]).astype(float)


    y_hat_small= regr.predict(test_small[['Total, booked, Claims','Total, booked, Deposits']]).astype(float) #predict the model for the random sample of 15% of the data
    x_small_test = np.asanyarray(test_small[['Total, booked, Claims','Total, booked, Deposits']]).astype(float)
    y_small_test = np.asanyarray(test_small[['GDP']]).astype(float)
    #calculate the error
    percent_error = np.mean(np.abs((y_rand_test - y_hat_rand) / y_rand_test)) * 100
    print('random sample Percent error: %.2f' % percent_error)
    error_small = np.mean(np.abs((y_small_test - y_hat_small) / y_small_test)) * 100
    print('small sample Percent error: %.2f' % error_small)
    error_date = np.mean(np.abs((y_date_test - y_hat_date) / y_date_test)) * 100
    print('date sample Percent error: %.2f' % error_date)

    # Explained variance score: 1 is perfect prediction
    print('random sample Variance score: %.2f' % regr.score(x_rand_test, y_rand_test))
    print('small sample Variance score: %.2f' % regr.score(x_small_test, y_small_test))
    print('date sample Variance score: %.2f' % regr.score(x_date_test, y_date_test))
    error_list.append(percent_error)
    error_small_list.append(error_small)
    error_date_list.append(error_date)
    variance_list.append(regr.score(x_rand_test, y_rand_test))
    variance_small_list.append(regr.score(x_small_test, y_small_test))
    variance_date_list.append(regr.score(x_date_test, y_date_test))


average_error = np.mean(error_list)
print('average error: %.2f' % average_error)
average_error_small = np.mean(error_small_list)
print('average error small: %.2f' % average_error_small)
average_error_date = np.mean(error_date_list)
print('average error date: %.2f' % average_error_date)
average_variance = np.mean(variance_list)
print('average variance: %.2f' % average_variance)
average_variance_small = np.mean(variance_small_list)
print('average variance small: %.2f' % average_variance_small)
average_variance_date = np.mean(variance_date_list)
print('average variance date: %.2f' % average_variance_date)


    

#plot the model for the random sample of 80% of the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_rand[:, 0], x_rand[:, 1], y_rand, c='r', marker='o', alpha=0.5)
ax.scatter(x_rand_test[:, 0], x_rand_test[:, 1], y_hat_rand, c='#340599', marker='o', alpha=0.5)
x_plot_rand = np.array([np.min(x_rand_test[:,0]), np.max(x_rand_test[:,0])])
y_plot_rand = np.array([np.min(x_rand_test[:,1]), np.max(x_rand_test[:,1])])
z_plot_rand = regr.intercept_[0] + regr.coef_[0][0]*x_plot_rand + regr.coef_[0][1]*y_plot_rand
ax.plot(x_plot_rand, y_plot_rand, z_plot_rand, '#350995')
ax.set_xlabel('Total, booked, Claims')
ax.set_ylabel('Total, booked, Deposits')
ax.set_zlabel('GDP')
ax.set_title('Linear Regression: random sample of data')
ax.legend(['Actual', 'Predicted', 'Linear Regression'])



##plot the data for the certian dates 
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(x_date[:, 0], x_date[:, 1], y_date, c='r', marker='o', alpha=0.5)
ax.scatter(x_date_test[:, 0], x_date_test[:, 1], y_hat_date, c='#a314db', marker='o', alpha=0.5)
x_plot_date = np.array([np.min(x_date_test[:,0]), np.max(x_date_test[:,0])])
y_plot_date = np.array([np.min(x_date_test[:,1]), np.max(x_date_test[:,1])])
z_plot_date = regr.intercept_[0] + regr.coef_[0][0]*x_plot_date + regr.coef_[0][1]*y_plot_date
ax.plot(x_plot_date, y_plot_date, z_plot_date, '#a511df')
ax.set_xlabel('Total, booked, Claims')
ax.set_ylabel('Total, booked, Deposits')
ax.set_zlabel('GDP')
ax.set_title('Linear Regression: Good fiscal years to train and bad fiscal years to test')
ax.legend(['Actual', 'Predicted', 'Linear Regression'])


# #plot the data for the random sample of 15% of the data
fig3 = plt.figure()
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(x_small[:, 0], x_small[:, 1], y_small, c='r', marker='o', alpha=0.5)
ax.scatter(x_small_test[:, 0], x_small_test[:, 1], y_hat_small, c='#9e3f00', marker='o', alpha=0.5)
x_plot_small = np.array([np.min(x_small_test[:,0]), np.max(x_small_test[:,0])])
y_plot_small = np.array([np.min(x_small_test[:,1]), np.max(x_small_test[:,1])])
z_plot_small = regr.intercept_[0] + regr.coef_[0][0]*x_plot_small + regr.coef_[0][1]*y_plot_small
ax.plot(x_plot_small, y_plot_small, z_plot_small, '#ab551c')
ax.set_xlabel('Total, booked, Claims')
ax.set_ylabel('Total, booked, Deposits')
ax.set_zlabel('GDP')
ax.set_title('Linear Regression: small random sample of data')
ax.legend(['Actual', 'Predicted', 'Linear Regression'])


##create a plot with all three models on it
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(x_rand[:, 0], x_rand[:, 1], y_rand, c='r', marker='o', alpha=0.5)
ax.scatter(x_rand_test[:, 0], x_rand_test[:, 1], y_hat_rand, c='#340599', marker='o', alpha=0.5)
ax.plot(x_plot_rand, y_plot_rand, z_plot_rand, '#350995')
ax.scatter(x_small[:, 0], x_small[:, 1], y_small, c='r', marker='o', alpha=0.5)
ax.scatter(x_small_test[:, 0], x_small_test[:, 1], y_hat_small, c='#9e3f00', marker='o', alpha=0.5)
ax.plot(x_plot_small, y_plot_small, z_plot_small, '#ab551c')
ax.scatter(x_date[:, 0], x_date[:, 1], y_date, c='r', marker='o', alpha=0.5)
ax.scatter(x_date_test[:, 0], x_date_test[:, 1], y_hat_date, c='#a314db', marker='o', alpha=0.5)
ax.plot(x_plot_date, y_plot_date, z_plot_date, '#a511df')
ax.set_xlabel('Total, booked, Claims')
ax.set_ylabel('Total, booked, Deposits')
ax.set_zlabel('GDP')
ax.set_title('Linear Regression: all three models')
ax.legend(['Actual', 'Predict with 80% mask', 'Linear Regression for random sample',
           'Actual','Predict with 15% mask', 'Linear Regression for small sample',
           'Actual','Trained with good fiscal years', 'Linear Regression for date sample'],
            bbox_to_anchor=(-0.5, 0.25, 0.5, 0.5), borderaxespad=0, fontsize=8, )

plt.show()

