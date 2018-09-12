import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

plt.style.use('ggplot')
np.random.seed(123)


def get_data():

        df = pd.read_csv('carlow_weather.csv', parse_dates=['date'])
        df['maxtp'] = df.maxtp.apply(lambda x: 0 if x ==' ' else float(x))
        df.index = df.date

        y = df.maxtp.resample('1W').mean().values
        u = df.maxtp.resample('1W').std().values

        m = ~np.isnan(y)

        return np.arange(len(y))[m].reshape(-1,1), y[m], u[m]


def rbf(x, m, alpha=10):
        return np.exp(-1 /(2 * alpha) * (x - m)**2)


def make_design_matrix(x):

        wk_of_year = x % 52

        X_d = np.zeros((len(x), 52))
        for i, xi in enumerate(wk_of_year):
                X_d[i, :] = rbf(xi, np.arange(0, 52))

        return X_d


def fit_SVR(x_train, y_train, x_test):
        model = SVR(kernel='rbf', C=1, epsilon=1)
        model.fit(x_train % 52, y_train)

        return model.predict(x_test % 52)


def fit_linear(x_train, y_train, x_test):
        model = Ridge(alpha=20)
        model.fit(make_design_matrix(x_train), y_train)

        return model.predict(make_design_matrix(x_test))


if __name__ == '__main__':

        x, y, u = get_data()

        x_train, y_train, u_train = x[:-50], y[:-50], u[:-50]
        x_test, y_test, u_test = x[-50:], y[-50:], u[-50:]

        X_d = make_design_matrix(x_train)
        yhat = fit_linear(x_train, y_train, x)


        plt.errorbar(x_train, y_train, yerr=u_train, fmt='.b', alpha=0.5)
        plt.errorbar(x_test, y_test, yerr=u_test, fmt='.r', alpha=0.5)

        plt.plot(x, yhat, 'k')

        plt.show()
