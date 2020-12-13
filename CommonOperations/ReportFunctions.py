from CommonOperations import Accumulation
from CommonOperations import FractionalAccumulation
from CommonOperations import NewInformationPriorityAccumulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as py


class Generate_report_ago():
    def __init__(self):
        pass

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            t_all = np.concatenate((t_train, t_test), axis=0)
            x_all = np.concatenate((x0_train, x0_test), axis=0)
            x_pred = mdl.predict(t_all)

            lens = len(mdl.x_orig)
            err = x_pred[:lens] - mdl.x_orig
            m = len(t_train)

            x0_train = x_all[:m]
            x0_test = x_all[m - 1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, x0_train)
            plt.plot(t_train, x_pred[:m])
            plt.plot(t_test, x0_test)
            plt.plot(t_test, x_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Generate_report_ago.generate_report(None, mdl, t_train, x0_train, t_test, x0_test)


class Generate_report_fago():
    def __init__(self):
        pass

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            t_all = np.concatenate((t_train, t_test), axis=0)
            x_all = np.concatenate((x0_train, x0_test), axis=0)
            x_pred = mdl.predict(t_all)

            lens = len(mdl.x_orig)
            err = x_pred[:lens] - mdl.x_orig
            m = len(t_train)

            x0_train = x_all[:m]
            x0_test = x_all[m - 1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, x0_train)
            plt.plot(t_train, x_pred[:m])
            plt.plot(t_test, x0_test)
            plt.plot(t_test, x_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Generate_report_fago.generate_report(None, mdl, t_train, x0_train, t_test, x0_test)

class Generate_report_nipago():
    def __init__(self):
        pass

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            t_all = np.concatenate((t_train, t_test), axis=0)
            x_all = np.concatenate((x0_train, x0_test), axis=0)
            x_pred = mdl.predict(t_all)

            lens = len(mdl.x_orig)
            err = x_pred[:lens] - mdl.x_orig
            m = len(t_train)

            x0_train = x_all[:m]
            x0_test = x_all[m - 1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, x0_train)
            plt.plot(t_train, x_pred[:m])
            plt.plot(t_test, x0_test)
            plt.plot(t_test, x_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Generate_report_ago.generate_report(None, mdl, t_train, x0_train, t_test, x0_test)


class Generate_report_agon():
    def __init__(self):
        pass

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        if (mdl.is_fitted == True):
            y_all = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            y_pred = mdl.predict(t_x)
            t_all = t_x[0:, 0]

            lens = len(mdl.x_orig)
            m = len(y_train)
            err = y_pred[:lens] - mdl.x_orig

            y0_train = y_all[:m]
            y0_test = y_all[m-1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, y0_train)
            plt.plot(t_train, y_pred[:m])
            plt.plot(t_test, y0_test)
            plt.plot(t_test, y_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            y1 = Accumulation.ago(y_all, None, True)
            y1_pred = Accumulation.ago(y_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': y1,
                'x1_pred': y1_pred,
                'x_pred': y_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            y = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            mdl.fit(y, t_x)
            Generate_report_agon.generate_report(None, mdl, y_train, t_x_train, y_test, t_x_test)


class Generate_report_fagon():
    def __init__(self):
        pass

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        if (mdl.is_fitted == True):
            y_all = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            y_pred = mdl.predict(t_x)
            t_all = t_x[0:, 0]

            lens = len(mdl.x_orig)
            m = len(y_train)
            err = y_pred[:lens] - mdl.x_orig

            y0_train = y_all[:m]
            y0_test = y_all[m-1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, y0_train)
            plt.plot(t_train, y_pred[:m])
            plt.plot(t_test, y0_test)
            plt.plot(t_test, y_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            y1 = FractionalAccumulation.fago(y_all, mdl.r)
            y1_pred = FractionalAccumulation.fago(y_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': y1,
                'x1_pred': y1_pred,
                'x_pred': y_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            y = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            mdl.fit(y, t_x)
            Generate_report_fagon.generate_report(None, mdl, y_train, t_x_train, y_test, t_x_test)


class Generate_report_nipagon():
    def __init__(self):
        pass

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        if (mdl.is_fitted == True):
            y_all = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            y_pred = mdl.predict(t_x)
            t_all = t_x[0:, 0]

            lens = len(mdl.x_orig)
            m = len(y_train)
            err = y_pred[:lens] - mdl.x_orig

            y0_train = y_all[:m]
            y0_test = y_all[m-1:]
            t_train = t_all[:m]
            t_test = t_all[m - 1:]

            rmse = np.sqrt(sum((err) ** 2 / lens))
            rmse = np.round(rmse, 4)

            t3 = np.arange(lens) + 1

            p = plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t_train, y0_train)
            plt.plot(t_train, y_pred[:m])
            plt.plot(t_test, y0_test)
            plt.plot(t_test, y_pred[m - 1:lens])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.legend(['x_train', 'x_pred_train', 'x_test', 'x_pred_test'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            y1 = NewInformationPriorityAccumulation.nipago(y_all, mdl.r)
            y1_pred = NewInformationPriorityAccumulation.nipago(y_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': y1,
                'x1_pred': y1_pred,
                'x_pred': y_pred,
                'error': err,
            })
            c1 = [p, model_report]
            return c1
        else:
            y = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            mdl.fit(y, t_x)
            Generate_report_nipagon.generate_report(None, mdl, y_train, t_x_train, y_test, t_x_test)




class Formula_expression_gm():
    def __init__(self):
        pass

    def formula_expression(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, a, b = py.symbols('alpha_smo t a b')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            a = mdl.params[0]
            b = mdl.params[1]

            eq1 = py.Eq(x(t).diff(t), -a * x(t) + b)

            return eq1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Formula_expression_gm.formula_expression(None, mdl, t_train, x0_train, t_test, x0_test)


class Formula_expression_ngm():
    def __init__(self):
        pass

    def formula_expression(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, a, b = py.symbols('alpha_smo t a b')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            a = mdl.params[0]
            b = mdl.params[1]

            eq1 = py.Eq(x(t).diff(t), -a * x(t) + b * t)
            return eq1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Formula_expression_ngm.formula_expression(None, mdl, t_train, x0_train, t_test, x0_test)


class Formula_expression_bernoulligm():
    def __init__(self):
        pass

    def formula_expression(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, a, b, N = py.symbols('alpha_smo t a b N')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            a = mdl.params[0]
            b = mdl.params[1]
            N = mdl.n

            eq1 = py.Eq(x(t).diff(t), -a * x(t) + b * (x(t) ** N))
            return eq1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Formula_expression_bernoulligm.formula_expression(None, mdl, t_train, x0_train, t_test, x0_test)


class Formula_expression_dgm():
    def __init__(self):
        pass

    def formula_expression(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, b1, b2 = py.symbols('alpha_smo t b1 b2')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            b1 = mdl.params[0]
            b2 = mdl.params[1]

            eq1 = py.Eq(x(t + 1), b1 * x(t) + b2)
            return eq1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Formula_expression_dgm.formula_expression(None, mdl, t_train, x0_train, t_test, x0_test)


class Formula_expression_ndgm():
    def __init__(self):
        pass

    def formula_expression(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, b1, b2, b3, b4 = py.symbols('alpha_smo t b1 b2 b3 b4')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            b1 = mdl.params[0]
            b2 = mdl.params[1]
            b3 = mdl.params[2]
            b4 = 0
            n = max(t)
            for k in range(1, n):
                for j in range(1, k + 1):
                    c_n = j * (b1 ** (k - 1))
                    c = c + c_n
                b4_empty = (mdl.x_orig[k] - (b1 ** k) * mdl.x_orig[0] - b2 * c - (1 - (b1 ** k)) / (1 - b1) * b3) * (
                        b1 ** k)
                b4 = b4 + b4_empty

            eq1 = py.Eq(x(t + 1), b1 * x(t) + b2 * t + b3)
            eq2 = py.Eq(x(1), x(1) + b4)
            eq = [eq1, eq2]
            return eq
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            Formula_expression_ndgm.formula_expression(None, mdl, t_train, x0_train, t_test, x0_test)


class Formula_expression_gmn():
    def __init__(self):
        pass

    def formula_expression(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        if (mdl.is_fitted == True):
            x, t, b1, bn = py.symbols('alpha_smo t b1 bn')
            x = py.Function('alpha_smo')

            mdl.params = np.round(mdl.params, 4)

            b1 = mdl.params[0]
            bn = mdl.params[-1]

            eq = py.Eq(x(1), x(1))
            return eq
        else:
            y = np.concatenate((y_train, y_test), axis=0)
            t_x = np.concatenate((t_x_train, t_x_test), axis=0)
            mdl.fit(y, t_x)
            Formula_expression_gmn.formula_expression(None, mdl, y_train, t_x_train, y_test, t_x_test)

