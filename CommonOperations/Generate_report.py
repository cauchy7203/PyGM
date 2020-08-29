from CommonOperations import Accumulation
from CommonOperations import FractionalAccumulation
from CommonOperations import NewInformationPriorityAccumulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as py


class generate_report():
    def __init__(self):
        pass


    def generate_report_gm(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_gm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_gm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_fgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_gm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=0)
            x0 = np.concatenate((x0_train, x0_test), axis=0)
            mdl.fit(t, x0)
            generate_report.generate_report_fgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_nipgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_gm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_nipgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_ngm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ngm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_ngm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_fngm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ngm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_fngm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_nipngm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ngm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_nipngm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_dgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_dgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_dgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_fdgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_dgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_fdgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_nipdgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_dgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_nipdgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_ndgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ndgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_ndgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_fndgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ndgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_fndgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_nipndgm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_ndgm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_nipndgm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_bernoulligm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.ago(mdl.x_orig, None, True)
            x1_pred = Accumulation.ago(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_bernoulligm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_bernoulligm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_fbernoulligm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = FractionalAccumulation.fago(mdl.x_orig, mdl.r)
            x1_pred = FractionalAccumulation.fago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_bernoulligm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_fbernoulligm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_nipbernoulligm(self, mdl, t_train, x0_train, t_test, x0_test):
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = NewInformationPriorityAccumulation.nipago(mdl.x_orig, mdl.r)
            x1_pred = NewInformationPriorityAccumulation.nipago(x_pred, mdl.r)
            model_report = pd.DataFrame({
                'x_orig': mdl.x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_bernoulligm(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_nipbernoulligm(mdl, t_train, x0_train, t_test, x0_test)

    def generate_report_gmn(self, mdl, t_train, x0_train, t_test, x0_test):
        if (mdl.is_fitted == True):
            t_all = np.concatenate((t_train, t_test), axis=0)
            x_all = np.concatenate((x0_train, x0_test), axis=0)
            t_x = np.concatenate((t_all, mdl.x1), axis=1)
            x_pred = mdl.predict(t_x)

            x_orig = mdl.x_orig[0:, 1]
            lens = len(x_orig)
            err = x_pred[:lens] - x_orig
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
            plt.legend(['x_orig_models', 'x_pred_models', 'x_orig_detected', 'x_pred_detected'])
            plt.axvline(x=m, ls="--", lw=1, c='k')
            plt.title('Model and Test')
            plt.subplot(2, 1, 2)
            plt.stem(t3, err, linefmt="c:", markerfmt="o", basefmt="r-")
            plt.title('RMSE = {}'.format(rmse))
            plt.axvline(x=m, ls="--", lw=1, c='k')

            # 打印报告
            x1 = Accumulation.agom(mdl.x_orig, None, True)
            x1 = x1[0:, 1]
            x1_pred = Accumulation.agom(x_pred, None, True)
            model_report = pd.DataFrame({
                'x_orig': x_orig,
                'x1': x1,
                'x1_pred': x1_pred,
                'x_pred': x_pred,
                'error': err,
            })

            # 公式表达
            eq = generate_report.formula_expression_gmn(None, mdl)

            c1 = [p, model_report, eq]
            return c1
        else:
            t = np.concatenate((t_train, t_test), axis=1)
            x0 = np.concatenate((x0_train, x0_test), axis=1)
            mdl.fit(t, x0)
            generate_report.generate_report_gmn(mdl, t_train, x0_train, t_test, x0_test)

    # 各模型公式
    def formula_expression_gm(self, mdl):
        x, t, a, b = py.symbols('x t a b')
        x = py.Function('x')

        mdl.params = np.round(mdl.params, 4)

        a = mdl.params[0]
        b = mdl.params[1]

        eq1 = py.Eq(x(t).diff(t), -a * x(t) + b)

        return eq1

    def formula_expression_ngm(self, mdl):
        x, t, a, b = py.symbols('x t a b')
        x = py.Function('x')

        mdl.params = np.round(mdl.params, 4)

        a = mdl.params[0]
        b = mdl.params[1]

        eq1 = py.Eq(x(t).diff(t), -a * x(t) + b * t)
        return eq1

    def formula_expression_bernoulligm(self, mdl):
        x, t, a, b, N = py.symbols('x t a b N')
        x = py.Function('x')

        mdl.params = np.round(mdl.params, 4)

        a = mdl.params[0]
        b = mdl.params[1]
        N = mdl.n

        eq1 = py.Eq(x(t).diff(t), -a * x(t) + b * (x(t) ** N))
        return eq1

    def formula_expression_dgm(self, mdl):
        x, t, b1, b2 = py.symbols('x t b1 b2')
        x = py.Function('x')

        mdl.params = np.round(mdl.params, 4)

        b1 = mdl.params[0]
        b2 = mdl.params[1]

        eq1 = py.Eq(x(t + 1), b1 * x(t) + b2)
        return eq1

    def formula_expression_ndgm(self, mdl):
        x, t, b1, b2, b3, b4 = py.symbols('x t b1 b2 b3 b4')
        x = py.Function('x')

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

    def formula_expression_gmn(self, mdl):
        x, x1, xn, t, a, b1, bn = py.symbols('x x1 xn t a b1 bn')
        x = py.Function('x')

        mdl.params = np.round(mdl.params, 4)

        a = mdl.params[0]
        b1 = mdl.params[1]
        bn = mdl.params[-1]

        eq2 = py.Eq(x(t).diff(t), -a * x(t) + b1 * x1 + ~ + bn * xn)

        py.pprint(eq2)
        py.pprint(mdl.B)
