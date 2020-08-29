from CommonOperations._report_func_list import *




class Report_GM():
    def __init__(self):
        self.mdl_name = 'gm11'
        self.accu_type = 'ago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_DGM():
    def __init__(self):
        self.mdl_name = 'dgm11'
        self.accu_type = 'ago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NGM():
    def __init__(self):
        self.mdl_name = 'ngm'
        self.accu_type = 'ago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_BernoulliGM():
    def __init__(self):
        self.mdl_name = 'bergm'
        self.accu_type = 'ago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NDGM():
    def __init__(self):
        self.mdl_name = 'ndgm'
        self.accu_type = 'ago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all

class Report_GMN():
    def __init__(self):
        self.mdl_name = 'gm1n'
        self.accu_type = 'agon'

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, y_train, t_x_train, y_test, t_x_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, y_train, t_x_train, y_test, t_x_test)
        res_all = [c, eq]
        return res_all


class Report_FGM():
    def __init__(self):
        self.mdl_name = 'gm11'
        self.accu_type = 'fago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_FDGM():
    def __init__(self):
        self.mdl_name = 'dgm11'
        self.accu_type = 'fago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_FNGM():
    def __init__(self):
        self.mdl_name = 'ngm'
        self.accu_type = 'fago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_FBernoulliGM():
    def __init__(self):
        self.mdl_name = 'bergm'
        self.accu_type = 'fago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_FNDGM():
    def __init__(self):
        self.mdl_name = 'ndgm'
        self.accu_type = 'fago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_FGMN():
    def __init__(self):
        self.mdl_name = 'gm1n'
        self.accu_type = 'fagon'

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, y_train, t_x_train, y_test, t_x_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, y_train, t_x_train, y_test, t_x_test)
        res_all = [c, eq]
        return res_all


class Report_NIPGM():
    def __init__(self):
        self.mdl_name = 'gm11'
        self.accu_type = 'nipago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NIPDGM():
    def __init__(self):
        self.mdl_name = 'dgm11'
        self.accu_type = 'nipago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NIPNGM():
    def __init__(self):
        self.mdl_name = 'ngm'
        self.accu_type = 'nipago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NIPBernoulliGM():
    def __init__(self):
        self.mdl_name = 'bergm'
        self.accu_type = 'nipago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NIPNDGM():
    def __init__(self):
        self.mdl_name = 'ndgm'
        self.accu_type = 'nipago'

    def generate_report(self, mdl, t_train, x0_train, t_test, x0_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, t_train, x0_train, t_test, x0_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, t_train, x0_train, t_test, x0_test)
        res_all = [c, eq]
        return res_all


class Report_NIPGMN():
    def __init__(self):
        self.mdl_name = 'gm1n'
        self.accu_type = 'nipagon'

    def generate_report(self, mdl, y_train, t_x_train, y_test, t_x_test, **kwargs):
        c = accumulation_type[self.accu_type].generate_report(mdl, y_train, t_x_train, y_test, t_x_test)
        eq = report_funcs[self.mdl_name].formula_expression(mdl, y_train, t_x_train, y_test, t_x_test)
        res_all = [c, eq]
        return res_all