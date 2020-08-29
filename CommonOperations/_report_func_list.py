from CommonOperations.ReportFunctions import *

report_funcs = {'gm11': Formula_expression_gm(), 'ngm': Formula_expression_ngm(),
                'bergm': Formula_expression_bernoulligm(),
                'dgm11': Formula_expression_dgm(), 'ndgm': Formula_expression_ndgm(),
                'gm1n':Formula_expression_gmn()}

accumulation_type = {'ago': Generate_report_ago(), 'fago': Generate_report_fago(), 'nipago': Generate_report_nipago(),
                     'agon':Generate_report_agon(), 'fagon':Generate_report_fagon(), 'nipagon': Generate_report_nipagon()}
