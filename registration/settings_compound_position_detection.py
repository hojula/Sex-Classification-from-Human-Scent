class Settings:
    def __init__(self, box_add_x, box_add_y, metric, weighted, weighting_function, only_significant=False):
        self.box_add_x = box_add_x
        self.box_add_y = box_add_y
        self.metric = metric
        self.weighted = weighted
        self.weighting_function = weighting_function
        self.only_significant = only_significant

    def __str__(self):
        import inspect
        if self.weighting_function is not None:
            if inspect.isfunction(self.weighting_function) and self.weighting_function.__name__ == "<lambda>":
                lambda_source = inspect.getsource(self.weighting_function).strip()
                weighting_function_str = lambda_source.split(':', 1)[1].strip()
                weighting_function_str = weighting_function_str.split(',')[0]
            else:
                weighting_function_str = str(self.weighting_function)
        else:
            weighting_function_str = "None"
        return f"box size x: {2 * self.box_add_x + 1}, box size y: {2 * self.box_add_y + 1}, metric: {self.metric}, weighted: {self.weighted}, weighting function: {weighting_function_str}, only significant: {self.only_significant}"


def set_settings_for_box_test():
    settings1 = Settings(1, 1, 'dot', False, None, False)
    settings2 = Settings(1, 100, 'dot', False, None, False)
    settings3 = Settings(1, 50, 'dot', False, None, False)
    settings4 = Settings(3, 1, 'dot', False, None, False)
    settings5 = Settings(3, 100, 'dot', False, None, False)
    settings6 = Settings(3, 50, 'dot', False, None, False)
    settings7 = Settings(5, 1, 'dot', False, None, False)
    settings8 = Settings(5, 100, 'dot', False, None, False)
    settings9 = Settings(5, 50, 'dot', False, None, False)
    settings10 = Settings(20, 1, 'dot', False, None, False)
    settings11 = Settings(20, 100, 'dot', False, None, False)
    settings12 = Settings(20, 50, 'dot', False, None, False)
    return [settings1, settings2, settings3, settings4, settings5, settings6, settings7, settings8, settings9,
            settings10, settings11, settings12]


def set_settings_for_metric_test():
    settings1 = Settings(5, 50, 'dot', False, None, False)
    settings2 = Settings(5, 50, 'l1', False, None, False)
    settings3 = Settings(5, 50, 'l2', False, None, False)
    settings4 = Settings(5, 50, 'l_inf', False, None, False)
    settings5 = Settings(5, 50, 'svm', False, None, False)
    settings6 = Settings(5, 50, 'cnn', False, None, False)
    return [settings1, settings2, settings3, settings4, settings5]


def set_settings_for_significant_compounds_test():
    settings1 = Settings(5, 50, 'cnn_box', False, None, True)
    # settings2 = Settings(5, 50, 'dot', False, None, False)
    return [settings1]
    # return [settings1, settings2]


def set_settings_svm():
    settings1 = Settings(-1, -1, 'svm', False, None, False)
    return [settings1]


def set_settings_for_weighting_test():
    settings1 = Settings(5, 50, 'dot', False, None, False)
    settings2 = Settings(5, 50, 'dot', True, lambda x: x, False)
    settings3 = Settings(5, 50, 'dot', True, lambda x: x ** 2, False)
    settings4 = Settings(5, 50, 'dot', True, lambda x: x ** 4, False)
    settings5 = Settings(5, 50, 'dot', True, lambda x: x ** 8, False)
    settings6 = Settings(5, 50, 'dot', True, lambda x: torch.exp(x), False)

    return [settings1, settings2, settings3, settings4, settings5, settings6]


def set_settings_for_weighting_test_24():
    settings4 = Settings(5, 50, 'dot', True, lambda x: x ** 4, True)
    return [settings4]


def set_setting_plot():
    settings1 = Settings(5, 50, 'dot', False, lambda x: x ** 4, True)
    return [settings1]


def set_settings_cnn():
    settings1 = Settings(-1, -1, 'cnn', False, None, True)
    return [settings1]