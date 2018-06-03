import xlwt


class SaveToFile:

    @staticmethod
    def format_gridsearchcv_result(grid_result):
        list_to_export = []

        # for key in grid_result.cv_results_['params'].items()]:

        # store the columns header
        columns = ['score', 'std_deviation']
        columns.extend((grid_result.cv_results_['params'][0]))
        list_to_export.append(columns)

        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for current_mean, current_stdev, current_params in zip(means, stds, params):
            # store the columns header
            columns = [current_mean, current_stdev]
            # for key,val in [(key,val) for key, val in dct.items()
            columns.extend([val for key, val in current_params.items()])
            list_to_export.append(columns)

        return list_to_export

    # score, params

    def save_to_excel(filename, sheet, datalist):
        book = xlwt.Workbook(encoding='latin-1')
        sh = book.add_sheet(sheet)

        # You may need to group the variables together
        # for n, (v_desc, v) in enumerate(zip(desc, variables)):
        for current_line_nb, current_line in enumerate(datalist):
            for current_column_nb, current_cell in enumerate(current_line):
                print(current_line_nb, current_column_nb)
                print(type(current_cell))
                aaaa = current_cell[0] if type(current_cell) == list else current_cell
                sh.write(current_line_nb, current_column_nb, aaaa)

        book.save(filename)
