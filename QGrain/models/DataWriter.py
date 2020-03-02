__all__ = ["DataWriter"]

import csv
import json
import os
from datetime import date, datetime, time
from typing import List

import numpy as np
import xlsxwriter
import xlwt

from QGrain.algorithms import DistributionType
from QGrain.models.DataLoader import FileType
from QGrain.models.FittingResult import FittingResult

QGRAIN_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def column_to_char(column_index: int):
    column = column_index + 1
    column_str = str()
    while column != 0:
        res = column % 26
        if res == 0:
            res = 26
            column -= 26
        column_str = chr(ord('A') + res - 1) + column_str
        column = column // 26
    return column_str

def to_cell_name(row: int, column: int):
    return "{0}{1}".format(column_to_char(column), row+1)


class DataWriter:
    """
    The class to save the fitting results to local files.
    """
    MAX_PARAM_COUNT = 3
    def __init__(self):
        super().__init__()
        self.style_file_path = os.path.join(QGRAIN_ROOT_PATH, "settings", "chart_styles.json")
        # see https://xlsxwriter.readthedocs.io/chart.html

    def get_distribution_name(self, distribution_type: DistributionType):
        if distribution_type == DistributionType.Normal:
            return "Normal"
        elif distribution_type == DistributionType.Weibull:
            return "Weibull"
        elif distribution_type == DistributionType.GeneralWeibull:
            return "Gen. Weibull"
        else:
            raise NotImplementedError(distribution_type)

    def try_save_data(self, filename: str, file_type: FileType,
                      results: List[FittingResult], draw_charts: bool):
        if filename is None or filename == "":
            raise ValueError(filename)

        if file_type == FileType.XLSX:
            self.try_save_as_excel(filename, results, True, draw_charts)
        elif file_type == FileType.XLS:
            self.try_save_as_excel(filename, results, False, draw_charts)
        elif file_type == FileType.CSV:
            self.try_save_as_csv(filename, results)
        else:
            raise NotImplementedError(file_type)

    # If type is csv, it will use built-in csv module to handle this request.
    def try_save_as_csv(self, filename: str, results: List[FittingResult]):
        # use `newline=""` to avoid redundant newlines
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            max_component_number = max([result.component_number for result in results])
            # write the hearders
            headers = ["Sample Name", "Distribution Type", "Mean Squared Error",
            "Pearson Correlation Coefficient", "P Value", "Kendall's TAU", "P Value",
            "Spearman Correlation Coefficient", "P Value"]
            for i in range(max_component_number):
                headers.append("")
                headers.append("Fraction (%)")
                headers.append("Mean (mu m)")
                headers.append("Median (mu m)")
                headers.append("Mode (mu m)")
                headers.append("Variance")
                headers.append("Standard Deviation")
                headers.append("Skewness")
                headers.append("Kurtosis")
                for j in range(self.MAX_PARAM_COUNT):
                    headers.append("Parameter" + " {0}".format(j+1))
            w.writerow(headers)
            # write the contents
            for result in results:
                row = [result.name,
                       self.get_distribution_name(result.distribution_type),
                       result.mean_squared_error, *result.pearson_r,
                       *result.kendall_tau, *result.spearman_r]
                for component in result.components:
                    row.append("")
                    row.append(component.fraction * 100.0)
                    row.append(component.mean)
                    row.append(component.median)
                    row.append(component.mode)
                    row.append(component.variance)
                    row.append(component.standard_deviation)
                    row.append(component.skewness)
                    row.append(component.kurtosis)
                    # write the parameters of each base distribution
                    for i in range(self.MAX_PARAM_COUNT):
                        if i < result.param_count:
                            row.append(component.params[i])
                        else:
                            row.append("")
                w.writerow(row)

    def try_save_as_excel(self, filename: str, results: List[FittingResult],
                          is_xlsx: bool, draw_charts: bool):
        # the local func to check the value validation
        # must block unaccepted values before calling the `write` func of worksheet
        def check_value(value):
            accepted_types = (bool, datetime, date, time, xlwt.Formula)
            if value is None:
                return "None"
            # must judge this first,
            # because `Formula` can pass `numpy.isreal` but raise error in `numpy.isnan`
            elif type(value) in accepted_types:
                return value
            # if value is real number (i.e. int, float, numpy.int32, numpy.float32, etc.)
            elif np.isreal(value):
                # this func will both recognize float("nan") and numpy.nan
                if np.isnan(value):
                    return "NaN"
                # similar to above
                elif np.isinf(value):
                    return "Inf"
                else:
                    return value
            # if the type of value is unaccepted, use its readable str to write
            else:
                return str(value)

        def write(sheet, row, col, value, style):
            sheet.write(row, col, check_value(value), style)

        # the wrappers to handle the differences between xlwt and xlsxwriter
        if is_xlsx:
            # see https://xlsxwriter.readthedocs.io/worksheet.html#merge_range
            def mwrite(sheet, lrow, lcol, rrow, rcol, value, style):
                sheet.merge_range(lrow, lcol, rrow, rcol, check_value(value), style)
            def set_col(sheet, col, width):
                sheet.set_column(col, col, width)
        else:
            def mwrite(sheet, lrow, lcol, rrow, rcol, value, style):
                sheet.write_merge(lrow, rrow, lcol, rcol, check_value(value), style)
            def set_col(sheet, col, width):
                sheet.col(col).width = width*256
        # generate different styles
        if is_xlsx:
            book = xlsxwriter.Workbook(filename=filename)
            # see https://xlsxwriter.readthedocs.io/format.html#format
            global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#F5F5F5", "num_format": "0.00"})
            dark_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#A9A9A9", "num_format": "0.00"})
            detail_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#F5F5F5", "num_format": "0.0000"})
            dark_detail_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#A9A9A9", "num_format": "0.0000"})
            exp_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#F5F5F5", "num_format": "0.00E+00"})
            dark_exp_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#A9A9A9", "num_format": "0.00E+00"})
            fraction_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#F5F5F5", "num_format": "0.00%"})
            dark_fraction_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#A9A9A9", "num_format": "0.00%"})
            header_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "font_color": "#FFF5EE", "bold": True, "align": "center", "valign": "vcenter", "text_wrap": True, "pattern": 1, "fg_color": "#4169E1", "num_format": "0.00"})
            # use this sheet to record the statistic data
            summary_sheet_name = "Summary" # this name will be referred in detail sheet
            summary_sheet = book.add_worksheet(summary_sheet_name)
            detail_sheet_name = "Detail"
            detail_sheet = book.add_worksheet(detail_sheet_name)
        else:
            book = xlwt.Workbook()
            # create styles by easyxf
            # see https://github.com/python-excel/xlwt/blob/master/examples/xlwt_easyxf_simple_demo.py
            global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00")
            dark_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00")
            detail_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.0000")
            dark_detail_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.0000")
            exp_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00E+00")
            dark_exp_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00E+00")
            fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00%")
            dark_fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00%")
            header_style = xlwt.easyxf("font: name Times New Roman, height 320, bold on, colour 1; align: wrap on, vert centre, horiz center; pattern: pattern solid, fore_colour 30", num_format_str="0.00")
            # use this sheet to record the statistic data
            summary_sheet_name = "Summary" # this name will be referred in detailed sheet
            summary_sheet = book.add_sheet(summary_sheet_name)
            detail_sheet_name = "Detail"
            detail_sheet = book.add_sheet(detail_sheet_name)

        max_component_number = max([result.component_number for result in results])
        SUMMARY_HEADER_ROWS = 2
        SUMMARY_COMPONENTS = 8 + self.MAX_PARAM_COUNT
        SUMMARY_COLUMN_SPAN = SUMMARY_COMPONENTS + 1
        SUMMARY_COMPONENT_START_COLUMN = 10
        # sheet 1.
        # headers of summary sheet
        mwrite(summary_sheet, 0, 0, 1, 0, "Sample Name", header_style)
        mwrite(summary_sheet, 0, 1, 1, 1, "Distribution Type", header_style)
        mwrite(summary_sheet, 0, 2, 1, 2, "Mean Squared Error", header_style)
        mwrite(summary_sheet, 0, 3, 0, 4, "Pearson's Correlation Coefficient", header_style)
        write(summary_sheet, 1, 3, "Correlation", header_style)
        write(summary_sheet, 1, 4, "Two-tailed P-value", header_style)
        mwrite(summary_sheet, 0, 5, 0, 6, "Kendall’s TAU", header_style)
        write(summary_sheet, 1, 5, "The TAU Statistic", header_style)
        write(summary_sheet, 1, 6, "Two-sided P-value", header_style)
        mwrite(summary_sheet, 0, 7, 0, 8, "Spearman's Correlation Coefficient", header_style)
        write(summary_sheet, 1, 7, "Correlation", header_style)
        write(summary_sheet, 1, 8, "Two-sided P-value", header_style)
        for i in range(max_component_number):
            left_col = i*SUMMARY_COLUMN_SPAN+SUMMARY_COMPONENT_START_COLUMN
            mwrite(summary_sheet, 0, left_col, 0, left_col+SUMMARY_COMPONENTS-1, "Component {0}".format(i+1), header_style)
            write(summary_sheet, 1, left_col, "Fraction", header_style)
            write(summary_sheet, 1, left_col+1, "Mean (μm)", header_style)
            write(summary_sheet, 1, left_col+2, "Median (μm)", header_style)
            write(summary_sheet, 1, left_col+3, "Mode (μm)", header_style)
            write(summary_sheet, 1, left_col+4, "Variance", header_style)
            write(summary_sheet, 1, left_col+5, "Standard Deviation", header_style)
            write(summary_sheet, 1, left_col+6, "Skewness", header_style)
            write(summary_sheet, 1, left_col+7, "Kurtosis", header_style)
            for j in range(self.MAX_PARAM_COUNT):
                write(summary_sheet, 1, left_col+8+j, "Parameter" + " {0}".format(j+1), header_style)

        # set widths of columns
        set_col(summary_sheet, 0, 16)
        set_col(summary_sheet, 1, 16)
        for i in range(2, 9):
            if i % 2 == 0:
                set_col(summary_sheet, i, 12)
            else:
                set_col(summary_sheet, i, 8)
        for i in range(SUMMARY_COMPONENT_START_COLUMN, max_component_number*SUMMARY_COLUMN_SPAN+SUMMARY_COMPONENT_START_COLUMN):
            set_col(summary_sheet, i, 8)
        # contents of summary sheet
        for row, result in enumerate(results, SUMMARY_HEADER_ROWS):
            # take this way to let color changes alternately
            if row % 2 == 0:
                current_global_style = global_style
                current_exp_style = exp_style
                current_fraction_style = fraction_style
            else:
                current_global_style = dark_global_style
                current_exp_style = dark_exp_style
                current_fraction_style = dark_fraction_style
            write(summary_sheet, row, 0, result.name, current_global_style)
            write(summary_sheet, row, 1, self.get_distribution_name(result.distribution_type), current_global_style)
            write(summary_sheet, row, 2, result.mean_squared_error, current_exp_style)
            write(summary_sheet, row, 3, result.pearson_r[0], current_global_style)
            write(summary_sheet, row, 4, result.pearson_r[1], current_exp_style)
            write(summary_sheet, row, 5, result.kendall_tau[0], current_global_style)
            write(summary_sheet, row, 6, result.kendall_tau[1], current_exp_style)
            write(summary_sheet, row, 7, result.spearman_r[0], current_global_style)
            write(summary_sheet, row, 8, result.spearman_r[1], current_exp_style)

            for i, component in enumerate(result.components):
                left_col = i*SUMMARY_COLUMN_SPAN+SUMMARY_COMPONENT_START_COLUMN
                write(summary_sheet, row, left_col, component.fraction, current_fraction_style)
                write(summary_sheet, row, left_col+1, component.mean, current_global_style)
                write(summary_sheet, row, left_col+2, component.median, current_global_style)
                write(summary_sheet, row, left_col+3, component.median, current_global_style)
                write(summary_sheet, row, left_col+4, component.variance, current_global_style)
                write(summary_sheet, row, left_col+5, component.standard_deviation, current_global_style)
                write(summary_sheet, row, left_col+6, component.skewness, current_global_style)
                write(summary_sheet, row, left_col+7, component.kurtosis, current_global_style)

                for j in range(self.MAX_PARAM_COUNT):
                    if j < result.param_count:
                        write(summary_sheet, row, left_col+8+j, component.params[j], current_global_style)

        # sheet 2.
        # headers of detail sheet
        DETAIL_COMPONENT_START_COLUMN = 3
        write(detail_sheet, 0, 0, "Sample Name", header_style)
        write(detail_sheet, 0, 1, "Series Name", header_style)
        set_col(detail_sheet, 0, 16)
        set_col(detail_sheet, 1, 16)
        classes_length = len(results[0].real_x)
        for i in range(DETAIL_COMPONENT_START_COLUMN, DETAIL_COMPONENT_START_COLUMN+classes_length):
            set_col(detail_sheet, i, 12)
        # classes headers
        for col, value in enumerate(results[0].real_x, DETAIL_COMPONENT_START_COLUMN):
            write(detail_sheet, 0, col, value, header_style)
        # contents of detail sheet
        row = 1
        for data_index, result in enumerate(results):
            if data_index % 2 == 0:
                current_global_style = detail_global_style
            else:
                current_global_style = dark_detail_global_style
            mwrite(detail_sheet, row, 0, row+result.component_number+1, 0, result.name, current_global_style)
            # target row
            for col, value in enumerate(result.target_y, DETAIL_COMPONENT_START_COLUMN):
                write(detail_sheet, row, col, value, current_global_style)
            write(detail_sheet, row, 1, "Target", current_global_style)
            row += 1
            # sum row
            for col, value in enumerate(result.fitted_y, DETAIL_COMPONENT_START_COLUMN):
                write(detail_sheet, row, col, value, current_global_style)
            write(detail_sheet, row, 1, "Fitted Sum", current_global_style)
            row += 1
            # components' rows
            for component_index, component in enumerate(result.components):
                write(detail_sheet, row, 1, "C{0}".format(component_index+1), current_global_style)
                for col, value in enumerate(component.component_y, DETAIL_COMPONENT_START_COLUMN):
                    write(detail_sheet, row, col, value, current_global_style)
                row += 1

        # Save file if it is xls or no need to draw charts
        if not is_xlsx or not draw_charts:
            if is_xlsx:
                book.close()
                return
            else:
                book.save(filename)
                return

        # add charts to workbook
        chart_styles = None
        column_chart_number = 5
        x_margin = 10
        y_margin = 10
        width = 480
        height = 288
        try:
            with open(self.style_file_path, "r", encoding="utf-8") as style_file:
                chart_styles = json.load(style_file)
                column_chart_number = chart_styles["column_chart_number"]
                x_margin = chart_styles["x_margin"]
                y_margin = chart_styles["y_margin"]
                width = chart_styles["size"]["x_scale"]*480
                height = chart_styles["size"]["y_scale"]*288
        except OSError:
            pass

        chart_sheet = book.add_worksheet("Charts")
        row = 1
        for i, result in enumerate(results):
            chart = book.add_chart({'type': 'scatter'})
            if chart_styles is not None:
                chart.set_x_axis(chart_styles["x_axis"])
                chart.set_y_axis(chart_styles["y_axis"])
                chart.set_title({**chart_styles["title"], "name": [detail_sheet_name, row, 0]})
                chart.set_legend(chart_styles["legend"])
                chart.set_size(chart_styles["size"])
                for j in range(result.component_number+2):
                    if j  == 0:
                        style = chart_styles["target_series"]
                    elif j == 1:
                        style = chart_styles["fitted_series"]
                    else:
                        style = chart_styles["component_series"]
                    chart.add_series({**{"name": [detail_sheet_name, row, 1],
                                        "categories": [detail_sheet_name, 0, DETAIL_COMPONENT_START_COLUMN, 0, DETAIL_COMPONENT_START_COLUMN+classes_length-1],
                                        "values": [detail_sheet_name, row, DETAIL_COMPONENT_START_COLUMN, row, DETAIL_COMPONENT_START_COLUMN+classes_length-1]}, **style})
                    row += 1
            else:
                chart.set_x_axis({"log_base": 10, "crossing": 0.01})
                chart.set_title({"name": [detail_sheet_name, row, 0]})
                for j in range(result.component_number+2):
                    chart.add_series({"name": [detail_sheet_name, row, 1],
                                      "categories": [detail_sheet_name, 0, DETAIL_COMPONENT_START_COLUMN, 0, DETAIL_COMPONENT_START_COLUMN+classes_length-1],
                                      "values": [detail_sheet_name, row, DETAIL_COMPONENT_START_COLUMN, row, DETAIL_COMPONENT_START_COLUMN+classes_length-1]})
                    row += 1

            x_offset = (i%column_chart_number) * (width+x_margin)
            y_offset = (i//column_chart_number) * (height+y_margin)
            chart_sheet.insert_chart('A1', chart, {"x_offset": x_offset, "y_offset": y_offset})

        book.close()
