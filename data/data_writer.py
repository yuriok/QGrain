import csv
import json
import logging
import os
from typing import List

import numpy as np
import xlsxwriter
import xlwt
from PySide2.QtCore import QObject, Signal

from data import FittedData


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

# TODO: CHECK THE INPUT DATA
# IF DATA HAS INVALID VALUE LIKE NAN OR NONE

# TODO: SIMPLIFY THE CODES
class DataWriter(QObject):
    sigWorkFinished = Signal(bool)
    logger = logging.getLogger("root.data.DataWriter")
    def __init__(self):
        super().__init__()


    def try_save_data(self, filename, classes: np.ndarray, data: List[FittedData], file_type: str):
        if filename is None or filename == "":
            raise ValueError(filename)
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)
        
        if file_type == "xlsx":
            self.try_save_as_xlsx(filename, classes, data)
        elif file_type == "xls":
            self.try_save_as_xls(filename, classes, data)
        elif file_type == "csv":
            self.try_save_as_csv(filename, data)
        else:
            raise NotImplementedError(file_type)

    # If type is csv,
    # it will use built-in csv module to handle this request.
    # It will write the statistic data only.
    # If you need the detailed data including the target data to fit,
    # fitted data of summing all component, and fitted data of each component,
    # please use `try_save_as_xls` or `try_save_as_xlsx` func.
    # If you need the charts automatically drew in workbook,
    # please use `try_save_as_xlsx` func.
    def try_save_as_csv(self, filename, data: List[FittedData]):
        try:
            f = open(filename, "w", newline="") # use `newline=""` to avoid redundant newlines
            w = csv.writer(f)
            max_ncomp = max([len(fitted.statistic) for fitted in data])
            headers = ["Sample Name", "Mean Squared Error"]
            for i in range(max_ncomp):
                headers.append("")
                headers.append("Fraction")
                headers.append("Mean (μm)")
                headers.append("Median (μm)")
                headers.append("Mode (μm)")
                headers.append("Variance")
                headers.append("Standard Deviation")
                headers.append("Skewness")
                headers.append("Kurtosis")
                headers.append("Beta")
                headers.append("Eta")
                headers.append("Location")
                headers.append("X Offset")
            w.writerow(headers)

            for fitted_data in data:
                row  = [fitted_data.name, fitted_data.mse]
                for comp in fitted_data.statistic:
                    row.append("")
                    row.append(comp.get("fraction"))
                    row.append(comp.get("mean"))
                    row.append(comp.get("median"))
                    row.append(comp.get("mode"))
                    row.append(comp.get("variance"))
                    row.append(comp.get("standard_deviation"))
                    row.append(comp.get("skewness"))
                    row.append(comp.get("kurtosis"))
                    row.append(comp.get("beta"))
                    row.append(comp.get("eta"))
                    row.append(comp.get("loc"))
                    row.append(comp.get("x_offset"))
                w.writerow(row)
            self.sigWorkFinished.emit(True)

        except Exception:
            self.logger.exception("File saving failed.", stack_info=True)
            self.sigWorkFinished.emit(False)
        finally:
            f.close()

    # If type is 97-2003 Excel Workbook,
    # It will use xlwt to handle this request.
    # It will write the statistic data as the sheet `Summary`.
    # It also will write the detailed data as sheet `Detail`.
    # With detailed data, users can redraw the plots in Excel or other apps.
    # If you need the charts automatically drew in workbook,
    # please use `try_save_as_xlsx` func.
    def try_save_as_xls(self, filename, classes, data: List[FittedData]):
        book = xlwt.Workbook()
        # create styles by easyxf
        # see https://github.com/python-excel/xlwt/blob/master/examples/xlwt_easyxf_simple_demo.py
        global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00")
        dark_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00")
        detail_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.0000")
        dark_detail_global_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.0000")
        mse_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00E+00")
        dark_mse_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00E+00")
        fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00%")
        dark_fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00%")
        header_style = xlwt.easyxf("font: name Times New Roman, height 320, bold on, colour 1; align: wrap on, vert centre, horiz center; pattern: pattern solid, fore_colour 30", num_format_str="0.00")
        # use this sheet to record the statistic data
        summary_sheet_name = "Summary" # this name will be referred in detailed sheet
        summary_sheet = book.add_sheet(summary_sheet_name)
        # write headers
        summary_sheet.write_merge(0, 1, 0, 0, "Sample Name", style=header_style)
        summary_sheet.write_merge(0, 1, 1, 1, "Mean Squared Error", style=header_style)
        max_ncomp = max([len(fitted.statistic) for fitted in data])
        for i in range(max_ncomp):
            summary_sheet.write_merge(0, 0, i*13+3, i*13+14, "Component {0}".format(i+1), header_style)
            summary_sheet.write(1, i*13+3, "Fraction", header_style)
            summary_sheet.write(1, i*13+4, "Mean (μm)", header_style)
            summary_sheet.write(1, i*13+5, "Median (μm)", header_style)
            summary_sheet.write(1, i*13+6, "Mode (μm)", header_style)
            summary_sheet.write(1, i*13+7, "Variance", header_style)
            summary_sheet.write(1, i*13+8, "Standard Deviation", header_style)
            summary_sheet.write(1, i*13+9, "Skewness", header_style)
            summary_sheet.write(1, i*13+10, "Kurtosis", header_style)
            summary_sheet.write(1, i*13+11, "Beta", header_style)
            summary_sheet.write(1, i*13+12, "Eta", header_style)
            summary_sheet.write(1, i*13+13, "Location", header_style)
            summary_sheet.write(1, i*13+14, "X Offset", header_style)

        # set widths of columns
        summary_sheet.col(0).width = 256 * 16
        summary_sheet.col(1).width = 256 * 12
        for i in range(2, max_ncomp*13+2):
            summary_sheet.col(i).width = 256 * 8

        # write data content
        for row, fitted_data in enumerate(data, 2):
            # take this way to let color changes alternately
            if row % 2 == 0:
                current_global_style = global_style
                current_mse_style = mse_style
                current_fraction_style = fraction_style
            else:
                current_global_style = dark_global_style
                current_mse_style = dark_mse_style
                current_fraction_style = dark_fraction_style

            summary_sheet.write(row, 0, fitted_data.name, current_global_style)
            summary_sheet.write(row, 1, fitted_data.mse, current_mse_style)
            for i, comp in enumerate(fitted_data.statistic):
                summary_sheet.write(row, i*13+3, comp.get("fraction"), current_fraction_style)
                summary_sheet.write(row, i*13+4, comp.get("mean"), current_global_style)
                summary_sheet.write(row, i*13+5, comp.get("median"), current_global_style)
                summary_sheet.write(row, i*13+6, comp.get("mode"), current_global_style)
                summary_sheet.write(row, i*13+7, comp.get("variance"), current_global_style)
                summary_sheet.write(row, i*13+8, comp.get("standard_deviation"), current_global_style)
                summary_sheet.write(row, i*13+9, comp.get("skewness"), current_global_style)
                summary_sheet.write(row, i*13+10, comp.get("kurtosis"), current_global_style)
                summary_sheet.write(row, i*13+11, comp.get("beta"), current_global_style)
                summary_sheet.write(row, i*13+12, comp.get("eta"), current_global_style)
                summary_sheet.write(row, i*13+13, comp.get("loc"), current_global_style)
                summary_sheet.write(row, i*13+14, comp.get("x_offset"), current_global_style)

        # use this sheet to record the detail data
        detail_sheet = book.add_sheet("Detail")
        detail_sheet.write(0, 0, "Sample Name", header_style)
        detail_sheet.write(0, 1, "Series Name", header_style)
        detail_sheet.col(0).width = 256 * 16
        detail_sheet.col(1).width = 256 * 12
        detail_sheet.col(2).width = 256 * 12
        for col, value in enumerate(classes, 3):
            detail_sheet.write(0, col, value, header_style)
            detail_sheet.col(col).width = 256 * 12
        # write data content
        row = 1
        for data_index, fitted_data in enumerate(data):
            if data_index % 2 == 0:
                current_global_style = detail_global_style
            else:
                current_global_style = dark_detail_global_style
            
            ncomp = len(fitted_data.statistic)
            # rows: target + fitted sumn + components
            detail_sheet.write_merge(row, row+ncomp+1, 0, 0, fitted_data.name, current_global_style)
            left = fitted_data.statistic[0]["loc"]
            
            # filling non-existent cells with 0
            for i in range(row, row+ncomp+2):
                # 3 to 3 + left (exclusive) is the left blank region
                for col in range(3, 3+left):
                    detail_sheet.write(i, col, 0, current_global_style)
                # 3+left+len(target[1]) to 3+len(classes) (exclusive) is the right blank region
                for col in range(3+left+len(fitted_data.target[1]), 3+len(classes)):
                    detail_sheet.write(i, col, 0, current_global_style)
            # target row
            for col, value in enumerate(fitted_data.target[1], 3+left):
                detail_sheet.write(row, col, value, current_global_style)
            detail_sheet.write(row, 1, "Target", current_global_style)
            row += 1
            # sum row
            for col, value in enumerate(fitted_data.sum[1], 3+left):
                detail_sheet.write(row, col, value, current_global_style)
            detail_sheet.write(row, 1, "Fitted Sum", current_global_style)
            row += 1
            # coponent rows
            for component_index, component in enumerate(fitted_data.components):
                detail_sheet.write(row, 1, "C{0}".format(component_index+1), current_global_style)
                for col, value in enumerate(component[1], 3+left):
                    # 1. use values
                    # detail_sheet.write(row, col, value, current_global_style)
                    # 2. use formula
                    detail_sheet.write(row, col, xlwt.Formula(
                        "WEIBULL({0}!{1}+{2}, {0}!{3}, {0}!{4}, FALSE)*{0}!{5}".format(
                            summary_sheet_name, to_cell_name(data_index+2, component_index*13+14), col-3-left,
                            to_cell_name(data_index+2, component_index*13+11),
                            to_cell_name(data_index+2, component_index*13+12),
                            to_cell_name(data_index+2, component_index*13+3))), current_global_style)
                row += 1

        try:
            book.save(filename)
            self.sigWorkFinished.emit(True)
        except Exception:
            logging.exception("File saving failed.", stack_info=True)
            self.sigWorkFinished.emit(False)


    def try_save_as_xlsx(self, filename, classes, data: List[FittedData]):
        book = xlsxwriter.Workbook(filename=filename)
        
        # see https://xlsxwriter.readthedocs.io/format.html#format
        global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#87CEFA", "num_format": "0.00"})
        dark_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#6495ED", "num_format": "0.00"})
        detail_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#87CEFA", "num_format": "0.0000"})
        dark_detail_global_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#6495ED", "num_format": "0.0000"})
        mse_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#87CEFA", "num_format": "0.00E+00"})
        dark_mse_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#6495ED", "num_format": "0.00E+00"})
        fraction_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#87CEFA", "num_format": "0.00%"})
        dark_fraction_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "align": "center", "valign": "vcenter", "pattern": 1, "fg_color": "#6495ED", "num_format": "0.00%"})
        header_style = book.add_format({"font_name": "Times New Roman", "font_size": 12, "font_color": "#FFF5EE", "bold": True, "align": "center", "valign": "vcenter", "text_wrap": True, "pattern": 1, "fg_color": "#4169E1", "num_format": "0.00"})
        # use this sheet to record the statistic data
        summary_sheet_name = "Summary" # this name will be referred in detail sheet
        summary_sheet = book.add_worksheet(summary_sheet_name)
        # see https://xlsxwriter.readthedocs.io/worksheet.html#merge_range
        summary_sheet.merge_range(0, 0, 1, 0, "Sample Name", header_style)
        summary_sheet.merge_range(0, 1, 1, 1, "Mean Squared Error", header_style)
        max_ncomp = max([len(fitted.statistic) for fitted in data])
        for i in range(max_ncomp):
            summary_sheet.merge_range(0, i*13+3, 0, i*13+14, "Component {0}".format(i+1), header_style)
            summary_sheet.write(1, i*13+3, "Fraction", header_style)
            summary_sheet.write(1, i*13+4, "Mean (μm)", header_style)
            summary_sheet.write(1, i*13+5, "Median (μm)", header_style)
            summary_sheet.write(1, i*13+6, "Mode (μm)", header_style)
            summary_sheet.write(1, i*13+7, "Variance", header_style)
            summary_sheet.write(1, i*13+8, "Standard Deviation", header_style)
            summary_sheet.write(1, i*13+9, "Skewness", header_style)
            summary_sheet.write(1, i*13+10, "Kurtosis", header_style)
            summary_sheet.write(1, i*13+11, "Beta", header_style)
            summary_sheet.write(1, i*13+12, "Eta", header_style)
            summary_sheet.write(1, i*13+13, "Location", header_style)
            summary_sheet.write(1, i*13+14, "X Offset", header_style)

        # set widths of columns
        # see https://xlsxwriter.readthedocs.io/worksheet.html#worksheet-set-column
        summary_sheet.set_column(0, 0, 16)
        summary_sheet.set_column(1, 1, 16)
        summary_sheet.set_column(2, max_ncomp*13+1, 16)

        for row, fitted_data in enumerate(data, 2):
            # take this way to let color changes alternately
            if row % 2 == 0:
                current_global_style = global_style
                current_mse_style = mse_style
                current_fraction_style = fraction_style
            else:
                current_global_style = dark_global_style
                current_mse_style = dark_mse_style
                current_fraction_style = dark_fraction_style

            summary_sheet.write(row, 0, fitted_data.name, current_global_style)
            summary_sheet.write(row, 1, fitted_data.mse, current_mse_style)
            for i, comp in enumerate(fitted_data.statistic):
                summary_sheet.write(row, i*13+3, comp.get("fraction"), current_fraction_style)
                summary_sheet.write(row, i*13+4, comp.get("mean"), current_global_style)
                summary_sheet.write(row, i*13+5, comp.get("median"), current_global_style)
                summary_sheet.write(row, i*13+6, comp.get("mode"), current_global_style)
                summary_sheet.write(row, i*13+7, comp.get("variance"), current_global_style)
                summary_sheet.write(row, i*13+8, comp.get("standard_deviation"), current_global_style)
                summary_sheet.write(row, i*13+9, comp.get("skewness"), current_global_style)
                summary_sheet.write(row, i*13+10, comp.get("kurtosis"), current_global_style)
                summary_sheet.write(row, i*13+11, comp.get("beta"), current_global_style)
                summary_sheet.write(row, i*13+12, comp.get("eta"), current_global_style)
                summary_sheet.write(row, i*13+13, comp.get("loc"), current_global_style)
                summary_sheet.write(row, i*13+14, comp.get("x_offset"), current_global_style)

        detail_sheet_name = "Detail"
        detail_sheet = book.add_worksheet(detail_sheet_name)
        detail_sheet.write(0, 0, "Sample Name", header_style)
        detail_sheet.write(0, 1, "Series Name", header_style)
        detail_sheet.set_column(0, 2, 16)
        detail_sheet.set_column(3, 3+len(classes)-1, 12)
        for col, value in enumerate(classes, 3):
            detail_sheet.write(0, col, value, header_style)
        
        row = 1
        for data_index, fitted_data in enumerate(data):
            if data_index % 2 == 0:
                current_global_style = detail_global_style
            else:
                current_global_style = dark_detail_global_style
            
            ncomp = len(fitted_data.statistic)
            # rows: target + fitted sumn + components
            detail_sheet.merge_range(row, 0, row+ncomp+1, 0, fitted_data.name, current_global_style)
            left = fitted_data.statistic[0]["loc"]
            
            # filling non-existent cells with 0
            for i in range(row, row+ncomp+2):
                # 3 to 3 + left (exclusive) is the left blank region
                for col in range(3, 3+left):
                    detail_sheet.write(i, col, 0, current_global_style)
                # 3+left+len(target[1]) to 3+len(classes) (exclusive) is the right blank region
                for col in range(3+left+len(fitted_data.target[1]), 3+len(classes)):
                    detail_sheet.write(i, col, 0, current_global_style)
            # target row
            for col, value in enumerate(fitted_data.target[1], 3+left):
                detail_sheet.write(row, col, value, current_global_style)
            detail_sheet.write(row, 1, "Target", current_global_style)
            row += 1
            # sum row
            for col, value in enumerate(fitted_data.sum[1], 3+left):
                detail_sheet.write(row, col, value, current_global_style)
            detail_sheet.write(row, 1, "Fitted Sum", current_global_style)
            row += 1
            # coponent rows
            for component_index, component in enumerate(fitted_data.components):
                detail_sheet.write(row, 1, "C{0}".format(component_index+1), current_global_style)
                for col, value in enumerate(component[1], 3+left):
                    # 1. use values
                    # detail_sheet.write(row, col, value, current_global_style)
                    # 2. use formula
                    # see 
                    detail_sheet.write(to_cell_name(row, col), "=WEIBULL({0}!{1}+{2}, {0}!{3}, {0}!{4}, FALSE)*{0}!{5}".format(
                        summary_sheet_name,
                        to_cell_name(data_index+2, component_index*13+14),
                        col-3-left,
                        to_cell_name(data_index+2, component_index*13+11),
                        to_cell_name(data_index+2, component_index*13+12),
                        to_cell_name(data_index+2, component_index*13+3)), current_global_style)
                row += 1
        
        
        # add charts to workbook
        style_file_path = "./settings/chart_styles.json"
        # see https://xlsxwriter.readthedocs.io/chart.html
        try:
            style_file = open(style_file_path, "r")
        except:
            logging.exception("The chart style file can not been open, file path:\n{0}".format(style_file_path), stack_info=True)
            self.sigWorkFinished.emit(False)
            return
        chart_styles  = json.load(style_file)
        chart_sheet = book.add_worksheet("Charts")
        column_chart_number = chart_styles["column_chart_number"]
        x_margin = chart_styles["x_margin"]
        y_margin = chart_styles["y_margin"]
        width = chart_styles["size"]["x_scale"]*480
        height = chart_styles["size"]["y_scale"]*288
        
        row = 1
        for i, fitted_data in enumerate(data):
            chart = book.add_chart({'type': 'scatter'})
            ncomp = len(fitted_data.components)
            ncols = len(classes)
            chart.set_x_axis(chart_styles["x_axis"])
            chart.set_y_axis(chart_styles["y_axis"])
            chart.set_title({**chart_styles["title"], "name": [detail_sheet_name, row, 0]})
            chart.set_legend(chart_styles["legend"])
            chart.set_size(chart_styles["size"])
            for j in range(ncomp+2):
                if j  == 0:
                    style = chart_styles["target_series"]
                elif j == 1:
                    style = chart_styles["fitted_series"]
                else:
                    style = chart_styles["component_series"]
                chart.add_series({**{"name": [detail_sheet_name, row, 1],
                                    "categories": [detail_sheet_name, 0, 3, 0, 3+ncols-1],
                                    "values": [detail_sheet_name, row, 3, row, 3+ncols-1]}, **style})
                row += 1

            x_offset = (i%column_chart_number) * (width+x_margin)
            y_offset = (i//column_chart_number) * (height+y_margin)
            chart_sheet.insert_chart('A1', chart, {"x_offset": x_offset, "y_offset": y_offset})
        try:
            book.close()
            self.sigWorkFinished.emit(True)
        except Exception:
            logging.exception("File saving failed.", stack_info=True)
            self.sigWorkFinished.emit(False)
