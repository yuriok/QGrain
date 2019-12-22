import csv
import json
import logging
import os
from datetime import date, datetime, time
from typing import List

import numpy as np
import xlsxwriter
import xlwt
from PySide2.QtCore import QObject, Signal
from xlwt import Formula

from algorithms import DistributionType
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


class DataWriter(QObject):
    sigWorkFinished = Signal(bool)
    logger = logging.getLogger("root.data.DataWriter")
    gui_logger = logging.getLogger("GUI")

    def __init__(self):
        super().__init__()
        self.style_file_path = "./settings/chart_styles.json"
        # see https://xlsxwriter.readthedocs.io/chart.html
        self.draw_charts = True

    def on_settings_changed(self, settings: dict):
        for key, value in settings.items():
            if key == "draw_charts":
                self.draw_charts = value
                self.logger.info("The draw_charts option has been turned to [%s].", value)
            else:
                raise NotImplementedError(key)

    def try_save_data(self, filename, classes: np.ndarray, data: List[FittedData], file_type: str):
        if filename is None or filename == "":
            raise ValueError(filename)
        if os.path.exists(filename):
            self.logger.warning("This file has existed and will be replaced. Filename: %s.", filename)
        
        if file_type == "xlsx":
            self.try_save_as_excel(filename, classes, data, True)
        elif file_type == "xls":
            self.try_save_as_excel(filename, classes, data, False)
        elif file_type == "csv":
            self.try_save_as_csv(filename, data)
        else:
            raise NotImplementedError(file_type)

    # If type is csv,
    # it will use built-in csv module to handle this request.
    # It will write the statistic data only.
    # If you need the detailed data including the target data to fit,
    # fitted data of summing all component, and fitted data of each component,
    # please use `try_save_as_excel`
    # If you need the charts automatically drew in workbook,
    # please use `try_save_as_excel` func and pass `is_xlsx`=`True`.
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
                    row.append(comp.get("x_offset"))
                w.writerow(row)
            self.sigWorkFinished.emit(True)

        except Exception:
            self.logger.exception("File saving failed.", stack_info=True)
            self.gui_logger.error(self.tr("File saving failed, check the permission and occupation please."))
            self.sigWorkFinished.emit(False)
        finally:
            f.close()

    def try_save_as_excel(self, filename, classes, data: List[FittedData], is_xlsx: bool):
        # the local func to check the value validation
        # must block unaccepted values before calling the `write` func of worksheet
        def check_value(value):
            accepted_types = (bool, datetime, date, time, Formula)
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
        
        if is_xlsx:
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
            mse_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00E+00")
            dark_mse_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00E+00")
            fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 1", num_format_str="0.00%")
            dark_fraction_style = xlwt.easyxf("font: name Times New Roman, height 240; align: vert centre, horiz center; pattern: pattern solid, fore_colour 22", num_format_str="0.00%")
            header_style = xlwt.easyxf("font: name Times New Roman, height 320, bold on, colour 1; align: wrap on, vert centre, horiz center; pattern: pattern solid, fore_colour 30", num_format_str="0.00")
            # use this sheet to record the statistic data
            summary_sheet_name = "Summary" # this name will be referred in detailed sheet
            summary_sheet = book.add_sheet(summary_sheet_name)
            detail_sheet_name = "Detail"
            detail_sheet = book.add_sheet(detail_sheet_name)
        
        max_ncomp = max([len(fitted.statistic) for fitted in data])
        COMPONENTS = 11
        COLUMN_SPAN = COMPONENTS + 1
        # Sheet 1.
        # Headers of summary sheet
        mwrite(summary_sheet, 0, 0, 1, 0, "Sample Name", header_style)
        mwrite(summary_sheet, 0, 1, 1, 1, "Mean Squared Error", header_style)
        for i in range(max_ncomp):
            mwrite(summary_sheet, 0, i*COLUMN_SPAN+3, 0, i*COLUMN_SPAN+13, "Component {0}".format(i+1), header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+3, "Fraction", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+4, "Mean (μm)", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+5, "Median (μm)", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+6, "Mode (μm)", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+7, "Variance", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+8, "Standard Deviation", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+9, "Skewness", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+10, "Kurtosis", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+11, "Beta", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+12, "Eta", header_style)
            write(summary_sheet, 1, i*COLUMN_SPAN+13, "X Offset", header_style)
    
        # set widths of columns
        set_col(summary_sheet, 0, 16)
        set_col(summary_sheet, 1, 12)
        for i in range(2, max_ncomp*COLUMN_SPAN+2):
            set_col(summary_sheet, i, 8)
        # Contents of summary sheet
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
            write(summary_sheet, row, 0, fitted_data.name, current_global_style)
            write(summary_sheet, row, 1, fitted_data.mse, current_mse_style)
            for i, comp in enumerate(fitted_data.statistic):
                write(summary_sheet, row, i*COLUMN_SPAN+3, comp.get("fraction"), current_fraction_style)
                write(summary_sheet, row, i*COLUMN_SPAN+4, comp.get("mean"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+5, comp.get("median"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+6, comp.get("mode"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+7, comp.get("variance"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+8, comp.get("standard_deviation"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+9, comp.get("skewness"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+10, comp.get("kurtosis"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+11, comp.get("beta"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+12, comp.get("eta"), current_global_style)
                write(summary_sheet, row, i*COLUMN_SPAN+13, comp.get("x_offset"), current_global_style)
        
        # Sheet 2.
        # Headers of detail sheet
        write(detail_sheet, 0, 0, "Sample Name", header_style)
        write(detail_sheet, 0, 1, "Series Name", header_style)
        set_col(detail_sheet, 0, 16)
        set_col(detail_sheet, 1, 16)
        for i in range(3, 3+len(classes)):
            set_col(detail_sheet, i, 12)
        for col, value in enumerate(classes, 3):
            write(detail_sheet, 0, col, value, header_style)
        # Contents of detail sheet
        row = 1
        for data_index, fitted_data in enumerate(data):
            if data_index % 2 == 0:
                current_global_style = detail_global_style
            else:
                current_global_style = dark_detail_global_style
            ncomp = len(fitted_data.statistic)
            # rows: target + fitted sumn + components
            mwrite(detail_sheet, row, 0, row+ncomp+1, 0, fitted_data.name, current_global_style)
            left = fitted_data.statistic[0]["x_offset"]-1
            # left is the non-zero data start index
            
            # filling non-existent cells with 0
            for i in range(row, row+ncomp+2):
                # 3 to 3 + left (exclusive) is the left blank region
                for col in range(3, 3+left):
                    write(detail_sheet, i, col, 0, current_global_style)
                # 3+left+len(target[1]) to 3+len(classes) (exclusive) is the right blank region
                for col in range(3+left+len(fitted_data.target[1]), 3+len(classes)):
                    write(detail_sheet, i, col, 0, current_global_style)
            # target row
            for col, value in enumerate(fitted_data.target[1], 3+left):
                write(detail_sheet, row, col, value, current_global_style)
            write(detail_sheet, row, 1, "Target", current_global_style)
            row += 1
            # sum row
            for col, value in enumerate(fitted_data.sum[1], 3+left):
                write(detail_sheet, row, col, value, current_global_style)
            write(detail_sheet, row, 1, "Fitted Sum", current_global_style)
            row += 1
            # coponent rows
            if fitted_data.distribution_type == DistributionType.Normal:
                func_name = "NORMDIST"
            elif fitted_data.distribution_type == DistributionType.Weibull:
                func_name = "WEIBULL"
            else:
                raise NotImplementedError(fitted_data.distribution_type)
            for component_index, component in enumerate(fitted_data.components):
                write(detail_sheet, row, 1, "C{0}".format(component_index+1), current_global_style)
                for col, value in enumerate(component[1], 3+left):
                    # 1. use values
                    # write(detail_sheet, row, col, value, current_global_style)
                    # 2. use formula
                    if is_xlsx:
                        write(detail_sheet, row, col, "={5}({1}, {0}!{2}, {0}!{3}, FALSE)*{0}!{4}".format(
                            summary_sheet_name,
                            col-3-left+1,
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+11),
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+12),
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+3),
                            func_name), current_global_style)
                    else:
                        write(detail_sheet, row, col, xlwt.Formula("{5}({1}, {0}!{2}, {0}!{3}, FALSE)*{0}!{4}".format(
                            summary_sheet_name,
                            col-3-left+1,
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+11),
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+12),
                            to_cell_name(data_index+2, component_index*COLUMN_SPAN+3),
                            func_name)), current_global_style)
                row += 1

        # Save file if it is xls or no need to draw charts
        if not is_xlsx or not self.draw_charts:
            try:
                if is_xlsx:
                    book.close()
                else:
                    book.save(filename)
                self.logger.info("Excel (97-2003) workbook file has been saved. Filename: [%s].", filename)
                self.gui_logger.info(self.tr("File has been saved."))
                self.sigWorkFinished.emit(True)
                return
            except Exception:
                self.logger.exception("Excel (97-2003) Workbook File saving failed. Filename: [%s].", filename, stack_info=True)
                self.gui_logger.error(self.tr("File saving failed, check the permission and occupation please."))
                self.sigWorkFinished.emit(False)
                return

        # add charts to workbook
        chart_styles = None
        column_chart_number = 5
        x_margin = 10
        y_margin = 10
        width = 480
        height = 288
        try:
            style_file = open(self.style_file_path, "r")
            chart_styles = json.load(style_file)
            column_chart_number = chart_styles["column_chart_number"]
            x_margin = chart_styles["x_margin"]
            y_margin = chart_styles["y_margin"]
            width = chart_styles["size"]["x_scale"]*480
            height = chart_styles["size"]["y_scale"]*288
        except:
            self.logger.exception("The chart style file can not been opened, use default styles next. File path: [%s].", self.style_file_path, stack_info=True)
            self.gui_logger.warning(self.tr("The chart style file can not been opened, use default styles next."))
        
        chart_sheet = book.add_worksheet("Charts")
        row = 1
        for i, fitted_data in enumerate(data):
            chart = book.add_chart({'type': 'scatter'})
            ncomp = len(fitted_data.components)
            ncols = len(classes)
            if chart_styles is not None:
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
            else:
                chart.set_title({"name": [detail_sheet_name, row, 0]})
                for j in range(ncomp+2):
                    chart.add_series({"name": [detail_sheet_name, row, 1],
                                      "categories": [detail_sheet_name, 0, 3, 0, 3+ncols-1],
                                      "values": [detail_sheet_name, row, 3, row, 3+ncols-1]})
                    row += 1

            x_offset = (i%column_chart_number) * (width+x_margin)
            y_offset = (i//column_chart_number) * (height+y_margin)
            chart_sheet.insert_chart('A1', chart, {"x_offset": x_offset, "y_offset": y_offset})

        try:
            book.close()
            self.logger.info("File has been saved. Filename: [%s].", filename)
            self.gui_logger.info(self.tr("File has been saved."))
            self.sigWorkFinished.emit(True)
            return
        except Exception:
            self.logger.exception("File saving failed.", stack_info=True)
            self.gui_logger.error(self.tr("File saving failed, check the permission and occupation please."))
            self.sigWorkFinished.emit(False)
            return
