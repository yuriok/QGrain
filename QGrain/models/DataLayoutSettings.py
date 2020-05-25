__all__ = ["DataLayoutError", "DataLayoutSettings"]

class DataLayoutError(Exception):
    """Raises while the data layout settings are invalid."""
    pass


class DataLayoutSettings:
    """
    The class to represent the layout setting of raw data file.
    All types of raw data files can be regarded as the table(s) that contains rows and columns.
    For the grain size distribution files, it should be the following format:
        * The first valid row should be the headers (i.e. the classes of grain size).
        * The following valid rows should be the distributions of samples under the grain size classes.
        * The first valid column shoud be the name (i.e. id) of samples.
    To make it more flexible, we use this setting to control the data loader.
    """
    def __init__(self, classes_row=0, sample_name_column=0,
                 distribution_start_row=1, distribution_start_column=1):
        # make sure the types are int in other codes
        assert type(classes_row) == int
        assert type(sample_name_column) == int
        assert type(distribution_start_row) == int
        assert type(distribution_start_column) == int

        # handle these errors at front-end
        if classes_row < 0 or \
            sample_name_column < 0 or \
            distribution_start_row < 0 or \
            distribution_start_column < 0:
            raise DataLayoutError("Row or column index must be non-negative.")
        if classes_row >= distribution_start_row:
            raise DataLayoutError("The start row index of distribution data must be greater than the row index of classes.")
        if sample_name_column >= distribution_start_column:
            raise DataLayoutError("The start column index of distribution data must be greater than the column index of sample names.")

        self.__classes_row = classes_row
        self.__sample_name_column = sample_name_column
        self.__distribution_start_row = distribution_start_row
        self.__distribution_start_column = distribution_start_column

    @property
    def classes_row(self) -> int:
        return self.__classes_row

    @property
    def sample_name_column(self) -> int:
        return self.__sample_name_column

    @property
    def distribution_start_row(self) -> int:
        return self.__distribution_start_row

    @property
    def distribution_start_column(self) -> int:
        return self.__distribution_start_column

    def __str__(self):
        return "Class Row Index: {0}, Sample Name Column: {1}, Distribution Start Row: {2}, Distribution Start Column: {3}.".format(
            self.classes_row, self.sample_name_column, self.distribution_start_row, self.distribution_start_column)
