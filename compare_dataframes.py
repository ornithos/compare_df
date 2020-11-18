import pandas as pd
import numpy as np
import pytz
import argparse, sys
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_datetime64_any_dtype
from warnings import warn


df_cmp_ascii = """
     _                   __                          
    | |      _         / __)                      
  _ | | ____| |_  ____| |__ ____ ____ ____   ____ 
 / || |/ _  |  _)/ _  |  __) ___) _  |    \ / _  )
( (_| ( ( | | |_( ( | | | | |  ( ( | | | | ( (/ / 
 \____|\_||_|\___)_||_|_| |_|   \_||_|_|_|_|\____)
                                                  
                                    _                  
                                   (_)                 
  ____ ___  ____  ____   ____  ____ _  ___  ___  ____  
 / ___) _ \|    \|  _ \ / _  |/ ___) |/___)/ _ \|  _ \ 
( (__| |_| | | | | | | ( ( | | |   | |___ | |_| | | | |
 \____)___/|_|_|_| ||_/ \_||_|_|   |_(___/ \___/|_| |_|
                 |_|                                   
"""
def coerce_timezone_to_utc(x):
    """
        coerce_timezone_to_utc(x)

    `x` should be a series with a DateTime type. Different constructors/
    platforms may result in different (although hopefully equivalent)
    timezones.

    This function is a noop if `x` is already in datetime..[,utc] format,
    will coerce if `x` is just in (naive) datetime.. format, and throw
    an Exception otherwise.
    """

    assert isinstance(x, pd.core.series.Series)
    if isinstance(x.dtype, object):
        # column read in as text. Let pandas infer format.
        x = pd.to_datetime(x)

    if isinstance(x.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        x_tz = x.dtype.tz
        if x_tz == pytz.UTC:
            return x
        else:
            raise TypeError("Expecting UTC time zone; received: {:s}".format(str(type(x_tz))))
    else:
        # Non-localized / non-time-zone datetime appears to fallback to numpy datetype
        if isinstance(x.dtype, np.dtype):
            return x.dt.tz_localize('utc')
        else:
            raise TypeError("Expecting datetime Series / vector; received: {:s}".format(
                str(type(x.dtype))))


def read_parquet(filename, none_to_NaN=False, verbose=True):
    """
    read_parquet(filename, none_to_NaN=False, downcast_float=False)

    This mostly exists to avoid writing the none_to_NaN logic twice.
    Parquet can handle both NULL and NaN values, where the former
    are represented as `None` in python. Unfortunately `float`
    datatypes cannot handle these, and the resulting column is
    represented as an object type, which `numpy` functions, such as
    `np.isclose` cannot handle. This then usually breaks the
    downstream checks.
    """
    verbose and print("Loading dataframe...")
    df = pd.read_parquet(filename)
    if none_to_NaN:
        for c, x in df.items():
            nones = x.isnull()
            if nones.sum() > 0:
                df.loc[nones, c] = np.NaN
    return df


def read_csv(filename, verbose=True):
    """
    read_csv(filename)

    ~~A companion for `read_parquet` for which `downcast_float=True` will
    downcast to float32 if safe to do so.~~
    """
    verbose and print("Loading dataframe...")
    df = pd.read_csv(filename)
    # if downcast_float:
    #     for c, x in df.items():
    #         if x.dtype.kind == 'f':
    #             df[c] = pd.to_numeric(x, downcast="float")
    return df


class CompareDataFrameReport(object):
    """
    A CompareDataFrameReport object is a placeholder for various values to be calculated
    when comparing two data frames. This has fields for structural comparison (e.g.
    number of rows, columns of each dataframe, and their intersection) as well as
    elementwise comparison (number of equal fields in each column).

    An "empty" report can be instantiated with the constructor:
    ```
    report = CompareDataFrameReport(atol, rtol)
    ```
    where the `atol` and `rtol` are the absolute and relative tolerance for comparing values
    between the data frames. See `?np.isclose` for more details. There are three methods for
    adding data to the report:

    * `report.add_structure(nrow1, nrow2, nrowboth, columns1, columns2)`

    where `nrow1` is the number of rows *unique* to dataframe 1 (sim. for `nrow2`, m.m.).
    `nrowboth` is the number of rows *common* to both dataframes, and `columns1`/`columns2`
    is the list of column names for both dataframes.

    * `report.add_unique_index_counts(index_cols, nunique_index_cols1, nunique_index_cols2)`

    where `index_cols` are the index columns of the dataframes used to join/merge the tables,
    `nunique_index_cols1/2` are the number of unique values of each index for the indices
    which did *not* match. E.g. if one index is patient_id, this gives a count of the
    number of patients that have additional rows in dataframes 1/2.

    * `report.add_comparison(nrows_eq, nrows_eq_per_col)`.

    This is where we add the comparison of elements between the dataframes where both the
    indices and the columns match. (I.e. the subset of both tables which is directly
    comparable.) `nrows_eq` is the number of *complete* rows which are the same between
    the tables. `nrows_eq_per_col` is a vector giving the number of matching elements
    in each column. E.g. if only one column were different between datasets, this would
    result in a list of [n,..., n, 0, n, ...., n] which results in no complete matching
    rows, but d-1 matching columns.

    To create the report, hit `report.create_report()`.
    """
    def __init__(self, atol, rtol):
        self.atol = atol
        self.rtol = rtol
        self.checklist = np.zeros(4)

    def add_structure(self, nrow1, nrow2, nrowboth, columns1, columns2):
        self.checklist[0] = 1
        # row statistics
        self.nrow1_only = nrow1
        self.nrow2_only = nrow2
        self.nrow1 = nrow1 + nrowboth
        self.nrow2 = nrow2 + nrowboth
        self.nrowboth = nrowboth

        # column statistics
        self.columns1 = list(columns1)
        self.columns2 = list(columns2)
        self.cols1_only = list(set(columns1) - set(columns2))
        self.cols2_only = list(set(columns2) - set(columns1))
        cols_both = list(set(columns1).intersection(columns2))
        self.cols_both = list(filter(lambda x: x in cols_both, columns1)) # keep order
        self.ncol1 = len(columns1)
        self.ncol2 = len(columns2)
        self.ncol1_only = len(self.cols1_only)
        self.ncol2_only = len(self.cols2_only)
        self.ncolboth = len(self.cols_both)
    
    def add_types(self, dtypes1, dtypes2, indices1, indices2):
        assert len(dtypes1) == self.ncolboth
        assert len(dtypes2) == self.ncolboth
        self.checklist[1] = 1
        self.dtypes1 = dtypes1
        self.dtypes2 = dtypes2
        # The str equality here is to avoid pd/np bug see e.g. fb prophet #256
        self.dtypescmp = [str(t1) == str(t2) for (t1, t2) in zip(dtypes1, dtypes2)]
        self.dtypescmpstr = ["{:^25s}".format("-") if str(t1) == str(t2) else \
                             "    {:s} <-> {:s}".format(str(t1), str(t2))
                             for (t1, t2) in zip(dtypes1[indices1], dtypes2[indices2])]
        
        
    def add_unique_index_counts(self, index_cols, nunique_index_cols1, nunique_index_cols2):
        self.checklist[2] = 1
        self.index_cols = index_cols
        self.nunique_index_cols1 = nunique_index_cols1
        self.nunique_index_cols2 = nunique_index_cols2

    def add_comparison(self, nrows_eq, nrows_eq_per_col):
        self.checklist[3] = 1
        self.nrow_eq = nrows_eq
        self.nrow_eq_per_col = nrows_eq_per_col
        
    def create_report(self, verbose=1):
        # check everything has been added
        if not self.checklist.sum() == 4:
            if self.checklist[1] == 0 and self.nrow1_only == 0 and \
                self.nrow2_only == 0:
                pass
            else:
                todo = ["structure", "types", "index counts", "comparison"][~(self.checklist == 0)]
                raise RuntimeError("Cannot create report until the following have been " +
                                   "added: {:s}".format(", ".join(todo)))
        # begin report
        if verbose > 1:
            out = df_cmp_ascii
        else:
            out = "=" * 70 + "\n"
            out += "{:^70s}\n".format("COMPARE DATAFRAMES")
            out += "=" * 70 + "\n"
        # out += "-"*50 + "\n"
        out += "\nSTRUCTURE:\n"
        out += "-"*10 + "\n\n"
        out += "DataFrame1: shape ({:d}, {:d})\n".format(self.nrow1, self.ncol1)
        out += "DataFrame2: shape ({:d}, {:d})\n".format(self.nrow2, self.ncol2)
        if self.ncol1_only > 0 or self.nrow1_only > 0:
            out += "\nDataFrame 1:\n"
            out += "  Num. additional rows: {:d}\n".format(self.nrow1_only)
            if self.nrow1_only > 0:
                out += "  - of which:\n"
                for i, c in enumerate(self.index_cols):
                    out += "      - nunique {:s}: {:d}\n".format(c, self.nunique_index_cols1[i])
            out += "  Additional columns:  {:s}\n".format(str(self.cols1_only)
                                                    if self.ncol1_only > 0 else "None")
        else:
            out += "\nDataFrame 1: ✔ (all observations are common with DataFrame 2)\n"
        if self.ncol2_only > 0 or self.nrow2_only > 0:
            out += "\nDataFrame 2:\n"
            out += "  Num. additional rows: {:d}\n".format(self.nrow2_only)
            if self.nrow2_only > 0:
                out += "  - of which:\n"
                for i, c in enumerate(self.index_cols):
                    out += "      - nunique {:s}: {:d}\n".format(c, self.nunique_index_cols2[i])
            out += "  Additional columns:  {:s}\n".format(str(self.cols2_only)
                                                    if self.ncol2_only > 0 else "None")
        else:
            out += "\nDataFrame 2: ✔ (all observations are common with DataFrame 1)\n"

        out += "\nMATCHED contents:\n"
        out += "-" * 17 + "\n\n"
        out += "(atol={:.1e}, rtol={:.1e})\n\n".format(self.atol, self.rtol)
        out += "Complete matching rows: {:d}/{:d} ({:.1f}%). ".format(self.nrow_eq,
                                                self.nrowboth, self.nrow_eq/self.nrowboth*100)
        if self.nrow_eq < self.nrowboth:
            out += "Num. matching elements per column: \n\n"
            max_len = max([len(s) for s in self.cols_both])
            nrow_digits = int(np.ceil(np.log10(self.nrowboth)))
            cols_excl_join = list(filter(lambda x: x not in self.index_cols, self.cols_both))

            # Create table header
            out += "{:^{width}s}".format("Name", width=max_len+5) + \
                   "{:^{width}s}".format("Matches", width=nrow_digits*2+10) + \
                   " "*8 + "{:s}\n".format("Type differences")
            out += "-"*(max_len+5-1) + " " + "-"*(nrow_digits*2+10+6-1) + " " + "-"*24 + "\n"
            for i, c in enumerate(cols_excl_join):
                res = "{:{width}d}/{:{width}d} ({:6.2f}%)".format(self.nrow_eq_per_col[i], 
                                                                  self.nrowboth,
                                                                  self.nrow_eq_per_col[i] / self.nrowboth*100,
                                                                  width=nrow_digits)
                out += "  - {:{width}s}  {result:s}{types:s}\n".format(c,
                                                                       result=res, 
                                                                       width=max_len+1,
                                                                       types=self.dtypescmpstr[i])
        else:
            out += "\n    ✔ All elements in comparable sections are matching.\n"

        out += "\n(EOF)"
        return out

    def dataframes_are_equal(self, check_types=True, only_check_matched=False):
        struct_is_same = self.nrow1_only == 0 and \
            self.nrow2_only == 0 and \
            self.ncol1_only == 0 and \
            self.ncol2_only == 0
        types_are_same = all(self.dtypescmp)
        matched_are_same = (self.nrow_eq == self.nrowboth)

        types_are_same = True if not check_types else types_are_same
        struct_is_same = True if only_check_matched else struct_is_same
        return struct_is_same and types_are_same and matched_are_same


def basic_column_coercion(c1, c2, colname="?", err_str=""):
    """
    basic_column_coercion(c1, c2, colname="?", err_str="")
    
    Attempt to coerce Series `c1` and `c2` to the same data type. This is
    currently only written for the case when one is of the general `object`
    type and the other is more specific. The function returns the two
    Series successfully coerced to the same type, or throws an error which
    includes `err_str`.
    """
    type1, type2 = c1.dtype, c2.dtype
    
    if is_numeric_dtype(type1) and is_numeric_dtype(type2):
        # merge can handle conversion of numeric datatypes natively
        return c1, c2
    
    throw_error = True
    try:
        if type1 == object:
            if is_datetime64_any_dtype(type2):
                # datetime type in itself does not contain parsing info. Use inference.
                c1 = pd.to_datetime(c1)
            c1 = c1.astype(type2)
            if len(set(c1).intersection(c2)) > 0:
                throw_error = False
        if type2 == object:
            if is_datetime64_any_dtype(type1):
                # datetime type in itself does not contain parsing info. Use inference.
                c2 = pd.to_datetime(c2)
            c2 = c2.astype(type1)
            if len(set(c1).intersection(c2)) > 0:
                    throw_error = False
    except:
        # Will throw the RuntimeError anyway if this fails.
        pass
    
    if throw_error:
        raise RuntimeError(("Column '{:s}' is of different types ({:s}, {:s}). " + 
                           "{:s} ").format(colname, str(type1), str(type2), err_str))
    return c1, c2


def compare_columns(c1, c2, atol, rtol):
    """
    compare_columns(c1, c2, atol, rtol)
    
    Compare two Series `c1`, `c2` of the same length, returning a Boolean vector
    of the equality of the corresponding elements. For numeric data, the equality
    is approximate using `atol`, `rtol` (see `np.isclose`).
    """
    if is_numeric_dtype(c1):
        if not is_numeric_dtype(c2):
            return np.zeros_like(c1, bool)
        return np.isclose(c1, c2, atol=atol, rtol=rtol, equal_nan=True)
    
    elif is_string_dtype(c1):
        if not is_string_dtype(c2):
            return np.zeros_like(c1, bool)
        # pass through
   
    elif is_datetime64_any_dtype(c1):
        if not is_datetime64_any_dtype(c2):
            return np.zeros_like(c1, bool)
        # pass through
    else:
        raise NotImplementedError("Unexpected data type: {:s}".format(str(c1.dtype)))
    
    objcmpvals = c1.values == c2.values
    objcmpvals[(c1.isna().values & c2.isna()).values] = True
    return objcmpvals


def compare_dataframes(x1, x2, index_cols,
                       atol=1e-6, rtol=1e-4,
                       return_nonmatching_data=False,
                       return_matching_data=False,
                       verbose=1):
    """
        `compare_dataframes(x1, x2, index_cols)`

    Compare two dataframes, `x1` and `x2` by matching rows and columns
    and comparing the subset of `x1`/`x2` in common. The function will
    also provide statistics about the excluded subsets (if any) and
    return the set difference of rows (in either direction) and the
    intersection as dataframes.

    Arguments:
    :param x1: First dataframe to compare (pandas object).
    :param x2: Second dataframe to compare (pandas object).
    :param index_cols: The list of index column(s) to join on. For
        instance this might be `["Patient_Id", "Time_Stamp"]`. This
        defines how to match rows between the dataframes.
    :param atol: The absolute tolerance for difference between
        dataframe elements (see `np.isclose`). (Def: 1e-6)
    :param rtol: The relative tolerance for difference between
        dataframe elements (see `np.isclose`). (Def: 1e-4)
    :param return_nonmatching_data: return the subset of `x1` and `x2`
        which do not match (based on row indices). (Def: True)
    :param return_matching_data: return the subset of `x1` and `x2`
        which match (based on row indices). (Def: False)
    :param verbose: (0): print nothing, (>=1): print report

    Returns:
    - equality (Bool): Do the dataframes match exactly?
    - report object (CompareDataFrameReport).
    - x1_only (DataFrame): rows of `x1` which don't have equivalent in `x2`. (Optional)
    - x2_only (DataFrame): rows of `x2` which don't have equivalent in `x1`. (Optional)
    - x1_both (DataFrame): rows of `x1` which do have equivalent in `x2`. (Optional)
    - x2_both (DataFrame): rows of `x2` which do have equivalent in `x1`. (Optional)
    """
    
    # Collect types for comparison later
    x1_types = x1.dtypes
    x2_types = x2.dtypes

    # Merge dataframes on `index_cols` in order to match rows
    # ----------------------------------------------------

    # Cannot merge unless specified columns are same type.
    # If this is not so, we will attempt some basic coercion
    for c in index_cols:
        x1_type, x2_type = x1[c].dtype, x2[c].dtype
        if not str(x1_type) == str(x2_type):     # kinda gross I know, but np/pd dont cmp o.w.
            c1, c2 = basic_column_coercion(x1[c], x2[c], colname=c, 
                                  err_str="Cannot merge for index.")
            x1[c] = c1
            x2[c] = c2
            
    # Ensure timezones are comparable for the join (if applicable)
    time_cols1 = [c for c, d in zip(x1.columns, x1.dtypes) if is_datetime64_any_dtype(d)]
    time_cols2 = [c for c, d in zip(x2.columns, x2.dtypes) if is_datetime64_any_dtype(d)]
    for c in time_cols1:
        x1[c] = coerce_timezone_to_utc(x1[c])
    for c in time_cols2:
        x2[c] = coerce_timezone_to_utc(x2[c])

    # Merge reduced dataframes, and add original index into the merge
    x1_merge = x1[index_cols].reset_index()
    x2_merge = x2[index_cols].reset_index()

    # check for duplicates (merge is ~ undefined in this case)
    dup_ix = False
    if x1_merge.duplicated(subset=index_cols).sum() > 0 or \
            x2_merge.duplicated(subset=index_cols).sum() > 0:
        dup_ix = True

    # perform merge
    cmp = pd.merge(x1_merge, x2_merge, how="outer", left_on=index_cols,
                   right_on=index_cols, indicator=True)

    # Split dataframes into subsets which match, and which don't match
    # ----------------------------------------------------------------

    # Split out rows that are particular to exactly one dataframe
    ixs1_only = cmp.loc[cmp['_merge'] == "left_only", 'index_x']
    ixs2_only = cmp.loc[cmp['_merge'] == "right_only", 'index_y']

    # Get indices for matching rows
    ixs1_both = cmp.loc[cmp['_merge'] == "both", 'index_x']
    ixs2_both = cmp.loc[cmp['_merge'] == "both", 'index_y']

    # Ensure dataype of index is preserved (as NaNs may change available dtypes)
    ixs1_only, ixs1_both = ixs1_only.astype(x1.index.dtype), ixs1_both.astype(x1.index.dtype)
    ixs2_only, ixs2_both = ixs2_only.astype(x2.index.dtype), ixs2_both.astype(x2.index.dtype)

    # Add various metrics to the report
    # ----------------------------------------------------------------

    # Add some high level statistics to the report
    report = CompareDataFrameReport(atol, rtol)
    report.add_structure(len(ixs1_only), len(ixs2_only), len(ixs2_both),
                         x1.columns, x2.columns)

    # get columns common to both dataframes
    cols_both = report.cols_both
    cols_both_exjoin = list(filter(lambda x: x not in index_cols, cols_both))
    
    # add type comparison
    report.add_types(x1_types[[i for (i,x) in enumerate(x1.columns) if x in cols_both]],
                     x2_types[[i for (i,x) in enumerate(x2.columns) if x in cols_both]],
                     [i for i, x in enumerate(x1.columns) if x not in index_cols],
                     [i for i, x in enumerate(x2.columns) if x not in index_cols])
    
    # Compare matching parts of each dataframe
    eq_approx = []
    for c in cols_both_exjoin:
        eq_approx.append(compare_columns(x1.loc[ixs1_both, c], x2.loc[ixs2_both, c], atol, rtol))

    eq_approx = np.stack(eq_approx)
    nrows_eq = eq_approx.all(axis=0).sum()
    nrows_eq_per_col = eq_approx.sum(axis=1)
    report.add_comparison(nrows_eq, nrows_eq_per_col)

    # Summarise unique parts of each dataframe
    nunique_index_cols1 = [x1.loc[ixs1_only, c].nunique() for c in index_cols]
    nunique_index_cols2 = [x1.loc[ixs2_only, c].nunique() for c in index_cols]
    report.add_unique_index_counts(index_cols, nunique_index_cols1, nunique_index_cols2)

    # Create report
    report_out = report.create_report(verbose=verbose)
    if verbose > 0: print(report_out)
    if dup_ix:
        dup_warn = "\n" + "~!" * 50 + "\n"
        dup_warn += "\n{:^60s}\n".format("! Duplicate indices exist - the above may be inaccurate !")
        dup_warn += "\nTo remedy for duplicate indices, either:"
        dup_warn += "\n 1. Specify extra/alternative indices so that rows are uniquely identified."
        dup_warn += "\n 2. Use argument `--drop-duplicates` to keep only the first of each duplicate row.\n"
        dup_warn += "\n" + "~!" * 50 + "\n"
        warn(dup_warn)
    equality = report.dataframes_are_equal()

    # Return data
    if return_nonmatching_data:
        x1_only, x2_only = x1.loc[ixs1_only, :], x2.loc[ixs2_only, :]
        if return_matching_data:
            x1_both, x2_both = x1.loc[ixs1_both, :], x2.loc[ixs2_both, :]
            return equality, report, x1_only, x2_only, x1_both, x2_both
        else:
            return equality, report, x1_only, x2_only
    else:
        if return_matching_data:
            x1_both, x2_both = x1.loc[ixs1_both, :], x2.loc[ixs2_both, :]
            return equality, report, x1_both, x2_both
        else:
            return equality, report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compare two dataframes saved either as CSV or ' +
                                    'Parquet format.\n\nThis program uses approximate comparison ' +
                                    'rather than exact equality to handle differences e.g. due to ' +
                                    'numerical precision. Note also that at present this program ' +
                                    'does not make a distinction between NaN and NULL values.')

    parser.add_argument('df1', help="file name of first dataframe.")
    parser.add_argument('df2', help="file name of second dataframe.")
    parser.add_argument('--index', action="append", help="the columns on which" +
                        " to match/merge the dataframes. Multiple values can" +
                        " be specified.", required=True)
    parser.add_argument('--format1', default="", choices=["", "csv", "parquet"],
                        help="file format of both dataframes. To use a different " +
                        "file format for each dataframe, use the --format2 option.",
                        type=str.lower)
    parser.add_argument('--format2', action="store", default="",
                        choices=["", "csv", "parquet"],
                        help="file format of second dataframe. First " +
                        "dataframe will use the '--format' specified.",
                        type=str.lower)
    parser.add_argument('--atol', default=1e-6, type=float,
                        help="absolute tolerance for approximate equality.")
    parser.add_argument('--rtol', default=1e-4, type=float,
                        help="relative tolerance for approximate equality.")
    parser.add_argument('--quiet', action='store_true', help=
                        "Suppress all stdout.")
    parser.add_argument('-v', '--verbose', action='count', default=0, help= \
                        "Verbosity level: prints additional info to stdout.")
    parser.add_argument('--parquet-keep-nones', action='store_true', help= \
                        "Data dumped to parquet can result in None values as well" +
                        " as NaNs. By default these will be replaced with NaNs and" +
                        " coerced back to numeric. This flag will leave Nones as " +
                        " they are. However, since the resulting column is " +
                        " represented then as an `object` type, later logic usually fails.")
    parser.add_argument('--drop-duplicates', action='store_true', help="Drop any rows which" +
                        "are duplicates (in their entirety). Such rows can be problematic" +
                        "for calculating the order in which to match the dataframes together.")

    args = parser.parse_args()

    verbose = 0 if args.quiet else args.verbose + 1

    format1 = args.format1
    format2 = args.format2
    format1 = args.df1.split(".")[-1] if len(args.format1) == 0 else args.format1
    format2 = args.df2.split(".")[-1] if len(args.format2) == 0 else args.format2
    
    if format1.upper() == "CSV":
        df1 = read_csv(args.df1, verbose=(verbose>1))
    elif format1.upper() == "PARQUET":
        df1 = read_parquet(args.df1, not args.parquet_keep_nones, verbose=(verbose>1))
    else:
        raise NotImplementedError("Unexpected file format specified: '{:s}'".format(format1))
    
    if format2.upper() == "CSV":
        df2 = read_csv(args.df2, verbose=(verbose>1))
    elif format2.upper() == "PARQUET":
        df2 = read_parquet(args.df2, not args.parquet_keep_nones, verbose=(verbose>1))
    else:
        raise NotImplementedError("Unexpected file format specified: '{:s}'".format(format2))

    for c in args.index:
        assert c in df1.columns, "index column {:s} not found in df1".format(c)
        assert c in df2.columns, "index column {:s} not found in df2".format(c)


    if args.drop_duplicates:
        (verbose > 1) and print("Dropping duplicate rows...")
        df1 = df1.drop_duplicates(subset=args.index, keep="first")
        df2 = df2.drop_duplicates(subset=args.index, keep="first")

    (verbose > 1) and print("Performing comparison...")
    equal_dfs, report = compare_dataframes(df1, df2, args.index,
                       atol=args.atol, rtol=args.rtol,
                       return_nonmatching_data=False,
                       return_matching_data=False,
                       verbose=verbose)

    if equal_dfs:
        sys.exit(0)
    else:
        sys.exit(1)
