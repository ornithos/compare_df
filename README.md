## Compare dataframes

_This utility was originally written for the DECOVID project, funded by EPSRC. I've copied it out to my GH account so it resides in a publicly accessible repo._

Compare two dataframes `x1`, `x2` which may have different rows and columns, and be ordered differently. This function 
will perform a comparison by matching rows based on the `index_cols`, and using approximate equality for
numerical data to avoid false negatives due to numerical issues. The function returns a `CompareDataFrameReport` object 
from which:

* the number of matching rows and columns
* the number of matching elements within each column
* the types of each column

can be accessed. By default, the function provides a text version of the report (see below). One can also
request that the matching and/or non-matching sections of the dataframes be returned for further inspection.

An example of the report text is provided below, comparing the CSV and RDS versions of the 2005 BrainIT 
"Neurological Status" table.

```
======================================================================
                          COMPARE DATAFRAMES                          
======================================================================

STRUCTURE:
----------

DataFrame1: shape (12129, 10)
DataFrame2: shape (12129, 10)

DataFrame 1:
  Num. additional rows: 4070
  - of which:
      - nunique Patient_Id: 153
      - nunique Time_Stamp: 3770
  Additional columns:  None

DataFrame 2:
  Num. additional rows: 4070
  - of which:
      - nunique Patient_Id: 153
      - nunique Time_Stamp: 3770
  Additional columns:  None

MATCHED contents:
-----------------

(atol=1.0e-06, rtol=1.0e-04)

Complete matching rows: 0/8059 (0.0%). Num. matching elements per column: 

          Name                Matches              Type differences
------------------------ ----------------------- ------------------------
  - GCS_Eye                8059/8059 (100.00%)    object <-> category
  - GCS_Motor              8059/8059 (100.00%)    object <-> category
  - GCS_Verbal             8059/8059 (100.00%)    object <-> category
  - Left_Pupil_Reaction    8057/8059 ( 99.98%)    object <-> category
  - Left_Pupil_Size        8059/8059 (100.00%)    object <-> category
  - Right_Pupil_Reaction   8057/8059 ( 99.98%)    object <-> category
  - Right_Pupil_Size       8059/8059 (100.00%)    object <-> category
  - BIT_comment               0/8059 (  0.00%)    float64 <-> object

(EOF)
```


**A guided tour of the example report**

The summary splits into two parts. The first part displays the differences between the row/column structure, and the
number of non-matching rows of each table. The number of unique values in the specified join columns (used to index each
row) are also given. In this case we can see that the non-matched rows contain details of 153 patients, or 3770 datetime
values. Since the DataFrames are of the same dimension (or shape), the number of non-matching or "additional" rows in 
each DataFrame is the same. Both DataFrames contain the same columns, hence the "Additional columns" of each DataFrame
are `None`.

The second part of the summary is concerned only with the subsection of the two DataFrames which match. Observe that the
first half gives us that each DataFrame has 12129 rows, but the second half compares only 8059 of these. The number of
matching elements in each column are shown in the table, where numerical columns use approximate equality according to 
the absolute and relative tolerances displayed (`atol`/`rtol`). The columns used to join the DataFrame (in this case 
"Patient_Id" and "Time_Stamp") are omitted from this table since these will match 100% of the time by construction.
Where columns are of different datatype, the `Type differences` column will show the difference with the first and
second dataframe's column types shown respectively.

**Notes**

Since we cannot assume that DataFrames will order the rows in the same way, the rows are matched based on what is 
assumed to be a unique index, defined by the supplied `index_cols`. (Under the hood, these columns are used in order to
perform an outer join between the two dataframes.)
Where the index columns do not define a unique index to each row, the report output is undefined, and should not be
trusted. Therefore a check is made for duplication, and if it is found, a warning will display at the close of the 
function regardless of the `verbose` level. If using the function in interactive mode, one should either drop duplicate
rows or provide/construct alternative index column(s) to join upon. When using from the command line, the 
`--drop-duplicates` flag is available to drop all duplicate rows, keeping only the first of each instance.

**From python**:

```python
compare_dataframes(x1, x2, index_cols,
                       atol=1e-6, rtol=1e-4,
                       return_nonmatching_data=False,
                       return_matching_data=False,
                       verbose=1)
```

The python interface has three positional arguments, respectively, the two dataframes to compare, and a list of one or
more columns which form a unique index for each row. By default, the function will 
* print the text report to stdout, and
* return (i) whether the DataFrames are identical [`Bool`] and (ii) the report object [`CompareDataFrameReport`]. 

The check for (approximate) equality in (i) will check the structure of the DataFrames (up to reordering), the 
values, and the types. If any of these three comparisons return `False`, the equality check
will also return `False`. Less stringent criteria are available from the `report` object in (ii), which can be accessed
via:

```python
report.dataframes_are_equal(check_types=True, only_check_matched=False)
```
Finally, the non-matching rows in each DataFrame may be returned in the function output via the argument 
`return_nonmatching_data=True`; these will be the (iii) and (iv) output values. Note that these portions will
still only include the _matching columns_, since the non-equality between disjoint columns is clear. (The matching 
portion of the DataFrames is also available using `return_matching_data=True`, which will return as another two
outputs after the non-matching data.)

**From the command line**:

The function can also be called from the command line. This currently supports reading from CSV and Parquet files only. 
The syntax is as follows:

```shell
python compare_dataframes.py path/to/dataframe1.csv path/to/dataframe2.parquet --index Patient_Id --index Time_Stamp
```

The function takes two positional arguments which are the paths of the relevant files (either CSV or Parquet), and 
(possibly) multiple index columns, specified by the `--index` keyword. The CLI only currently reads CSV or Parquet files
and their file type will be inferred from their extension. To explicitly specify the file type, use `--format1` and/or
`--format2` which specify the format ("csv"/"parquet") for file 1 or 2 respectively. Two options beyond the Python
interface exist:

* `--parquet-keep-nones`: Since the Parquet format can encode both `NaN` and `None`/`NULL`, this causes issues with 
column type designations. For example a `float64` column supports only `NaN`, and on encountering a `None` will fall 
back to `object` type, failing the equality comparison even if all numeric values are equal. By default, the CLI will
convert `None`s to `NaN`s, but this is not always desirable. To avoid this implicit conversion, add the 
`--parquet-keep-nones` flag.
* `--drop-duplicates`: The output of this  comparison tool is not defined when either DataFrame includes duplicate values (acc. to the index).
The `--drop-duplicates` flag drops all duplicate rows of both DataFrames, keeping only
the first of each. This may often not be desirable for comparing DataFrames, but is available at the CLI since few other
options are available to a user.

