"""Two additional validation checks for the attention metric.

These checks only ADD to the existing pipeline; they import and reuse its model
loader, attention reducer, metric and probe files, and write their own results
under ``outputs/metric_validation/``. Nothing in the existing packages, outputs
or tests is modified.

  first_token  can the metric detect the strongest known sink (position 0)?
  near_block   the key contrasts measured on the near buffer instead of far Q.
"""
