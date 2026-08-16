"""
ONE-TIME SCRIPT - Run this ONCE before starting pipelines
==========================================================
Purpose : Read existing CSV → extract raw columns only
          → upload to NEW feature group karachi_aqi_raw (v1)

Why     : Automation pipelines need a raw data starting point
          Hourly pipeline will append to this feature group
          Daily training will fetch from this feature group

Columns : timestamp, aqi, pm10, pm25, co, o3
          (5 raw pollutants — what we have from original fetch)

Run     : python re_upload_raw.py
"""