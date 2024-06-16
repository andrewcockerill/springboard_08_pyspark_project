# Databricks notebook source
# MAGIC %md
# MAGIC # PySpark/Azure Databricks Exercise
# MAGIC This notebook is used to perform ingestion of raw data files into a Spark session by means of an Azure Databricks notebook environment. Here, we demonstrate the use of the methods provided in PySpark for performing analysis of tabular datasets (metadata, aggregations, etc).

# COMMAND ----------

import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import *

# COMMAND ----------

# Packages
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import os

# COMMAND ----------

# Constants
DATA_PATH = r'file://'+os.path.abspath('../data')
LOAN_PATH = os.path.join(DATA_PATH, 'loan.csv')
CREDIT_PATH = os.path.join(DATA_PATH, 'credit card.csv')
TXN_PATH = os.path.join(DATA_PATH, 'txn.csv')

# Helper functions
def clean_columns(df):
    # Removes beginning/trailing whitespace from columns in a spark dataframe
    cols = df.columns
    for c in cols:
        df = df.withColumnRenamed(c, c.strip())
    return df

# COMMAND ----------

# MAGIC %md
# MAGIC ## LOAN DATASET #

# COMMAND ----------

# Read in loan dataset, note that some numeric columns are recorded as strings with commas - convert these to int
loan_df = spark.read.csv(LOAN_PATH, header=True, inferSchema=True)
loan_df = clean_columns(loan_df) \
    .withColumn('Loan Amount', F.trim(F.regexp_replace(F.col('Loan Amount'),',','')).astype('int')) \
    .withColumn('Debt Record', F.trim(F.regexp_replace(F.col('Debt Record'),',','')).astype('int'))

# COMMAND ----------

# Print the schema of the loan dataset
loan_df.printSchema()

# COMMAND ----------

# Print the first 5 lines of the loan dataset
loan_df.show(5)

# COMMAND ----------

# Print the number of columns in the loan dataset
len(loan_df.columns)

# COMMAND ----------

# Print the number of rows in the loan dataset
loan_df.count()

# COMMAND ----------

# Print the count of distinct records in the loan dataset
loan_df.distinct().count()

# COMMAND ----------

# Find the number of loans in each category
loan_df \
    .groupBy('Loan Category') \
    .agg(F.count(F.lit(1)).alias('Count')) \
    .orderBy(F.desc('Count')) \
    .show()

# COMMAND ----------

# Find the number of people who have taken more than 1 lakh
loan_df.filter(F.col('Loan Amount')>100000).count()

# COMMAND ----------

# Find the number of people with income greater than 60000 rupees
loan_df.filter(F.col('Income')>60000).count()

# COMMAND ----------

# Find the number of people with 2 or more returned cheques and income less than 50000
loan_df.filter((F.col('Returned Cheque')>=2) & (F.col('Income')<50000)).count()

# COMMAND ----------

# Find the number of people with 2 or more returned cheques and are single
loan_df.filter((F.col('Returned Cheque')>=2) & (F.col('Marital Status')=='SINGLE')).count()

# COMMAND ----------

# Find the number of people with expenditure over 50000 a month
loan_df.filter(F.col('Expenditure')>50000).count()

# COMMAND ----------

# MAGIC %md
# MAGIC # CREDIT CARD DATASET #

# COMMAND ----------

# Load the credit card dataset
credit_df = spark.read.csv(CREDIT_PATH, header=True, inferSchema=True)

# COMMAND ----------

# Print the schema of the credit card dataset
credit_df.printSchema()

# COMMAND ----------

# Print the number of columns in the credit card dataset
len(credit_df.columns)

# COMMAND ----------

# Print the number of rows in the credit card dataset
credit_df.count()

# COMMAND ----------

# Print the number of distinct records in the credit card dataset
credit_df.distinct().count()

# COMMAND ----------

# Print the first 5 rows in the credit card dataset
credit_df.show(5)

# COMMAND ----------

credit_df.filter(F.col('CreditScore')>700).count()

# COMMAND ----------

# Find the number of members who are elgible for credit card (assume this means credit score > 700)
credit_df.filter(F.col('CreditScore')>700).count()

# COMMAND ----------

# Find the number of members who are  elgible and active in the bank
credit_df.filter(F.col('CreditScore')*F.col('IsActiveMember')>700).count()

# COMMAND ----------

# Find the credit card users in Spain 
credit_df.filter(F.col('Geography')=='Spain').show()

# COMMAND ----------

# Find the credit card users with Estiamted Salary greater than 100000 and have exited the card
credit_df.filter(F.col('EstimatedSalary')*F.col('Exited')>100000).count()


# COMMAND ----------

# Find the credit card users with Estiamted Salary less than 100000 and have more than 1 products
credit_df.filter((F.col('EstimatedSalary')<100000) & (F.col('NumOfProducts')>1)).count()

# COMMAND ----------

# MAGIC %md
# MAGIC # TRANSACTION DATASET #

# COMMAND ----------

# Load the transacton dataset
txn_df = spark.read.csv(TXN_PATH, header=True, inferSchema=True)
txn_df = clean_columns(txn_df)

# COMMAND ----------

# Print the schema of the transacton dataset
txn_df.printSchema()

# COMMAND ----------

#COUNT OF TRANSACTION ON EVERY ACCOUNT
txn_df \
    .groupBy('Account No') \
    .agg(F.count(F.lit(1)).alias('count')).show()

# COMMAND ----------

# Find the Maximum withdrawal amount for each account
txn_df \
    .groupBy('Account No') \
    .agg(F.max('WITHDRAWAL AMT').alias('MaxWithdrawl')).show()


# COMMAND ----------

# MINIMUM WITHDRAWAL AMOUNT OF AN ACCOUNT
txn_df \
    .groupBy('Account No') \
    .agg(F.min('WITHDRAWAL AMT').alias('MinWithdrawl')).show()

# COMMAND ----------

#MAXIMUM DEPOSIT AMOUNT OF AN ACCOUNT
txn_df \
    .groupBy('Account No') \
    .agg(F.max('DEPOSIT AMT').alias('MaxDeposit')).show()

# COMMAND ----------

#MINIMUM DEPOSIT AMOUNT OF AN ACCOUNT
txn_df \
    .groupBy('Account No') \
    .agg(F.min('DEPOSIT AMT').alias('MinDeposit')).show()

# COMMAND ----------

#sum of balance in every bank account
txn_df \
    .groupBy('Account No') \
    .agg(F.sum('BALANCE AMT').alias('SumBalance')).show()

# COMMAND ----------

#Number of transaction on each date
txn_df \
    .groupBy('VALUE DATE') \
    .agg(F.count(F.lit(1)).alias('count')).show()


# COMMAND ----------

#List of customers with withdrawal amount more than 1 lakh
txn_df.filter(F.col('WITHDRAWAL AMT')>100000).show()


# COMMAND ----------


