# Databricks notebook source
import cloudpickle as pickle
from omegaconf import OmegaConf
import omegaconf
import numpy as np
import pandas as pd
from dataclasses import dataclass

import seaborn as sns
from matplotlib import pyplot as plt

from pyspark.sql.functions import countDistinct, count
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, current_timestamp, months_between, round, ceil, datediff, current_date, month, date_format, to_date
import databricks.koalas as ks
from pyspark.sql.functions import weekofyear, year, StringType
from pyspark.sql.types import FloatType, IntegerType

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, ceil, concat

from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import GBTClassifier

from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

# COMMAND ----------

# MAGIC %md # Read 2018 transactions

# COMMAND ----------

storageaccount = "dlstestrawanalyticseuw"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestrawanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storageaccount}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

container_name = 'default'
subfolder = '/root/general/Ahlmart/FITRANSXXXX/2018'

filepath = f"abfss://{container_name}@{storageaccount}.dfs.core.windows.net{subfolder}"

data_2018 = spark.read.format("parquet")\
                      .load(filepath)\
                      .filter( (col('COMPNO')=='10') \
                              & (col('TRANSREFNO').isNotNull()) \
                              & (col('OTYPE').isin([61, 66])) )

# COMMAND ----------

# MAGIC %md # Read 2019-2020 transactions

# COMMAND ----------

storageaccount = "dlstestrawanalyticseuw"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestrawanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storageaccount}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

container_name = 'default'
subfolder = '/root/general/Ahlmart/FITRANSXXXX/2019-2020'

filepath = f"abfss://{container_name}@{storageaccount}.dfs.core.windows.net{subfolder}"

data_2019_2020 = spark.read.format("parquet")\
                            .load(filepath)\
                            .filter( (col('COMPNO')=='10') \
                                    & (col('TRANSREFNO').isNotNull()) \
                                    & (col('OTYPE').isin([61, 66])) \
                                    & (col('POSTDATE') < F.lit('2020-10-01')) )

# COMMAND ----------

# MAGIC %md # Read 2021-forward

# COMMAND ----------

storageaccount = "dlstestbaseanalyticseuw"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestbaseanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storageaccount}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

container_name = 'enriched'
subfolder = '/root/Ahlmart/general/TransactionsAggr'

filepath = f"abfss://{container_name}@{storageaccount}.dfs.core.windows.net{subfolder}"

data_2021_2022 = spark.read.format("delta")\
                      .load(filepath)\
                      .filter( (col('COMPNO')=='10') \
                              & (col('TRANSREFNO').isNotNull()) \
                              & (col('OTYPE').isin([61, 66])) \
                              & (col('POSTDATE') >= F.lit('2020-10-01')) )

# COMMAND ----------

display(data_2021_2022)

# COMMAND ----------

# MAGIC %md # Concatenate all years

# COMMAND ----------

cols = ['CUSTNO', 'POSTDATE', 'TRANSREFNO', "SALES"]

df_1_new = data_2018.select(cols)
df_2_new = data_2019_2020.select(cols)
df_3_new = data_2021_2022.select(cols)

data_all_concatenated_middle = df_1_new.union(df_2_new)
data_all_concatenated_middle = data_all_concatenated_middle.union(df_3_new)

data_all_concatenated_middle = data_all_concatenated_middle.withColumn('year_month', date_format(col('POSTDATE'),'yyyy-MM'))

# COMMAND ----------

display(data_all_concatenated_middle)

# COMMAND ----------

display(data_all_concatenated_middle.groupby('year_month').agg(countDistinct('TRANSREFNO').alias('UNIQUE_TRANSREFNO')))

# COMMAND ----------

max_date = data_all_concatenated_middle.selectExpr('max(POSTDATE) as max_col1').first().max_col1
min_date = data_all_concatenated_middle.selectExpr('min(POSTDATE) as min_col1').first().min_col1
print(max_date, min_date)

# COMMAND ----------

# MAGIC %md # Read custbuy

# COMMAND ----------

storageaccount = "dlstestrawanalyticseuw"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestrawanalyticseuw-storage-key1"

container_name = 'default'
subfolder = '/root/general/Ahlmart/RPT_CUSTBUY_CUSTBILL/2022/24'

filepath = f"abfss://{container_name}@{storageaccount}.dfs.core.windows.net{subfolder}"

custbuy_custbill = spark.read.format("parquet").load(filepath)

# COMMAND ----------

display(custbuy_custbill.orderBy(col('CREDATE').desc()))

# COMMAND ----------

data_all_concatenated = data_all_concatenated_middle.join(custbuy_custbill.select("custno","compid"),on = "custno",how = "left")
data_all_concatenated.cache()

# COMMAND ----------

# MAGIC %md # Get Training/Test data - Function

# COMMAND ----------

def create_week_variable(x, last_stat_day = '2022-01-31', number_of_weeks = 7):
    grand_total_days = (number_of_weeks-1) * 7
    
    week_start = (datetime.strptime(last_stat_day, '%Y-%m-%d').date() - timedelta(days = 6)).strftime('%Y-%m-%d')
    week_end = last_stat_day
    
    column_date = x.strftime('%Y-%m-%d')
        
    j=0
    for i in range(0,grand_total_days+1,7):
        j=j+1
        week_start_1 = (datetime.strptime(week_start, '%Y-%m-%d').date() - timedelta(days = i)).strftime('%Y-%m-%d')
        week_end_1 = (datetime.strptime(week_end, '%Y-%m-%d').date() - timedelta(days = i)).strftime('%Y-%m-%d')
        
        if  week_start_1 <= column_date <= week_end_1:
            return j
        else:
            pass
                        
udf_create_week_variable = udf(create_week_variable, IntegerType())

# COMMAND ----------

def get_relative_month(manad=5):
    
    month_dictionary = {}
    
    j = 0
    for i in range(1,13):
        if i <= manad:
            month_dictionary[i] = abs(i - manad) + 1 - 1
        else:
            month_dictionary[i] = i + (abs(i - 12) - j) - 1
            j = j + 1
        
    return month_dictionary
            
test = get_relative_month(6)
print(test)

# COMMAND ----------

def get_training_test_data(df,
                  startdate_month_features,
                  startdate_lag_features,
                  enddate_all_features,
                  
                  startdate_active_customer,
                  enddate_active_customer,
                  
                  startdate_targetvariable,
                  enddate_target_variable,
               
                  customer_level = "custno",
                          
                  train_or_pred = 'train'):
    
    # cutoff_date_end = enddate_features
    month_variable = int(enddate_all_features[5:7])
    
    # print(month_variable)
    
    mapping = get_relative_month(month_variable)
    
    tr_list_view = ['customer_level']
    
    data_features = df.filter( (col('POSTDATE') >= F.lit(startdate_month_features)) & (col('POSTDATE') <= F.lit(enddate_all_features)) )
    data_features = data_features.withColumn('Date_variable', date_format(col('POSTDATE'),"MM-dd"))
    data_features = data_features.withColumn("customer_level",col(customer_level))
    
    data_features = data_features.withColumn('Month_variable_orig', month(col('POSTDATE')))
    data_features = data_features.withColumn('Month_variable', month(col('POSTDATE')))
    data_features = data_features.replace(to_replace=mapping, subset=['Month_variable'])
    #data_features = data_features.withColumn('Month_variable', ((month(col('POSTDATE')) - month_variable + 12))%lit(12))
        
    data_week = data_features.filter( (col('POSTDATE') >= F.lit(startdate_lag_features)) & (col('POSTDATE') <= F.lit(enddate_all_features)) )\
                             .withColumn('Week_variable', udf_create_week_variable(col('POSTDATE'), lit(enddate_all_features), lit(6)) )
    data_week = data_week.filter(col('Week_variable') >= lit(1))
    
    df_aggr = data_features.select("customer_level", "Month_variable", "TRANSREFNO")\
                           .groupby("customer_level", "Month_variable")\
                           .agg(countDistinct('TRANSREFNO').alias('UNIQUE_TRANSREFNO'))
    
    df_week = data_week.select("customer_level", "Week_variable", "TRANSREFNO")\
                           .groupby("customer_level", "Week_variable")\
                           .agg(countDistinct('TRANSREFNO').alias('UNIQUE_TRANSREFNO'))
    
    df_date = data_features.select("customer_level", "Date_variable", "TRANSREFNO","POSTDATE")\
                           .filter((col("POSTDATE") >= F.lit(startdate_lag_features)) & (col("POSTDATE") < F.lit(enddate_all_features))).withColumn("lag", datediff(F.lit(enddate_all_features),(col("POSTDATE"))))\
                           .groupby("customer_level","lag").agg(countDistinct("TRANSREFNO").alias("UNIQUE_TRANSREFNO"))
    
    summary_sales = data_features.filter((col('POSTDATE') >= F.lit(startdate_month_features)) & (col('POSTDATE') <= F.lit(enddate_all_features)))\
                                         .select('customer_level',"sales")\
                                         .groupby('customer_level')\
                                         .agg(F.sum(col("sales")).alias("total_sales"))

    df_aggr_tr = df_aggr.groupBy(tr_list_view).pivot("Month_variable").agg(F.first("UNIQUE_TRANSREFNO"))
    df_aggr_tr = df_aggr_tr.na.fill(value=0)
    df_aggr_tr = df_aggr_tr.select([F.col(c).alias("Month_"+c) if (c != "customer_level") else c for c in df_aggr_tr.columns])
    
    df_week_tr = df_week.groupBy(tr_list_view).pivot("Week_variable").agg(F.first("UNIQUE_TRANSREFNO"))
    df_week_tr = df_week_tr.na.fill(value=0)
    df_week_tr = df_week_tr.select([F.col(c).alias("Week_"+c) if (c != "customer_level") else c for c in df_week_tr.columns])

    #df_date_tr = df_date.groupBy(tr_list_view).pivot("lag").agg(F.first("UNIQUE_TRANSREFNO"))
    #df_date_tr = df_date_tr.na.fill(value=0)
    #df_date_tr = df_date_tr.select([F.col(c).alias("lag_"+c) if (c != "customer_level") else c for c in df_date_tr.columns])
    
    #for i in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15', 'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20']:
    #    if i not in df_date_tr.columns:
    #        df_date_tr = df_date_tr.withColumn(i,lit(0))
    
    df_full = df_aggr_tr.join(df_week_tr, on = "customer_level", how = "outer")
    df_full = df_full.join(summary_sales, on = "customer_level", how='left')
    df_full = df_full.na.fill(value = 0)
    
    customers = custbuy_custbill.filter( (col('CREDATE') <= F.lit(enddate_all_features)) & ((col('KILLDATE') > F.lit(enddate_all_features)) | col('KILLDATE').isNull()  )    )
    customers = customers.withColumn('killed', F.when( (F.col('KILLDATE') < F.lit(enddate_target_variable)), 1).otherwise(0)).withColumn("customer_level",col(customer_level))
    
    df_active = data_features.select("customer_level", "TRANSREFNO", 'POSTDATE')\
                           .filter((col("POSTDATE") >= F.lit(startdate_active_customer)) & (col("POSTDATE") <= F.lit(enddate_active_customer)))\
                           .groupby("customer_level").agg(countDistinct("TRANSREFNO").alias("UNIQUE_TRANSREFNO_ACTIVE"))
    
    if train_or_pred == 'train':
        df_target = df.withColumn("customer_level",col(customer_level)).select("customer_level", "TRANSREFNO", 'POSTDATE')\
                      .filter((col("POSTDATE") >= F.lit(startdate_targetvariable)) & (col("POSTDATE") <= F.lit(enddate_target_variable)))\
                      .groupby("customer_level").agg(countDistinct("TRANSREFNO").alias("UNIQUE_TRANSREFNO_TARGET"))
    
        df_full_new = customers.join(df_full, on='customer_level', how='inner')
        df_full_new = df_full_new.join(df_active, on='customer_level', how='left')
        df_full_new = df_full_new.join(df_target, on='customer_level', how='left')
        df_full_new = df_full_new.withColumn('target_churn', F.when((F.col('UNIQUE_TRANSREFNO_TARGET').isNull()) , 1).otherwise(0))
    else:
        df_full_new = customers.join(df_full, on='customer_level', how='inner')
        df_full_new = df_full_new.join(df_active, on='customer_level', how='left')
        
    df_full_new = df_full_new.withColumn('feature_days_customer', datediff(F.lit(enddate_all_features),(col("CREDATE")))+1)
    
    df_full_new = df_full_new.filter(col("UNIQUE_TRANSREFNO_ACTIVE") > 0)
    
    #df_full_new = df_full_new.select('customer_level', 'total_sales', 'REGION', 'CUSTCLASS', 'target_churn', \
    #                                                                             'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', \
    #                                                                             'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12', \
    #                                                                             'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6', \
    #                                                                             'feature_days_customer', 'total_sales')
    
    return df_full_new

# COMMAND ----------

# MAGIC %md # Prepare data for modelbuilding

# COMMAND ----------

def transformColumnsToNumeric(df_to_transform_numeric, df_to_fit_numeric, inputCol):
    
    #apply StringIndexer to inputCol
    inputCol_indexer = StringIndexer(inputCol = inputCol, outputCol = inputCol + "-index", stringOrderType="alphabetDesc").fit(df_to_fit_numeric)
    transformed = inputCol_indexer.setHandleInvalid("keep").transform(df_to_transform_numeric)
    
    onehotencoder_vector = OneHotEncoder(inputCol = inputCol + "-index", outputCol = inputCol + "-vector")
    transformed = onehotencoder_vector.fit(transformed).transform(transformed)
    
    return transformed

def prepareDataForModel(df_to_transform, df_to_fit, train_or_pred = 'train'):
    
    #if train_or_pred == 'train':
    #    labelIndexer = StringIndexer(inputCol="target_churn", outputCol="indexedLabel")
    #    new_pop = labelIndexer.setHandleInvalid("keep").fit(df_to_fit).transform(df_to_transform)
    #else:
    #    new_pop = df_to_transform

    new_pop = transformColumnsToNumeric(df_to_transform, df_to_fit, "CUSTCLASS")
    new_pop = transformColumnsToNumeric(new_pop, df_to_fit, "REGION")

    inputCols=[
            'CUSTCLASS-vector', 'REGION-vector',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 
            'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6',
            'feature_days_customer']
    
    df_va = VectorAssembler(inputCols = inputCols, outputCol="features")
    
    new_pop_t = df_va.transform(new_pop)

    if train_or_pred == 'train':
        df_transformed = new_pop_t.select(['features','target_churn', 'total_sales', 'customer_level'])
    else:
        df_transformed = new_pop_t.select(['features', 'total_sales', 'customer_level'])
    
    
    return df_transformed

# COMMAND ----------

# MAGIC %md # Backtesting

# COMMAND ----------

def backtest(model, model_training_data, periods = 5, customer_level = "custno", last_historic_day_of_sales = '2021-09-30'):
    
    result_dictionary = {}
    
    confusion_matrix_dfs = {}
    predicted_churns = {}
    
    startdate_month_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 364)).strftime('%Y-%m-%d')
    startdate_lag_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 50)).strftime('%Y-%m-%d')
    enddate_all_features = last_historic_day_of_sales

    startdate_active_customer = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 90)).strftime('%Y-%m-%d')
    enddate_active_customer = last_historic_day_of_sales

    startdate_targetvariable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = 1)).strftime('%Y-%m-%d')
    enddate_target_variable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = 365)).strftime('%Y-%m-%d')

    
    for i in tqdm(range(1,periods)):
        
        # Add period of days + 365 days to all dates
        startdate_month_features_offset = (datetime.strptime(startdate_month_features, '%Y-%m-%d').date()+timedelta(days = i + 365)).strftime('%Y-%m-%d')
        startdate_lag_features_offset = (datetime.strptime(startdate_lag_features, '%Y-%m-%d').date()+timedelta(days = i + 365)).strftime('%Y-%m-%d')
        enddate_all_features_offset = (datetime.strptime(enddate_all_features, '%Y-%m-%d').date()+timedelta(days = i + 365)).strftime('%Y-%m-%d')

        startdate_active_customer_offset = (datetime.strptime(startdate_active_customer, '%Y-%m-%d').date()+timedelta(days = i+ 365)).strftime('%Y-%m-%d')
        enddate_active_customer_offset = (datetime.strptime(enddate_active_customer, '%Y-%m-%d').date()+timedelta(days = i+ 365)).strftime('%Y-%m-%d')

        startdate_targetvariable_offset = (datetime.strptime(startdate_targetvariable, '%Y-%m-%d').date()+timedelta(days = i+ 365)).strftime('%Y-%m-%d')
        enddate_target_variable_offset = (datetime.strptime(enddate_target_variable, '%Y-%m-%d').date()+timedelta(days = i+ 365)).strftime('%Y-%m-%d')

        # Get data
        df_test = get_training_test_data(df = data_all_concatenated,

                            startdate_month_features = startdate_month_features_offset,
                            startdate_lag_features = startdate_lag_features_offset,
                            enddate_all_features = enddate_all_features_offset,

                            startdate_active_customer = startdate_active_customer_offset,
                            enddate_active_customer = enddate_active_customer_offset,

                            startdate_targetvariable = startdate_targetvariable_offset,
                            enddate_target_variable = enddate_target_variable_offset,
                            customer_level = customer_level)

        # Prepare testdata for model
        df_testdata_prepared = prepareDataForModel(df_test, model_training_data, 'train')

        # Apply model on testdata
        predictions = model.transform(df_testdata_prepared)

        # Convert predictions to pandas and select columns
        predictions_pandas = predictions.select(['indexedLabel', 'prediction', 'total_sales', "customer_level"]).toPandas()
        
        # Save confusion matrix
        conf_matrix = predictions.groupby('indexedLabel', 'prediction').count()
        conf_matrix = conf_matrix.withColumn('loop_day', F.lit(i))
        
        # Save result dict
        result_dictionary[i] = {}
        result_dictionary[i]["confusion_matrix"] = conf_matrix.toPandas()
        result_dictionary[i]["predicted_churns"] = predictions_pandas.loc[predictions_pandas.prediction == 1,"customer_level"].values
        result_dictionary[i]["predictions_pandas"] = predictions_pandas
        
        
    return result_dictionary
    

# COMMAND ----------

# MAGIC %md # Metrics

# COMMAND ----------

def get_metrics(result_dict):
    df_temp = pd.concat([result_dict[k]["confusion_matrix"] for k in range(1,len(result_dict) +1)] ,axis=0)
    test_2= df_temp.pivot(index=['loop_day'], columns=['indexedLabel', 'prediction'], values='count')
    test_2.columns = ["true_positives","false_positive","false_negative","true_negative"]
    test_2 = test_2.reset_index()
    
    recall = []
    precision = []
    accuracy = []
    loop_day = []
    newly_classified = []
    fell_out = []
    total_classified = []
    total_companies = []
    
    compare = set()
    
    for i in range(len(test_2)):
        recall.append(test_2.iloc[i,:].loc["true_positives"]/(test_2.iloc[i,:].loc["true_positives"] + test_2.iloc[i,:].loc["false_negative"]))
        precision.append(test_2.iloc[i,:].loc["true_positives"]/(test_2.iloc[i,:].loc["true_positives"] + test_2.iloc[i,:].loc["false_positive"]))
        loop_day.append(test_2.iloc[i,:].loc["loop_day"])
        accuracy.append((test_2.iloc[i,:].loc["true_positives"] + test_2.iloc[i,:].loc["true_negative"])/(test_2.iloc[i,1:].sum()))
        
        new = len(set(result_dict[i + 1]["predicted_churns"]).difference(compare))
        old = len(set(compare).difference(set(result_dict[i + 1]["predicted_churns"])))
        newly_classified.append(new)
        fell_out.append(old)
        
        compare = list(compare)
        for k in result_dict[i + 1]["predicted_churns"]:
            compare.append(k)
            
        compare = set(compare)
        
        total_classified.append(len(list(compare)))
        total_companies.append(test_2.iloc[i,1:].sum())
        
                        
    return pd.DataFrame({"loop_day":loop_day,"recall":recall,
                         "precision":precision,"accuracy":accuracy,
                         "newly_classified":newly_classified,"fallen_out":fell_out,
                         "total_classified":total_classified,"total_companies":total_companies})
    

# COMMAND ----------

class Model:
    def __init__(self,base_model):
        self.base_model = base_model
    
    def predict(self,X):
        return self.base_model.predict(X)
    
    def transform(X):
        return X


# COMMAND ----------

# MAGIC %md # Main

# COMMAND ----------

max_date = data_all_concatenated.selectExpr('max(POSTDATE) as max_col1').first().max_col1
max_date_for_model = (max_date - timedelta(days = 365) - timedelta(days = 30)).strftime('%Y-%m-%d')
print(max_date)

# COMMAND ----------

last_historic_day_of_sales = max_date_for_model
definition_churn_days = 365
startdate_lag_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = definition_churn_days)).strftime('%Y-%m-%d')
print(startdate_lag_features)

# COMMAND ----------

# MAGIC %md ## Evaluation

# COMMAND ----------

 def main_for_evaluation():
        
        # Fetch training data for compid and train model
        customer_level = "custno"
        periods = 5
        
        last_historic_day_of_sales = '2021-05-11'
        
        # Derived
        startdate_month_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 364)).strftime('%Y-%m-%d')
        startdate_lag_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 50)).strftime('%Y-%m-%d')
        enddate_all_features = last_historic_day_of_sales

        startdate_active_customer = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 90)).strftime('%Y-%m-%d')
        enddate_active_customer = last_historic_day_of_sales

        startdate_targetvariable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = 1)).strftime('%Y-%m-%d')
        enddate_target_variable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = 365)).strftime('%Y-%m-%d')

        df_training = get_training_test_data(df = data_all_concatenated,
                        
            startdate_month_features = startdate_month_features,
            startdate_lag_features = startdate_lag_features,
            enddate_all_features = enddate_all_features,

            startdate_active_customer = startdate_active_customer,
            enddate_active_customer = enddate_active_customer,

            startdate_targetvariable = startdate_targetvariable,
            enddate_target_variable = enddate_target_variable,
        
            customer_level = customer_level,
                                            
            train_or_pred = 'train')
        
        training_data = prepareDataForModel(df_training, df_training, train_or_pred = 'train')
        
        gbt = GBTClassifier(maxIter=5, maxDepth=8, labelCol="indexedLabel", featuresCol="features", seed=42)
        gbt.setMaxIter(30)

        model = gbt.fit(training_data)
        
        print("Model fit for customer level: {}".format(customer_level))
                
        result_dictionary_1 = backtest(model, df_training, periods, customer_level, last_historic_day_of_sales)
        
        print("Backtest computed for customer level: {}".format(customer_level))
        
        metrics_1 = get_metrics(result_dictionary_1)
        print(metrics_1)
        
        print("Metrics for {} are Done".format(customer_level))

        # Fetch training data for compid and train model
        
        return metrics_1,result_dictionary_1
    

# COMMAND ----------

metrics_1,result_dictionary_1 = main_for_evaluation()

# COMMAND ----------

metrics_1

# COMMAND ----------

plt.plot(metrics_1.loop_day,metrics_1.precision)

# COMMAND ----------

summary_sales_df = result_dictionary_1[1]["predictions_pandas"].loc[result_dictionary_1[1]["predictions_pandas"].prediction==1]
summary_sales_df = summary_sales_df.rename(columns={'customer_level':'custno'})

# COMMAND ----------

summary_sales_merged_2 = summary_sales_df.merge(custbuy_custbill.select("custno","custcat","czturnover").toPandas(), how = "left", on = ["custno","custno"])

# COMMAND ----------

summary_sales_merged_2

# COMMAND ----------

# summary_sales_merged_3 = summary_sales_merged_2.loc[summary_sales_merged_2.custcat == "860",:]

# COMMAND ----------

cutoffs = [0,1000,5000,10000,15000,20000,50000,150000,500000]
for cutoff in cutoffs:
    
    number_of_predicted = len(summary_sales_merged_2.loc[summary_sales_merged_2.total_sales > cutoff,:])
    correctly_predicted = summary_sales_merged_2.loc[summary_sales_merged_2.total_sales > cutoff,:].indexedLabel.sum()
    precision = correctly_predicted/number_of_predicted
    
    print("Cutoff = {}, Predicted = {}, Correct = {}, Precision = {}, ".format(cutoff,number_of_predicted,correctly_predicted,precision))

# COMMAND ----------

# MAGIC %md ## Current prediction

# COMMAND ----------

max_date = data_all_concatenated.selectExpr('max(POSTDATE) as max_col1').first().max_col1
max_date = (max_date - timedelta(days = 365)).strftime('%Y-%m-%d')
print(max_date)

# COMMAND ----------

last_historic_day_of_sales = max_date
definition_churn_days = 365
startdate_lag_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = definition_churn_days)).strftime('%Y-%m-%d')
print(startdate_lag_features)

# COMMAND ----------

 def main_for_prediction():
        
        # Fetch training data for compid and train model
        customer_level = "custno"
        last_historic_day_of_sales = max_date
        
        definition_churn_days = 365
        
        # Derived training dates
        startdate_month_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 364)).strftime('%Y-%m-%d')
        startdate_lag_features = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = 50)).strftime('%Y-%m-%d')
        enddate_all_features = last_historic_day_of_sales

        startdate_active_customer = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() - timedelta(days = definition_churn_days)).strftime('%Y-%m-%d')
        enddate_active_customer = last_historic_day_of_sales

        startdate_targetvariable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = 1)).strftime('%Y-%m-%d')
        enddate_target_variable = (datetime.strptime(last_historic_day_of_sales, '%Y-%m-%d').date() + timedelta(days = definition_churn_days)).strftime('%Y-%m-%d')

        # Print values
        print(f"startdate_month_features {startdate_month_features} \n startdate_lag_features {startdate_lag_features} \n enddate_all_features {enddate_all_features}\
        \n startdate_active_customer {startdate_active_customer} \n enddate_active_customer {enddate_active_customer} \n startdate_targetvariable {startdate_targetvariable}\
        \n enddate_target_variable {enddate_target_variable} \n")
        
        # Training data
        df_training = get_training_test_data(df = data_all_concatenated,
                        
            startdate_month_features = startdate_month_features,
            startdate_lag_features = startdate_lag_features,
            enddate_all_features = enddate_all_features,

            startdate_active_customer = startdate_active_customer,
            enddate_active_customer = enddate_active_customer,

            startdate_targetvariable = startdate_targetvariable,
            enddate_target_variable = enddate_target_variable,
        
            customer_level = customer_level,
                                           
            train_or_pred = 'train')
        
        training_data = prepareDataForModel(df_training, df_training, train_or_pred = 'train')
        
        
        # Train model
        gbt = GBTClassifier(maxIter=30, maxDepth=20, labelCol="target_churn", featuresCol="features", seed=42)
        #gbt.setMaxIter()

        model = gbt.fit(training_data)
        
        print("Model fit for training data")
        
        # Derived prediction dates
        startdate_month_features_pred = (datetime.strptime(startdate_month_features, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')
        startdate_lag_features_pred = (datetime.strptime(startdate_lag_features, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')
        enddate_all_features_pred = (datetime.strptime(enddate_all_features, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')

        startdate_active_customer_pred = (datetime.strptime(startdate_active_customer, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')
        enddate_active_customer_pred = (datetime.strptime(enddate_active_customer, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')

        startdate_targetvariable_pred = (datetime.strptime(startdate_targetvariable, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')
        enddate_target_variable_pred = (datetime.strptime(enddate_target_variable, '%Y-%m-%d').date() + timedelta(days = definition_churn_days+1)).strftime('%Y-%m-%d')
        
        # Print values
        print(f"startdate_month_features_pred {startdate_month_features_pred} \n startdate_lag_features_pred {startdate_lag_features_pred} \n enddate_all_features_pred {enddate_all_features_pred}\
        \n startdate_active_customer_pred {startdate_active_customer_pred} \n enddate_active_customer_pred {enddate_active_customer_pred} \n startdate_targetvariable_pred {startdate_targetvariable_pred}\
        \n enddate_target_variable_pred {enddate_target_variable_pred} \n")
        
        # Prediction data
        df_prediction = get_training_test_data(df = data_all_concatenated,
                        
            startdate_month_features = startdate_month_features_pred,
            startdate_lag_features = startdate_lag_features_pred,
            enddate_all_features = enddate_all_features_pred,

            startdate_active_customer = startdate_active_customer_pred,
            enddate_active_customer = enddate_active_customer_pred,

            startdate_targetvariable = startdate_targetvariable_pred,
            enddate_target_variable = enddate_target_variable_pred,
        
            customer_level = customer_level,
                                           
            train_or_pred = 'pred')
        
        prediction_data = prepareDataForModel(df_prediction, df_training, train_or_pred = 'pred')
        
        # Apply model on prediction data
        predictions = model.transform(prediction_data)
        print("Model applied to prediction data")
        
        # Convert predictions to pandas and select columns
        # predictions_pandas = predictions.select(['prediction', 'total_sales', "customer_level"]).toPandas()
        
        return predictions, df_training, df_prediction, model, prediction_data, training_data


# COMMAND ----------

predictions_current, training_dataset_current, prediction_dataset_current, model_test, pred_temp, train_temp  = main_for_prediction()

# COMMAND ----------

print(model_test.featureImportances)

# COMMAND ----------

display(predictions_current)

# COMMAND ----------

predictions_current.groupby('prediction').count().show()

# COMMAND ----------

display(training_dataset_current)

# COMMAND ----------

training_dataset_current.groupby('target_churn').count().show()

# COMMAND ----------

display(training_dataset_current)

# COMMAND ----------

display(training_dataset_current.groupby('region').count())

# COMMAND ----------

display(prediction_dataset_current.groupby('region').count())

# COMMAND ----------

display(prediction_dataset_current)

# COMMAND ----------

temp = training_dataset_current.select('customer_level','Month_0',	'Month_1',	'Month_2',	'Month_3',	'Month_4',	'Month_5',	'Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Week_1','Week_2','Week_3','Week_4','Week_5','Week_6', 'feature_days_customer', 'UNIQUE_TRANSREFNO_ACTIVE', 'total_sales', 'target_churn')
temp.cache()

# COMMAND ----------

display(training_dataset_current.filter(col('customer_level')=='6397507'))


# COMMAND ----------

display(temp)

# COMMAND ----------

test = prediction_dataset_current.select('customer_level','Month_0',	'Month_1',	'Month_2',	'Month_3',	'Month_4',	'Month_5',	'Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Week_1','Week_2','Week_3','Week_4','Week_5','Week_6', 'feature_days_customer', 'UNIQUE_TRANSREFNO_ACTIVE', 'total_sales')
test.cache()

# COMMAND ----------

test = test.join(predictions_current.select('customer_level', 'prediction'), on=['customer_level'], how='left')

# COMMAND ----------

display(test)

# COMMAND ----------

# MAGIC %md # Filter Current predictions for pilot

# COMMAND ----------

predictions_with_custbuyinfo = predictions_current.join(custbuy_custbill.withColumnRenamed('custno', 'customer_level'), ['customer_level'], 'left')
predictions_with_custbuyinfo.cache()

# COMMAND ----------

predictions_vs = predictions_with_custbuyinfo.select('customer_level', 'total_sales', 'prediction', 'NAME1', 'COMMENT1', 'TEL', 'REGION', 'DISTRICT', 'CUSTCLASS', 'CUSTCAT', 'CREDATE', 'SALESMAN', \
                                                                          'COMPID', 'EMAILLONG', 'CITY', 'SALESMAN_EMPLOYNAME', 'SALESMAN_EMAIL', 'CZSNI', 'CZTURNOVER', 'CZRATING')\
                                                   .filter((col('total_sales')>2000) \
                                                           & (col('prediction')==1) \
                                                           & (col('REGION')==87) \
                                                           & (col('DISTRICT')==202) \
                                                           & (col('CUSTCAT').isin([800, 801, 802])))

predictions_vs.count()

# COMMAND ----------

predictions_el = predictions_with_custbuyinfo.select('customer_level', 'total_sales', 'prediction', 'NAME1', 'COMMENT1', 'TEL', 'REGION', 'DISTRICT', 'CUSTCLASS', 'CUSTCAT', 'CREDATE', 'SALESMAN', \
                                                                          'COMPID', 'EMAILLONG', 'CITY', 'SALESMAN_EMPLOYNAME', 'SALESMAN_EMAIL', 'CZSNI', 'CZTURNOVER', 'CZRATING')\
                                                   .filter((col('total_sales')>3000) \
                                                           & (col('prediction')==1) \
                                                           & (col('REGION')==87) \
                                                           & (col('DISTRICT')==202) \
                                                           & (col('CUSTCAT').isin([840, 841, 842, 843])))

predictions_el.count()

# COMMAND ----------

predictions_bygg = predictions_with_custbuyinfo.select('customer_level', 'total_sales', 'prediction', 'NAME1', 'COMMENT1', 'TEL', 'REGION', 'DISTRICT', 'CUSTCLASS', 'CUSTCAT', 'CREDATE', 'SALESMAN', \
                                                                          'COMPID', 'EMAILLONG', 'CITY', 'SALESMAN_EMPLOYNAME', 'SALESMAN_EMAIL', 'CZSNI', 'CZTURNOVER', 'CZRATING')\
                                                   .filter((col('total_sales')>15000) \
                                                           & (col('prediction')==1) \
                                                           & (col('REGION')==83) \
                                                           & (col('DISTRICT').isin([116, 130])) \
                                                           & (col('CUSTCAT').isin([860, 861, 862, 863, 864, 868])))

predictions_bygg.count()

# COMMAND ----------

# MAGIC %md ## Save as csv in blob storage

# COMMAND ----------

output_container_name = "projects"
storage_account_name = 'dlstestwrkspanalyticseuw'
output_subfolder = "/salesanalytics/churn/vs"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestwrkspanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

output_filepath = f"abfss://{output_container_name}@{storage_account_name}.dfs.core.windows.net{output_subfolder}"

predictions_vs.coalesce(1).write.format("csv")\
                    .mode('overwrite')\
                    .option('header', True)\
                    .option('sep', ';')\
                    .option("encoding", "ISO-8859-1")\
                    .save(output_filepath)

# COMMAND ----------

output_container_name = "projects"
storage_account_name = 'dlstestwrkspanalyticseuw'
output_subfolder = "/salesanalytics/churn/bygg"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestwrkspanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

output_filepath = f"abfss://{output_container_name}@{storage_account_name}.dfs.core.windows.net{output_subfolder}"

predictions_bygg.coalesce(1).write.format("csv")\
                    .mode('overwrite')\
                    .option('header', True)\
                    .option('sep', ';')\
                    .option("encoding", "ISO-8859-1")\
                    .save(output_filepath)

# COMMAND ----------

output_container_name = "projects"
storage_account_name = 'dlstestwrkspanalyticseuw'
output_subfolder = "/salesanalytics/churn/el"
secretscopename = "dbw_secretscope_kv_appliedai"
access_key = "dlstestwrkspanalyticseuw-storage-key1"

spark.conf.set(f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
               dbutils.secrets.get(scope = f"{secretscopename}", key = f"{access_key}"))

output_filepath = f"abfss://{output_container_name}@{storage_account_name}.dfs.core.windows.net{output_subfolder}"

predictions_el.coalesce(1).write.format("csv")\
                    .mode('overwrite')\
                    .option('header', True)\
                    .option('sep', ';')\
                    .option("encoding", "ISO-8859-1")\
                    .save(output_filepath)

# COMMAND ----------

# MAGIC %md # Analyze

# COMMAND ----------

display(training_dataset_current)

# COMMAND ----------

training_dataset_current.cache()
prediction_dataset_current.cache()

# COMMAND ----------

def transformColumnsToNumeric(df_to_transform_numeric, df_to_fit_numeric, inputCol):
    
    #apply StringIndexer to inputCol
    inputCol_indexer = StringIndexer(inputCol = inputCol, outputCol = inputCol + "-index", stringOrderType="alphabetDesc").fit(df_to_fit_numeric)
    transformed = inputCol_indexer.setHandleInvalid("keep").transform(df_to_transform_numeric)
    
    onehotencoder_vector = OneHotEncoder(inputCol = inputCol + "-index", outputCol = inputCol + "-vector")
    transformed = onehotencoder_vector.fit(transformed).transform(transformed)
    
    return transformed

def prepareDataForModel(df_to_transform, df_to_fit, train_or_pred = 'train'):
    
    #if train_or_pred == 'train':
    #    labelIndexer = StringIndexer(inputCol="target_churn", outputCol="indexedLabel")
    #    new_pop = labelIndexer.setHandleInvalid("keep").fit(df_to_fit).transform(df_to_transform)
    #else:
    #    new_pop = df_to_transform

    new_pop = transformColumnsToNumeric(df_to_transform, df_to_fit, "CUSTCLASS")
    new_pop = transformColumnsToNumeric(new_pop, df_to_fit, "REGION")

    inputCols=[
            'CUSTCLASS-vector', 'REGION-vector',
            'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
            'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 
            'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6',
            'feature_days_customer']
    
    df_va = VectorAssembler(inputCols = inputCols, outputCol="features")
    
    new_pop_t = df_va.transform(new_pop)

    #if train_or_pred == 'train':
    #    df_transformed = new_pop_t.select(['features','target_churn', 'total_sales', 'customer_level'])
    #else:
    #    df_transformed = new_pop_t.select(['features', 'total_sales', 'customer_level'])
    
    
    return new_pop_t

# COMMAND ----------

test = prepareDataForModel(training_dataset_current, training_dataset_current, 'train')
#test = test.select('customer_level','Month_0',	'Month_1',	'Month_2',	'Month_3',	'Month_4',	'Month_5', 'CUSTCLASS', 'REGION', 'CUSTCLASS-vector', 'CUSTCLASS-index'	#,'Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Week_1','Week_2','Week_3','Week_4','Week_5','Week_6', 'feature_days_customer', 'UNIQUE_TRANSREFNO_ACTIVE', 'total_sales', 'target_churn')

display(test)

# COMMAND ----------

test_pred = prepareDataForModel(prediction_dataset_current, training_dataset_current, 'pred')
#test_pred = test_pred.select('customer_level','Month_0',	'Month_1',	'Month_2',	'Month_3',	'Month_4',	'Month_5', 'CUSTCLASS', 'REGION', 'CUSTCLASS-vector', 'CUSTCLASS-index'	#,'Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Week_1','Week_2','Week_3','Week_4','Week_5','Week_6', 'feature_days_customer', 'UNIQUE_TRANSREFNO_ACTIVE', 'total_sales')

display(test_pred)

# COMMAND ----------

display(test_pred.groupby('CUSTCLASS','CUSTCLASS-index').count())

# COMMAND ----------

display(test.groupby('CUSTCLASS','CUSTCLASS-index').count())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

display(predictions_with_custbuyinfo.filter(col('customer_level')=='6670154'))

# COMMAND ----------

display(training_dataset_current.filter(col('customer_level')=='6670154'))

# COMMAND ----------

display(prediction_dataset_current.filter(col('customer_level')=='6147473'))

# COMMAND ----------

temp = predictions_with_custbuyinfo.filter(col('prediction')==1).orderBy(col('total_sales').desc())
temp.count()
display(temp)

# COMMAND ----------

# MAGIC %md # Things to fix

# COMMAND ----------

'''
The monthly lag variable is now calculated overlapping with previous years sales. Which means it isn't a lag variable for latest month but also sales figures for last year (part of month). To have a lag-month variable this needs to be fixed. Use same principle for month-lag as week-lag. 


'''
