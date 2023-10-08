from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("DataPipeline").getOrCreate()

# Load the data from GCS
train_data_path = "gs://bdp_cw2_bucket/train.csv"
test_data_path = "gs://bdp_cw2_bucket/test.csv"
df_train = spark.read.format("csv").option("header", "true").load(train_data_path)
df_test = spark.read.format("csv").option("header", "true").load(test_data_path)

# Convert columns to double data type
df_train = df_train.withColumn("battery_power", col("battery_power").cast("double"))
df_train = df_train.withColumn("clock_speed", col("clock_speed").cast("double"))
df_train = df_train.withColumn("ram", col("ram").cast("double"))
df_train = df_train.withColumn("mobile_wt", col("mobile_wt").cast("double"))

# Convert columns to double data type for df_test
df_test = df_test.withColumn("battery_power", col("battery_power").cast("double"))
df_test = df_test.withColumn("clock_speed", col("clock_speed").cast("double"))
df_test = df_test.withColumn("ram", col("ram").cast("double"))
df_test = df_test.withColumn("mobile_wt", col("mobile_wt").cast("double"))

# Standard scaling of numeric features
# Define the columns to be scaled
columns_to_scale = ['battery_power', 'clock_speed', 'ram', 'mobile_wt']

# Assemble the features into a vector column
assembler = VectorAssembler(inputCols=columns_to_scale, outputCol='features')
df_train_assembled = assembler.transform(df_train)
df_test_assembled = assembler.transform(df_test)

# Create a StandardScaler object
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

# Fit and transform the scaler
scaler_model = scaler.fit(df_train_assembled)
df_train_scaled = scaler_model.transform(df_train_assembled)
df_test_scaled = scaler_model.transform(df_test_assembled)

# Encoding categorical features
# Categorical columns to be encoded
categorical_columns = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

# Apply StringIndexer for categorical columns
indexers = [StringIndexer(inputCol=column, outputCol=column + '_index') for column in categorical_columns]
indexer_models = [indexer.fit(df_train_scaled) for indexer in indexers]
df_train_indexed = df_train_scaled
df_test_indexed = df_test_scaled

# Transform data using the trained indexers
for model in indexer_models:
    df_train_indexed = model.transform(df_train_indexed)
    df_test_indexed = model.transform(df_test_indexed)

# Apply OneHotEncoder for indexed categorical columns
encoders = [OneHotEncoder(inputCol=column + '_index', outputCol=column + '_encoded') for column in categorical_columns]
encoder_models = [encoder.fit(df_train_indexed) for encoder in encoders]
df_train_encoded = df_train_indexed
df_test_encoded = df_test_indexed

# Transform data using the trained encoders
for model in encoder_models:
    df_train_encoded = model.transform(df_train_encoded)
    df_test_encoded = model.transform(df_test_encoded)

# Store the preprocessed data in Parquet format
preprocessed_train_path = "gs://bdp_cw2_bucket/data/preprocessed_train.parquet"
preprocessed_test_path = "gs://bdp_cw2_bucket/data/preprocessed_test.parquet"
df_train_encoded.write.mode("overwrite").parquet(preprocessed_train_path)
df_test_encoded.write.mode("overwrite").parquet(preprocessed_test_path)

from pyspark.sql import functions as F
# Example query: Calculate average RAM by price range
average_ram_by_price = df_train_encoded.groupBy('price_range').agg(F.avg('ram').alias('avg_ram'))




# Query: Calculate the average screen height and width by price range
average_screen_size_by_price = df_train_encoded.groupBy('price_range').agg( F.avg('sc_h').alias('avg_screen_height'), F.avg('sc_w').alias('avg_screen_width'))



import matplotlib.pyplot as plt
import seaborn as sns
# Select the numerical columns for correlation analysis
numerical_columns = ['battery_power', 'ram', 'px_height', 'px_width']
# Calculate the correlation matrix
correlation_matrix = df_train_encoded.select(numerical_columns).toPandas().corr()
# Visualization: Heatmap of the correlation matrix

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap')

plt.show()


# Visualization: Bar plot of price range distribution
price_range_distribution = df_train_encoded.groupBy('price_range').count().orderBy('price_range')
price_range_distribution = price_range_distribution.toPandas()
plt.figure(figsize=(8, 6))
sns.barplot(x='price_range', y='count', data=price_range_distribution, palette='coolwarm')
plt.title('Price Range Distribution')
plt.xlabel('Price Range')
plt.ylabel('Count')
plt.show()
