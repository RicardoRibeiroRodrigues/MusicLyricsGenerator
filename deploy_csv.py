import pyspark
from pyspark.pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col, array_contains, monotonically_increasing_id , avg, length, lit
from pyspark.sql.functions import udf
import numpy as np
from random import randint
import re



spark = SparkSession.builder \
    .appName('MusicGen') \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext

test = spark.read.format("parquet").load("test.parquet")
test.printSchema()

n_rows = test.count()
print(f"N rows: ", n_rows)


def get_snippet(text, n_tokens: int):
    """
    Returns a snippet of n_tokens length and the last N tokens of the text
    """
    tokenized = re.findall(r"\b\w+\b", text.lower())
    if len(tokenized) <= n_tokens:
        return text
    start = randint(0, len(tokenized) - n_tokens - 1)
    return " ".join(tokenized[start:start + n_tokens])

udf_snippet = udf(get_snippet, StringType())

test = test.withColumn("snippet", udf_snippet("lyrics", lit(10)))

# Write the snippets to a csv file
test.select("snippet").write.csv("test_snippets.csv")
