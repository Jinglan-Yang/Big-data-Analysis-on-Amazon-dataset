import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    result1 = (product_data.join(review_data, product_data["asin"]==review_data["asin"], "left_outer").groupBy(product_data["asin"]).agg(F.avg(F.col("overall")).alias("meanRating")))
    #result1.show()
    
    result2 = (product_data.join(review_data, product_data["asin"]==review_data["asin"], "left_outer").groupBy(product_data["asin"]).agg(F.when(F.count(F.col("overall")) > 0, F.count(F.col("overall"))).alias("countRating")))
    #result2.show()
    
    count_total = int(result1.count())
    #print("Total Rows (including null rows):", count_total)
    
    mean_meanRating = float(result1.agg(F.avg(F.col("meanRating"))).collect()[0][0])
    #print("mean value of meanRating: ", mean_meanRating)
    
    variance_meanRating = float(result1.agg(F.var_samp(F.col("meanRating"))).collect()[0][0])
    #print("Variance of meanRating:", variance_meanRating)
    
    nullNulls_meanRating = int(result1.filter(F.col("meanRating").isNull()).count())
    #print("Count of nulls in meanRating:", nullNulls_meanRating)
    
    mean_countRating = float(result2.agg(F.avg(F.col("countRating"))).collect()[0][0])
    #print("mean value of countRating: ", mean_countRating)   

    variance_countRating = float(result2.agg(F.var_samp(F.col("countRating"))).collect()[0][0])
    #print("Variance of countRating:", variance_countRating)
    
    nullNulls_countRating = int(result2.filter(F.col("countRating").isNull()).count())
    #print("Count of nulls in countRating:", nullNulls_countRating) 




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total']=count_total
    res['mean_meanRating']=mean_meanRating
    res['variance_meanRating']=variance_meanRating
    res['numNulls_meanRating']=nullNulls_meanRating
    res['mean_countRating']=mean_countRating
    res['variance_countRating']=variance_countRating
    res['numNulls_countRating']=nullNulls_countRating



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    category_withoutBlank=product_data.withColumn("categories", F.when(F.col("categories").getItem(0).getItem(0) == "", None).otherwise(F.col("categories")))

    category_column = (category_withoutBlank.withColumn("category", F.when(F.col("categories").isNotNull() & F.col("categories").getItem(0).getItem(0).isNotNull() & F.col("categories").getItem(0).isNotNull(), F.col("categories").getItem(0).getItem(0))).select("category")) 
    #category_column.show()
    
    category_column_check=category_column.distinct()
    #category_column_check.show(100,truncate=False)
    
    test=category_column.filter(F.col("category").isNotNull()).select("category").distinct()
    #test.show(100,truncate=False)
    
    bestSalesCategory_column=(product_data
    .withColumn("bestSalesCategory", F.when(F.map_keys(F.col("salesRank")).isNotNull()&F.map_keys(F.col("salesRank")).isNotNull()&F.map_keys(F.col("salesRank"))[0].isNotNull(), F.map_keys(F.col("salesRank"))[0]))
    .select('bestSalesCategory'))
    #bestSalesCategory_column.show()
    

    bestSalesRank_column=(product_data
    .withColumn('bestSalesRank', F.when(F.map_keys(F.col("salesRank")).isNotNull()&F.map_values(F.col("salesRank"))[0].isNotNull(), F.map_values(F.col("salesRank"))[0]))
    .select('bestSalesRank'))
    #bestSalesRank_column.show()
    
    count_total=int(category_column.count())
    
    mean_bestSalesRank=float(bestSalesRank_column.agg(F.avg(F.col("bestSalesRank"))).collect()[0][0])
    
    variance_bestSalesRank=float(bestSalesRank_column.agg(F.var_samp(F.col("bestSalesRank"))).collect()[0][0])
    
    numNulls_category = int(category_column.filter(F.col("category").isNull()).count())
    
    countDistinct_category = int(category_column.filter(F.col("category").isNotNull()).select("category").distinct().count())
    
    numNulls_bestSalesCategory=int(bestSalesCategory_column.filter(F.col("bestSalesCategory").isNull()).count())
    
    countDistinct_bestSalesCategory=int(bestSalesCategory_column.filter(F.col("bestSalesCategory").isNotNull()).select("bestSalesCategory").distinct().count())




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    res['count_total']=count_total
    res['mean_bestSalesRank']=mean_bestSalesRank
    res['variance_bestSalesRank']=variance_bestSalesRank
    res['numNulls_category']=numNulls_category
    res['countDistinct_category']=countDistinct_category
    res['numNulls_bestSalesCategory']=numNulls_bestSalesCategory
    res['countDistinct_bestSalesCategory']=countDistinct_bestSalesCategory



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    exploded_df = product_data.select("asin", F.explode("related.also_viewed").alias("also_viewed"))
    #exploded_df.show()
    
    price_df=product_data.select("asin","price")
    #price_df.show()
    
    meanPriceAlsoViewed_column=(exploded_df.join(price_df, exploded_df["also_viewed"]==price_df["asin"],'left_outer').groupBy(exploded_df["asin"]).agg(F.avg(F.col("price")).alias("meanPriceAlsoViewed_column")))
    #meanPriceAlsoViewed_column.show()
    
    countAlsoViewed_column=(exploded_df.groupBy(exploded_df["asin"]).agg(F.count(F.col("also_viewed")).alias('countAlsoViewed')))
    #countAlsoViewed_column.show()
    
    
    
    product_data_output = product_data.join(meanPriceAlsoViewed_column, [asin_column], how='left')
    product_data_output = product_data_output.join(countAlsoViewed_column, [asin_column], how='left')
    numNulls_meanPriceAlsoViewed=int(product_data_output.filter(F.col("meanPriceAlsoViewed_column").isNull()).count())
    #print(numNulls_meanPriceAlsoViewed)
    numNulls_countAlsoViewed=int(product_data_output.filter(F.col('countAlsoViewed').isNull()).count())
    #print(numNulls_countAlsoViewed)
    
    count_total=int(product_data_output.count())
    #print(count_total)

    
    mean_meanPriceAlsoViewed=float(meanPriceAlsoViewed_column.agg(F.avg(F.col("meanPriceAlsoViewed_column"))).collect()[0][0])
    #print(mean_meanPriceAlsoViewed)
    
    variance_meanPriceAlsoViewed=float(meanPriceAlsoViewed_column.agg(F.var_samp(F.col("meanPriceAlsoViewed_column"))).collect()[0][0])
    #print(variance_meanPriceAlsoViewed)
    
    mean_countAlsoViewed=float(countAlsoViewed_column.agg(F.avg(F.col('countAlsoViewed'))).collect()[0][0])
    #print(mean_countAlsoViewed)
    
    variance_countAlsoViewed=float(countAlsoViewed_column.agg(F.var_samp(F.col('countAlsoViewed'))).collect()[0][0])
    #print(variance_countAlsoViewed)




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    res['count_total']=count_total
    res['mean_meanPriceAlsoViewed']=mean_meanPriceAlsoViewed
    res['variance_meanPriceAlsoViewed']=variance_meanPriceAlsoViewed
    res['numNulls_meanPriceAlsoViewed']=numNulls_meanPriceAlsoViewed
    res['mean_countAlsoViewed']=mean_countAlsoViewed
    res['variance_countAlsoViewed']=variance_countAlsoViewed
    res['numNulls_countAlsoViewed']=numNulls_countAlsoViewed



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    price_column = product_data.select(F.col("price").cast("float"))
    #price_column.show()
    
    mean_price = price_column.select(F.mean("price")).collect()[0][0]
    
    meanImputedPrice_column = price_column.withColumn("meanImputedPrice", F.when(F.col("price").isNull(), mean_price).otherwise(F.col("price")))
    #meanImputedPrice_column.show()
    
    median_price = price_column.approxQuantile("price", [0.5],0.05)[0]
    
    medianImputedPrice_column=price_column.withColumn("medianImputedPrice", F.when(F.col("price").isNull(), median_price).otherwise(F.col("price")))
    #medianImputedPrice_column.show()
    
    title_column=product_data.select('title')
    #title_column.show()
    
    unknownImputedTitle_column = title_column.withColumn("unknownImputedTitle", F.when((F.col("title").isNull()) | (F.col("title") == ""), "unknown").otherwise(F.col("title")))
    #unknownImputedTitle_column.show()

    
    count_total=int(meanImputedPrice_column.count())
    #print(count_total)
    
    mean_meanImputedPrice=float(meanImputedPrice_column.agg(F.avg(F.col("meanImputedPrice"))).collect()[0][0])
    #print(mean_meanImputedPrice)

    variance_meanImputedPrice=float(meanImputedPrice_column.agg(F.var_samp(F.col("meanImputedPrice"))).collect()[0][0])
    #print(variance_meanImputedPrice)
    
    numNulls_meanImputedPrice=int(meanImputedPrice_column.filter(F.col("meanImputedPrice").isNull()).count())
    #print(numNulls_meanImputedPrice)
    
    mean_medianImputedPrice=float(medianImputedPrice_column.agg(F.avg(F.col("medianImputedPrice"))).collect()[0][0])
    #print(mean_medianImputedPrice)
    
    variance_medianImputedPrice=float(medianImputedPrice_column.agg(F.var_samp(F.col("medianImputedPrice"))).collect()[0][0])
    #print(variance_medianImputedPrice)
    
    numNulls_medianImputedPrice=int(medianImputedPrice_column.filter(F.col("medianImputedPrice").isNull()).count())
    #print(numNulls_medianImputedPrice)
    
    numUnknowns_unknownImputedTitle=unknownImputedTitle_column.filter(F.col("unknownImputedTitle") == "unknown").count()
    #print(numUnknowns_unknownImputedTitle)




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total']=count_total
    res['mean_meanImputedPrice']=mean_meanImputedPrice
    res['variance_meanImputedPrice']=variance_meanImputedPrice
    res['numNulls_meanImputedPrice']=numNulls_meanImputedPrice
    res['mean_medianImputedPrice']=mean_medianImputedPrice
    res['variance_medianImputedPrice']=variance_medianImputedPrice
    res['numNulls_medianImputedPrice']=numNulls_medianImputedPrice
    res['numUnknowns_unknownImputedTitle']=numUnknowns_unknownImputedTitle



    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    title_column=product_processed_data.select('title')
    
    titleArray_column=title_column.withColumn("titleArray", F.split(F.lower(F.col("title")), " "))
    #titleArray_column.show()
    
    titleArray_column=titleArray_column.select("titleArray")
    #titleArray_column.show()
    
    product_processed_data_output=titleArray_column
    #product_processed_data_output.show()
    
    word2Vec = M.feature.Word2Vec(
    vectorSize=16,
    minCount=100,
    seed=102,
    numPartitions=4,
    inputCol="titleArray",
    outputCol="word2vec")
    
    model = word2Vec.fit(titleArray_column)

    #model.save("word2vec_model")
    
    #model = M.feature.Word2VecModel.load("word2vec_model")
        
    count_total=product_processed_data_output.count()
    #print(count_total)
    
    size_vocabulary = model.getVectors().count()
    #print(size_vocabulary)




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    indexer = M.feature.StringIndexer(inputCol="category", outputCol="categoryIndex")
    
    encoder = M.feature.OneHotEncoder(inputCol="categoryIndex", outputCol="categoryOneHot", dropLast=False)
    
    pca = M.feature.PCA(k=15, inputCol="categoryOneHot", outputCol="categoryPCA")
    
    pipeline = M.Pipeline(stages=[indexer, encoder, pca])
    
    category_column=product_processed_data.select('category')
    model = pipeline.fit(category_column)
    df_transformed = model.transform(category_column)
    #df_transformed.show()
    
    count_total=int(df_transformed.count())
    #print(count_total)
    
    #summarizer = M.stat.Summarizer.metrics("mean")
    
    meanVector_categoryOneHot=df_transformed.select(M.stat.Summarizer.mean(df_transformed.categoryOneHot))
    #meanVector_categoryOneHot.show()
    #print(type(meanVector_categoryOneHot))
    meanVector_categoryOneHot=meanVector_categoryOneHot.collect()
    ##meanVector_categoryOneHot=meanVector_categoryOneHot.toArray().tolist()
    #print(meanVector_categoryOneHot)
    #print(type(meanVector_categoryOneHot))
    
    list_meanVector_categoryOneHot=[]
    for i in meanVector_categoryOneHot:
        #print(i)
        for j in i:
            #print(j)
            for z in j:
                #print(z)
                list_meanVector_categoryOneHot.append(float(z))
    #print(list_meanVector_categoryOneHot)
    #meanVector_categoryOneHot=[0.2513, 0.1523, 0.0562, 0.0525, 0.0521, 0.0463, 0.0368, 0.0356, 0.0351, 0.0287, 0.0286, 0.0285, 0.0278, 0.0275, 0.0207, 0.0182, 0.0142, 0.0124, 0.0117, 0.0115, 0.0076, 0.0073, 0.0065, 0.0052, 0.005, 0.0032, 0.0026, 0.0019, 0.0012, 0.0011, 0.0008, 0.0008, 0.0007, 0.0007, 0.0006, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    #common_list = [row.mean(categoryOneHot).toArray().tolist() for row in meanVector_categoryOneHot]
    #print(common_list)
    
    meanVector_categoryPCA=df_transformed.select(M.stat.Summarizer.mean(df_transformed.categoryPCA))
    #meanVector_categoryPCA.show()
    meanVector_categoryPCA=meanVector_categoryPCA.collect()
    #meanVector_categoryPCA=meanVector_categoryPCA.tolist()
    #print(meanVector_categoryPCA)
    #print(type(meanVector_categoryPCA))
    list_meanVector_categoryPCA=[]
    for i in meanVector_categoryPCA:
        #print(i)
        for j in i:
            #print(j)
            for z in j:
                #print(z)
                list_meanVector_categoryPCA.append(float(z))  
    #print(list_meanVector_categoryPCA)
    #meanVector_categoryPCA=[-0.15, -0.1752, 0.0165, -0.0027, 0.0334, -0.0537, 0.0099, -0.0046, 0.04, -0.0003, -0.0019, 0.0061, -0.0037, 0.0372, -0.0305]
    




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total']=count_total
    res['meanVector_categoryOneHot']=list_meanVector_categoryOneHot
    res['meanVector_categoryPCA']=list_meanVector_categoryPCA





    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------
    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    features = train_data.columns[:-1] 
    target = "overall"
    
    dt_model = M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=5)
    dt_model = dt_model.fit(train_data)
    
    predictions = dt_model.transform(test_data)

    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    rmse = float(evaluator.evaluate(predictions))
    #print(rmse)    
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse']=rmse

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    new_train_data, val_data = train_data.randomSplit([0.75, 0.25], seed=42)
    
    features = train_data.columns[:-1]
    target = "overall"
    
    dt_model =  M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=5)
    dt_model = dt_model.fit(new_train_data)
    predictions = dt_model.transform(val_data)
    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    valid_rmse_depth_5 = float(evaluator.evaluate(predictions))
    #print(valid_rmse_depth_5)
    
    dt_model =  M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=7)
    dt_model = dt_model.fit(new_train_data)
    predictions = dt_model.transform(val_data)
    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    valid_rmse_depth_7 = float(evaluator.evaluate(predictions))
    #print(valid_rmse_depth_7)   
    
    dt_model =  M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=9)
    dt_model = dt_model.fit(new_train_data)
    predictions = dt_model.transform(val_data)
    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    valid_rmse_depth_9 = float(evaluator.evaluate(predictions))
    #print(valid_rmse_depth_9)
    
    dt_model =  M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=12)
    dt_model = dt_model.fit(new_train_data)
    predictions = dt_model.transform(val_data)
    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    valid_rmse_depth_12 = float(evaluator.evaluate(predictions))
    #print(valid_rmse_depth_12)
    
    
    dt_model =  M.regression.DecisionTreeRegressor(featuresCol="features", labelCol=target, maxDepth=12)
    dt_model = dt_model.fit(new_train_data)
    predictions = dt_model.transform(test_data)
    evaluator = M.evaluation.RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    test_rmse = float(evaluator.evaluate(predictions))
    #print(test_rmse)   
    
    
    
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    res['test_rmse']=test_rmse
    res['valid_rmse_depth_5']=valid_rmse_depth_5
    res['valid_rmse_depth_7']=valid_rmse_depth_7
    res['valid_rmse_depth_9']=valid_rmse_depth_9
    res['valid_rmse_depth_12']=valid_rmse_depth_12

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

