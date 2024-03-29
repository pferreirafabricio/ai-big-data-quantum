﻿using System;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;

var sw = new Stopwatch();
sw.Start();

MLContext mlContext = new MLContext();

HouseData[] houseData =
{
    new HouseData() { Size = 1.1F, Price = 1.2F, IsExpensive = false },
    new HouseData() { Size = 1.3F, Price = 1.5F, IsExpensive = false },
    new HouseData() { Size = 1.5F, Price = 1.7F, IsExpensive = false },
    new HouseData() { Size = 1.7F, Price = 1.8F, IsExpensive = false },
    new HouseData() { Size = 1.9F, Price = 2.0F, IsExpensive = false },
    new HouseData() { Size = 2.1F, Price = 2.2F, IsExpensive = false },
    new HouseData() { Size = 2.3F, Price = 2.4F, IsExpensive = true },
    new HouseData() { Size = 2.5F, Price = 2.7F, IsExpensive = true },
    new HouseData() { Size = 2.7F, Price = 2.8F, IsExpensive = true },
    new HouseData() { Size = 2.9F, Price = 2.9F, IsExpensive = true },
    new HouseData() { Size = 3.1F, Price = 3.2F, IsExpensive = true }
};

IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

var pipeline = mlContext.Transforms
    .Concatenate("Features", new[] { "Size", "Price" })
    .Append(
        mlContext
            .BinaryClassification
            .Trainers
            .SdcaLogisticRegression(
                labelColumnName: "IsExpensive",
                maximumNumberOfIterations: 100_000
            )
    );

var model = pipeline.Fit(trainingData);

var validationData = new HouseData() { Size = 2.3F, Price = 2.5F };
var result = mlContext.Model
    .CreatePredictionEngine<HouseData, Prediction>(model)
    .Predict(validationData);

sw.Stop();

var size = (validationData.Size * 1000) / 10.76;
var price = (validationData.Price * 100_000) * 5;

Console.WriteLine($"Predicted price for size: {size:F2}m2 and price: {price:C2}. Is expensive? {result.IsExpensive}");
Console.WriteLine("Elapsed={0}", sw.Elapsed);

public class HouseData
{
    public bool IsExpensive { get; set; }
    public float Size { get; set; }
    public float Price { get; set; }
}

public class Prediction
{
    /// <summary>
    /// ColumnName attribute is used to change the column name from
    /// its default value, which is the name of the field.
    /// https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/getting-started/BinaryClassification_SentimentAnalysis/SentimentAnalysis/SentimentAnalysisConsoleApp/DataStructures/SentimentPrediction.cs
    /// </summary>
    [ColumnName("PredictedLabel")]
    public bool IsExpensive { get; set; }
}