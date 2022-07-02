using System;
using Microsoft.ML;
using Microsoft.ML.Data;

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
                maximumNumberOfIterations: 100
            )
    );

var model = pipeline.Fit(trainingData);

var validationData = new HouseData() { Size = 2.2F, Price = 2.4F };
var result = mlContext.Model
    .CreatePredictionEngine<HouseData, Prediction>(model)
    .Predict(validationData);

Console.WriteLine($"Predicted price for size: {validationData.Size * 1000} and price: {validationData.Price}. Is expensive? {result.IsExpensive}");

public class HouseData
{
    public bool IsExpensive { get; set; }
    public float Size { get; set; }
    public float Price { get; set; }
}

public class Prediction
{
    [ColumnName("PredictedLabel")]
    public bool IsExpensive { get; set; }
}