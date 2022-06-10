using System;
using Microsoft.ML;
using Microsoft.ML.Data;

MLContext mlContext = new MLContext();

// 1. Import or create training data
HouseData[] houseData =
{
    new HouseData() { Size = 1.1F, Price = 1.2F },
    new HouseData() { Size = 1.9F, Price = 2.3F },
    new HouseData() { Size = 2.8F, Price = 3.0F },
    new HouseData() { Size = 3.4F, Price = 3.7F },
    new HouseData() { Size = 3.9F, Price = 4.0F }
};

IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

// 2. Specify data preparation and model training pipeline
var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

// 3. Train model
var model = pipeline.Fit(trainingData);

// 4. Make a prediction
var size = new HouseData() { Size = 2.5F };
var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft= {price.Price * 100:C}k");

// Predicted price for size: 2500 sq ft= $261.98k

public class HouseData
{
    public float Size { get; set; }
    public float Price { get; set; }
}

public class Prediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}