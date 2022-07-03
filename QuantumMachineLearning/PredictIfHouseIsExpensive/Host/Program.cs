using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using PredictIfHouseIsExpensive;
using PredictIfHouseIsExpensive.Quantum;
using System.Linq;

HouseData[] houseData =
{
    new HouseData() { Size = 1.1D, Price = 1.2D, IsExpensive = false },
    new HouseData() { Size = 1.3D, Price = 1.5D, IsExpensive = false },
    new HouseData() { Size = 1.5D, Price = 1.7D, IsExpensive = false },
    new HouseData() { Size = 1.7D, Price = 1.8D, IsExpensive = false },
    new HouseData() { Size = 1.9D, Price = 2.0D, IsExpensive = false },
    new HouseData() { Size = 2.1D, Price = 2.2D, IsExpensive = false },
    new HouseData() { Size = 2.3D, Price = 2.4D, IsExpensive = true },
    new HouseData() { Size = 2.5D, Price = 2.7D, IsExpensive = true },
    new HouseData() { Size = 2.7D, Price = 2.8D, IsExpensive = true },
    new HouseData() { Size = 2.9D, Price = 2.9D, IsExpensive = true },
    new HouseData() { Size = 3.1D, Price = 3.2D, IsExpensive = true }
};

var validationData = new List<double[]>();
validationData.Add(new double[] { 1.1D, 1.2D });
// validationData.Add(new double[] { 2.2D, 2.4D });

var trainingData = new List<double[]>();
var trainingLabels = new List<long>();

var initialParameters = new[]
{
    new [] { 1.0111D, 2.2543534D, 2.45656546D, 3.111111D },
    new [] { 1.4483434D, 2.2645654675D, 2.4D, 3.8356349692D },
    new [] { 1.0453534D, 2.23333D, 2.443345D, 3.555557567 },
};

for (var index = 0; index < houseData.Length; index++)
{
    trainingData.Add(new double[] { houseData[index].Size, houseData[index].Price });
    trainingLabels.Add(houseData[index].IsExpensive ? 1L : 0L);
}

var quantumTrainingData = new QArray<QArray<double>>(
    trainingData.Select(vector => new QArray<double>(vector))
);

var quantumTrainingLabel = new QArray<long>(trainingLabels.Select(x => x));

var quantumInitialParameters = new QArray<QArray<double>>(
    initialParameters.Select(vector => new QArray<double>(vector))
);

var quantumValidationData = new QArray<QArray<double>>(
    validationData.Select(vector => new QArray<double>(vector))
);

using var targetMachine = new QuantumSimulator();

var (optimizedParameters, optimizedBias) = await TrainLinearlySeparableModel
    .Run(
        targetMachine,
        trainingVectors: quantumTrainingData,
        trainingLabels: quantumTrainingLabel,
        initialParameters: quantumValidationData
    );

var results = await ValidateClassifyLinearlySeparableModel
    .Run(
        targetMachine,
        samples: quantumValidationData,
        parameters: optimizedParameters,
        bias: optimizedBias,
        tolerance: 0.0005,
        numberOfMeasurements: 100_000
    );

foreach (var result in results)
{
    Console.WriteLine($"Result: {result:F10}.");
}

public class HouseData
{
    public bool IsExpensive { get; set; }
    public double Size { get; set; }
    public double Price { get; set; }
}

public class Prediction
{
    /// <summary>
    /// ColumnName attribute is used to change the column name from
    /// its default value, which is the name of the field.
    /// https://github.com/dotnet/machinelearning-samples/blob/main/samples/csharp/getting-started/BinaryClassification_SentimentAnalysis/SentimentAnalysis/SentimentAnalysisConsoleApp/DataStructures/SentimentPrediction.cs
    /// </summary>
    public bool IsExpensive { get; set; }
}
