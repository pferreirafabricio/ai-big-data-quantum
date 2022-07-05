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
    new HouseData() { Size = 3.1D, Price = 3.2D, IsExpensive = true },
};

var validationData = new List<double[]>();
validationData.Add(new double[] { 2.2D, 2.4D });

var trainingData = new List<double[]>();
var trainingLabels = new List<long>();

var initialParameters = new[]
{
    new [] { 0.74855, 0.86259, 0.50246, 1D },
    new [] { 0.20398, 0D, 0.88110, 0.49063, },
    new [] { 1D, 0.58477, 0.47506, 0.63679 },
    new [] { 0.81057, 0.90390, 0.60358, 0D },
    new [] { 0.46373, 0.94726, 1D, 0.93518 },
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
var tolerance = 0.000001D;
var learningRate = 0.00001D;
var numberOfMeasurements = 100_000;

var (optimizedParameters, optimizedBias) = await TrainLinearlySeparableModel
    .Run(
        targetMachine,
        trainingVectors: quantumTrainingData,
        trainingLabels: quantumTrainingLabel,
        initialParameters: quantumInitialParameters,
        learningRate: learningRate,
        tolerance: tolerance,
        numberOfMeasurements: numberOfMeasurements
    );

var results = await ValidateClassifyLinearlySeparableModel
    .Run(
        targetMachine,
        samples: quantumValidationData,
        parameters: optimizedParameters,
        bias: optimizedBias,
        tolerance: tolerance,
        numberOfMeasurements: numberOfMeasurements
    );

foreach (var result in results)
{
    Console.WriteLine($"Result: {result:F5}.");
}

public class HouseData
{
    public bool IsExpensive { get; set; }
    public double Size { get; set; }
    public double Price { get; set; }
}
