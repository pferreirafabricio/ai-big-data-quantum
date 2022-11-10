# ⚛ Predict if a house price could be considered expensive

<p align="left">
This is a Quantum Machine Learning algorithm that uses quantum classification to identify whether a house can be considered expensive for its properties like size and price.
  <br><br>
  <!-- License -->
  <a>
    <img alt="license url" src="https://img.shields.io/badge/License-GPL--3.0-green?style=for-the-badge&labelColor=1C1E26&color=FDDE4A">
  </a>
</p>

## 🗂 Folder structure

- Host
```
- Program.cs 
Core classical logic, written in C# with .NET 6, for interacting with quantum backend functions and operations.
```

- QuantumBackend
```
- Library.qs 
Core quantum logic, written in Q# with QDK, provides functions for training and validating models.
```


## ⚗️ Results Summary

```ini
Training complete, found optimal parameters: [0.74855,0.86259,0.50246,1], -0.33899029094962474 with 7 misses
Model: SequentialModel(([ControlledRotation(((0, []), PauliY, 0))], [0.74855,0.86259,0.50246,1], -0.33899029094962474))
Result: 1.00000.
Elapsed=00:00:00.9521979
```

## 🧪 Tests
- [Basic performance comparison between classical and quantum machine learning](https://github.com/pferreirafabricio/ai-big-data-quantum/releases/tag/v0.0.1)

## 🏃🏽‍♂️ Quick Start
 1. Clone this repository `git clone https://github.com/pferreirafabricio/ai-big-data-quantum.git`
 2. Enter in the project's folder: `cd ai-big-data-quantum/ClassicMachineLearning/PredictIfHouseIsExpensive`
 3. Enter in the host's folder: `cd Host`
 4. Finally run: `dotnet run` 😃

## 🧱 This project was built with: 
- [.NET](https://dotnet.microsoft.com/en-us/)
- [Q#](https://docs.microsoft.com/en-us/azure/quantum/overview-what-is-qsharp-and-qdk)
- [QDK](https://docs.microsoft.com/en-us/azure/quantum/overview-what-is-qsharp-and-qdk)
- [C#](https://docs.microsoft.com/en-us/dotnet/csharp/)
