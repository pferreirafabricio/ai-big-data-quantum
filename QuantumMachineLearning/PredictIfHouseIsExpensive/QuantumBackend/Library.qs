// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//////////////////////////////////////////////////////////////////////
// references:
// https://github.com/microsoft/QuantumKatas/tree/main/tutorials/QuantumClassification
// https://docs.microsoft.com/en-us/azure/quantum/user-guide/libraries/machine-learning/basic-classification?tabs=tabid-csharp
//////////////////////////////////////////////////////////////////////

namespace PredictIfHouseIsExpensive.Quantum {
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.MachineLearning;
    open Microsoft.Quantum.Math;

    function DefaultSchedule(samples : Double[][]) : SamplingSchedule {
        return SamplingSchedule([
            0..Length(samples) - 1
        ]);
    }

    function ClassifierStructure() : ControlledRotation[] {
        return [
            ControlledRotation((0, []), PauliY, 0)
        ];
    }

    operation TrainLinearlySeparableModel(
        trainingVectors : Double[][],
        trainingLabels : Int[],
        initialParameters : Double[][],
        learningRate : Double,
        tolerance : Double,
        numberOfMeasurements : Int
    ) : (Double[], Double) {
        Message("Beggining training.");
        Message($"Learning Rate: {learningRate} | Tolerance {tolerance} | Number of measurements {numberOfMeasurements}");

        let samples = Mapped(
            LabeledSample,
            Zipped(trainingVectors, trainingLabels)
        );

        Message($"Samples: {samples}");

        Message("Ready to train.");
        
        let (optimizedModel, nMisses) = TrainSequentialClassifier(
            Mapped(
                SequentialModel(ClassifierStructure(), _, 0.0),
                initialParameters
            ),
            samples,
            DefaultTrainingOptions()
                w/ LearningRate <- learningRate
                w/ Tolerance <- tolerance
                w/ NMeasurements <- numberOfMeasurements
                w/ VerboseMessage <- Message,
            DefaultSchedule(trainingVectors),
            DefaultSchedule(trainingVectors)
        );

        Message($"Training complete, found optimal parameters: {optimizedModel::Parameters}, {optimizedModel::Bias} with {nMisses} misses");
        return (optimizedModel::Parameters, optimizedModel::Bias);
    }

    operation ValidateClassifyLinearlySeparableModel(
        samples : Double[][],
        parameters : Double[],
        bias : Double,
        tolerance : Double,
        numberOfMeasurements : Int
    )
    : Int[] {
        let model = Default<SequentialModel>()
            w/ Structure <- ClassifierStructure()
            w/ Parameters <- parameters
            w/ Bias <- bias;

        Message($"Model: {model}");

        let probabilities = EstimateClassificationProbabilities(
            tolerance,
            model,
            samples,
            numberOfMeasurements
        );

        return InferredLabels(model::Bias, probabilities);
    }
}
