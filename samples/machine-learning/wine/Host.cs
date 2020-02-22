
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;
using static System.Math;

namespace Microsoft.Quantum.Samples
{
    using Microsoft.Quantum.MachineLearning;

    static class Program
    {
        static async Task Main(string[] args)
        {
            // Next, we initialize a full state-vector simulator as our target machine.
            using var targetMachine = new QuantumSimulator();

            // To help understand performance, it can be helpful to attach
            // timestamps to each log message.
            // For details on how this is done, see the EnableTimestampsInLog
            // extension method below.
            targetMachine.EnableTimestampsInLog();

            // Once we initialized our target machine,
            // we can then use that target machine to train a QCC classifier.
            var (optimizedParameters, optimizedBias) = await TrainWineModel.Run(
                targetMachine
            );

            // After training, we can use the validation data to test the accuracy
            // of our new classifier.
            var testMisses = await ValidateWineModel.Run(
                targetMachine,
                optimizedParameters,
                optimizedBias
            );
            System.Console.WriteLine($"Observed miss rate of {100 * testMisses:F2}%.");
        }

        static void EnableTimestampsInLog(this QuantumSimulator sim)
        {
            var stopwatch = new Stopwatch();
            var last = TimeSpan.FromMilliseconds(0);
            stopwatch.Start();
            sim.DisableLogToConsole();
            sim.OnLog += (message) =>
            {
                var now = stopwatch.Elapsed;
                var delta = now - last;
                last = now;
                System.Console.WriteLine($"[{now} +{delta}] {message}");
            };

        }
    }
}
