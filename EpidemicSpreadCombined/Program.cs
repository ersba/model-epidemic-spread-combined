using System;
using System.IO;
using System.Linq;
using EpidemicSpreadCombined.Model;
using Mars.Components.Starter;
using Mars.Interfaces.Model;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined
{
    internal static class Program
    
    {
        private static void Main()
        {
            // var calibNn = new SimpleCalibNn();
            // calibNn.Train(5);

            EpidemicSpreadSimulation();
        }
        public static Tensor EpidemicSpreadSimulation(bool train = false)
        {
            if (train == false && File.Exists(Params.OptimizedParametersPath))
            {
                var learnableParams = LearnableParams.Instance;
                var lines = File.ReadAllLines(Params.OptimizedParametersPath).ToList();
                var initialInfectionRate = tf.constant(float.Parse(lines[1].Split(';')[0]));
                var mortalityRate = tf.constant(float.Parse(lines[1].Split(';')[1]));

                learnableParams.InitialInfectionRate = initialInfectionRate;
                learnableParams.MortalityRate = mortalityRate;
            }
            var description = new ModelDescription();
            description.AddLayer<InfectionLayer>();
            description.AddAgent<Host, InfectionLayer>();
            
            var file = File.ReadAllText("config.json");
            var config = SimulationConfig.Deserialize(file);
            Params.Steps = (int)(config.Globals.Steps ?? 0);
            Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            
            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            var deaths = ((InfectionLayer)handle.Model.AllActiveLayers.First()).Deaths;
            starter.Dispose();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            return deaths;
        }
    }
}