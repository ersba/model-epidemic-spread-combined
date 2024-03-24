using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace EpidemicSpreadCombined
{
    public class SimpleCalibNn
    {
        private Sequential _model;
        
        private NDArray _features;
        
        private NDArray _labels;
        
        private string _modelPath;
        
        private string _projectDirectory;

        public SimpleCalibNn()
        {
            _projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            _modelPath = Path.Combine(_projectDirectory, "simple_calibnn");
            LoadData();
            InitModel();
            
        }
        
        //training using keras optimization
        public void AlternativeTrain(int epochs = 10)
        {
            _model.fit(_features, _labels, batch_size: 1, epochs: epochs, verbose: 1);
            _model.save(_modelPath, save_format:"tf");
        }

        public void Train(int epochs = 10)
        {
            var bestLoss = float.MaxValue;
            var bestBoundedPredictions = new [] {0f, 0f};
            
            if (File.Exists(Path.Combine(_projectDirectory, Params.OptimizedParametersPath)))
            {
                var lines = File.ReadAllLines(Path.Combine(_projectDirectory, Params.OptimizedParametersPath)).ToList();
                bestLoss = float.Parse(lines[1].Split(';')[2]);
            }
            var optimizer = new Adam();
            
            var bestEpochloss = float.MaxValue;
            
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                using (var tape = tf.GradientTape())
                {
                    var predictions = (Tensor)_model.predict(_features);
                    (var loss, var boundedPredictions) = CustomLoss(_labels, predictions);
                    
                    if (loss.numpy() < bestEpochloss)
                    {
                        bestEpochloss = loss.ToArray<float>()[0];
                        bestBoundedPredictions = boundedPredictions.ToArray<float>();
                    }

                    var gradients = tape.gradient(loss, _model.TrainableVariables);
                    optimizer.apply_gradients(zip(gradients, _model.TrainableVariables));
                    
                    Console.WriteLine($"epoch: {epoch + 1}, loss: {loss.numpy()}");
                    Console.Write("gradients: ");
                    tf.print(gradients[7]); //prints gradients of last(seventh) layer, useful to see whether the simulation is differentiable
                }
            }
            if (bestEpochloss < bestLoss)
            {
                using (StreamWriter writer = new StreamWriter(Path.Combine(_projectDirectory, Params.OptimizedParametersPath), false))
                {
                    writer.WriteLine("InitialInfectionRate;MortalityRate;Loss");
                    writer.WriteLine($"{bestBoundedPredictions[0]};{bestBoundedPredictions[1]};{bestEpochloss}");
                }
            }
            _model.save(_modelPath, save_format:"tf");
        }

        private (Tensor, NDArray) CustomLoss(Tensor target, Tensor predictions)
        {
            var lowerBounds = tf.constant(new [] {0.001f, 0.01f});
            var upperBounds = tf.constant(new [] {0.9f, 0.9f});
            var boundedPred = lowerBounds + (upperBounds - lowerBounds) * predictions;

            LearnableParams learnableParams = LearnableParams.Instance;
            
            Console.WriteLine("---------------------------------------------------");
            Console.Write("parameters:");
            tf.print(boundedPred);
            learnableParams.InitialInfectionRate = boundedPred[0, 0];
            learnableParams.MortalityRate = boundedPred[0, 1];
            var predictedDeaths = Program.EpidemicSpreadSimulation(true);
            
            Console.Write("deaths: ");
            tf.print(predictedDeaths);
            
            var loss = tf.reduce_mean(tf.square(target - predictedDeaths));
            
            return (loss, boundedPred.numpy());
        }

        //
        private Tensor LossThroughGumbel(Tensor target, Tensor prediction)
        {
            tf.print(prediction);
            var ones = tf.ones(new Shape(1000, 1));
            Tensor predColumn = ones * prediction;
            Tensor oneMinusPredColumn = ones * (1 - prediction);
            Tensor pTiled = tf.concat(new [] {predColumn, oneMinusPredColumn}, axis: 1);
            tf.print(tf.shape(pTiled));
            var infected = tf.reduce_sum(tf.cast(GumbelSoftmax.Execute(pTiled)[Slice.All, 0], dtype: TF_DataType.TF_FLOAT));
            tf.print(infected);
            return tf.reduce_mean(tf.square(target - infected));
        }
        private void LoadData()
        {
            var filePath = "Resources/training.csv";
            var lines = File.ReadAllLines(filePath).Skip(1).ToArray();
            var featureData = lines.Select(line => float.Parse(line.Split(',')[0])).ToArray();
            var labelData = lines.Select(line => float.Parse(line.Split(',')[1])).ToArray();

            _features = np.array(featureData).reshape(new Shape(-1, 1)); // Stellen Sie sicher, dass die Dimensionen stimmen
            _labels = np.array(labelData).reshape(new Shape(-1, 1)); // Stellen Sie sicher, dass die Dimensionen stimmen
            
        }
        
        private void InitModel()
        {
            if (Directory.Exists(_modelPath))
            {
                var model = keras.models.load_model(_modelPath);
                _model = (Sequential)model;
            }
            else
            {
                _model = keras.Sequential();
                _model.add(keras.layers.Dense(units: 16, activation: null, input_shape: new Shape(1)));
                _model.add(keras.layers.LeakyReLU());
                _model.add(keras.layers.Dense(32));
                _model.add(keras.layers.LeakyReLU());
                _model.add(keras.layers.Dense(32));
                _model.add(keras.layers.Dense(2, activation: "sigmoid"));
            }
            // _model.compile(optimizer: keras.optimizers.Adam(), loss: new CustomLoss());
        }
    }
    
    // Custom loss function used to integrate the abm in the optimization process of keras.
    // Only needed when training is done with AlternativeTrain
    class CustomLoss : ILossFunc
    {
        public Tensor Call(Tensor yTrue, Tensor yPred, Tensor sampleWeight = null)
        {
            var lowerBounds = tf.constant(new float[] {1.0f, 0.001f, 0.01f, 2.0f, 4.0f});
            var upperBounds = tf.constant(new float[] {9.0f, 0.9f, 0.9f, 6.0f, 7.0f});
            var boundedPred = lowerBounds + (upperBounds - lowerBounds) * yPred;
            
            LearnableParams learnableParams = LearnableParams.Instance;
            
            Console.Write("Params:");
            tf.print(boundedPred);
            learnableParams.MortalityRate = boundedPred[0, 2];
            var predictedDeaths = Program.EpidemicSpreadSimulation();
            Console.Write("Deaths: ");
            tf.print(predictedDeaths);
            var loss = tf.reduce_mean(tf.square(yTrue - predictedDeaths));
            
            return loss;
        }
        public string Reduction { get; }
        public string Name { get; }
    }
}