# Model epidemic spread combined

The `model-epidemic-spread-combined` model is a simulation of the spread of an epidemic using agent-based modeling 
and a neural network. It's written in C# and uses the .NET 6.0 framework. The model is inspired by the implementation
[Differentiable Agent-based Epidemiology](https://github.com/AdityaLab/GradABM) of the paper 
"Differentiable Agent-based Epidemiology." It combines the capability for granular multi-agent modeling of the [MARS](https://www.mars-group.org/docs/tutorial/intro)
framework with the superior gradient based optimization of neural networks using the [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET)
framework, hence the name 'model-epidemic-spread-combined'.

## Process flow
The process flow of the model consists of the following phases:

1. **Calibration**: The simple calibration neural network outputs the parameters for the epidemic spread simulation.

## Execution

To run the model-epidemic-spread-combined model, follow these steps:

1. Ensure you have .NET 6.0 SDK installed on your machine. You can download it from the [official Microsoft website](https://dotnet.microsoft.com/download/dotnet/6.0).
2. Clone the repository to your local machine.
3. Open the project in your preferred IDE and install the required NuGet packages.
4. Run the `Main` method in the `Program.cs` file.

Please note that the `Main` method in `Program.cs` is set to run the `EpidemicSpreadSimulation` by default. If you want 
to train the model using the `SimpleCalibNn` class, you'll need to uncomment 
```csharp
// var calibNn = new SimpleCalibNn();
// calibNn.Train(10);
```
and comment out 
```csharp
EpidemicSpreadSimulation();
```


## Configuration

The scenario provides attributes that can be modified to influence the simulation results. They are in the
`config.json` file. The following attributes can be adjusted:

- `steps` defines the amount of ticks in the simulation
- `console` determines whether a progress bar is displayed on the console during simulation.
- `count` defines the amount of agents in the simulation

```json
{
	"globals":{
		"deltaT": 1,
		"steps": 50,
		"pythonVisualization": false,
		"console": true,
		"output": "csv"
	},
	"layers":[
		{
			"name":"InfectionLayer"
		}
	],
	"agents":[
		{
			"name": "Host",
			"count": 1000,
			"file": "Resources/hosts_data.csv"
		}
	]
}
```
By default the neural network is trained for 10 epochs. To change the amount of epochs change the `Epochs` variable in the
`Params.cs` file.
```csharp
public static class Params
    {
        ...
        public static readonly int Epochs = 10;
        ...
     }
```
The neural network is trained so that the parameters it outputs result in 300 deaths at the end of the simulation. 
To change this, the value in the Resources/training.csv file needs to be adjusted to the desired value.

To change the behaviour of the epidemic spread you can change the variables in the `Params.cs` and `LearnableParams.cs` file.
While training the model, the neural network will adjust the variables of `LearnableParams.cs`. Noteworthy variables to
adjust are:

```csharp
public static class Params
    {
        ...
        public static readonly float R0Value = 5.18f;
        
        public static Tensor ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_INT32);
            
        public static Tensor InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_INT32);
        ...
     }
```

```csharp   
public class LearnableParams
    {
        ...
        private LearnableParams()
        {
            InitialInfectionRate = tf.constant(0.05, dtype: TF_DataType.TF_FLOAT);
            MortalityRate = tf.constant(0.1, dtype: TF_DataType.TF_FLOAT);
        }
        ...
    }
```

- `R0Value` defines the basic reproduction number of the epidemic
- `ExposedToInfectedTime` defines the time an agent is exposed to the virus before becoming infected
- `InfectedToRecoveredTime` defines the time an agent is infected before recovering or dying
- `InitialInfectionRate` defines the initial infection rate of the epidemic
- `MortalityRate` defines the mortality rate of the epidemic
## Visualization

To visualize the result of the simulation run the visualization.R file with RStudio. The script will read the Host CSV file
containing the stages of the agents and visualize the result in a graph.

An example of the visualization is shown below:

![example_visualization.png](https://raw.githubusercontent.com/ersba/images-model-epidemic-spread-combined/main/example_visualization.png)
