# Model epidemic spread combined

The `model-epidemic-spread-combined` model is a simulation of the spread of an epidemic using agent-based modeling 
and a neural network. It's written in C# and uses the .NET 6.0 framework. The model is inspired by the implementation
[Differentiable Agent-based Epidemiology](https://github.com/AdityaLab/GradABM) of the paper 
"Differentiable Agent-based Epidemiology." It combines the capability for granular multi-agent modeling of the [MARS](https://www.mars-group.org/docs/tutorial/intro)
framework with the optimization of neural networks using the [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET)
framework, hence the name 'model-epidemic-spread-combined'.

## Terms

- **Agent**: An agent is an individual entity in the simulation. In this model, the agents are hosts that can be in different stages of the epidemic.
- **Environment**: An environment is like its name suggests the environment in which the agents act. It enables them to interact with each other.
- **Layer**: A layer is the setting in which the agents live. In this model, the agents are in the InfectionLayer.
- **Tick**: A tick is a unit of time in the simulation. In each tick, the agents interact with each other and update their stages.


- **Stage**: The stage of an agent represents the state of the agent in the epidemic. The stages are susceptible, exposed, infected, recovered and mortality.
 It is based on the SEIRM progression model from [Nowcasting and Forecasting the Potential Domestic and
  International Spread of the 2019-nCoV Outbreak Originating in Wuhan, China: A Modelling Study](https://linkinghub.elsevier.com/retrieve/pii/S0140673620302609).


- **Tensor**: A tensor is a multi-dimensional array. Tensors are implemented using the [TensorFlow.Net Tensor](https://tensorflownet.readthedocs.io/en/latest/Tensor.html) implementation. 
In this model, tensors are used to store and process data including the stages of the agents. They are important because of their features:
  - They enable automatic differentiation, meaning they can automatically compute the gradient of a function with respect to its input
  - They allow for parallelization of operations across multiple processing units, such as CPUs and GPUs, leading to faster processing of large-scale data
  - They support a wide range of mathematical operations, enabling them to solve complex mathematical functions

To get an understanding of how tensors are used in [TensorFlow.NET](https://github.com/SciSharp/TensorFlow.NET) here is an example 
of an elementwise calculation, first with arrays and then with tensors: 

```csharp
float[,] arrayA = { { 1, 2 }, { 3, 4 } };
float[,] arrayB = { { 5, 6 }, { 7, 8 } };

float[,] arraySum = new float[2, 2];
for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
            {
                arraySum[i, j] = arrayA[i, j] + arrayB[i, j];
            }
    }


var tensorA = new Tensor(arrayA);
var tensorB = new Tensor(arrayB);

var tensorSum = tf.add(tensorA, tensorB);
```
arraySum and tensorSum contain the same values and are two-dimensional with a shape of 2 x 2:

![example_matrices.png](https://raw.githubusercontent.com/ersba/images-model-epidemic-spread-combined/main/example_matrices.png)

This example shows how tensors can efficiently perform mathematical operations without the need for loops. The stage of every
agent is stored in a n x 1 Tensor within the `InfectionLayer`object, where n is the amount of agents in the simulation. During
the Tick the agents read the index of their stage from an array copy of the tensor (the agents themselves don't use tensors) 
and act according to their stage simulating the transmission of an epidemic. During the PostTick the tensor and its copy are updated by the layer simulating the progression
of an epidemic.

- **Optimization**: The optimization of the neural network is based on the gradient descent algorithm. The neural network
calculates the loss of the simulation by comparing the amount of deaths with the label data. The aim is to minimize this loss. 
That is why it is important to have automatically differentiable tensors. The image below shows how the gradient descent algorithm works:
![gradient_descent.png](https://raw.githubusercontent.com/ersba/images-model-epidemic-spread-combined/main/gradient_descent.png)

## Process flow
The process flow of the model consists of the following phases:

1. **Calibration**: The simple calibration neural network returns the parameters for the epidemic spread simulation.
2. **Epidemic spread simulation**: The simulation of the epidemic spread with the MARS Framework and the parameters from the neural network.
   1. **Initialization**: The agents, the environment and the layer are initialized.
   2. **Simulation**: The simulation runs for a specified amount of steps.
      1. **Tick**: Each living host agent interacts with other living agents and updates its stage dependent of the stage tensor.
      2. **PostTick**: The layer updates the stage tensor dependent of the interactions in the current tick and aggregates the agents that died in the current tick.
   3. **Output**: The simulation outputs the results in a CSV file and returns the aggregation of dead agents.
3. **Optimization**: The neural network calculates the loss of the simulation by comparing the amount of deaths with the label data. 
With gradient based optimization the neural network adjusts its neurons.
4. **Loop**: If the number of epochs is not reached, the process continues with the calibration phase. If the number of epochs is reached, the process ends.

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
To change this, the value in the `Resources/training.csv` file needs to be adjusted to the desired value.

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

To visualize the result of the simulation run the `visualization.R` file with RStudio. The script will read the Host CSV file
containing the stages of the agents and visualize the result in a graph.

An example of the visualization is shown below:

![example_visualization.png](https://raw.githubusercontent.com/ersba/images-model-epidemic-spread-combined/main/example_visualization.png)
