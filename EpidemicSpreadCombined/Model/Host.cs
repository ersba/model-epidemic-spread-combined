using System;
using System.Collections.Generic;
using Mars.Interfaces.Agents;
using Mars.Interfaces.Annotations;
using Mars.Interfaces.Layers;
using MathNet.Numerics.Distributions;
using NumSharp;

namespace EpidemicSpreadCombined.Model 
{
    public class Host : IAgent<InfectionLayer>
    {
        [PropertyDescription]
        public int Index { get; set; }
        
        [PropertyDescription]
        public int MyAgeGroup { get; set; }
        
        [PropertyDescription]
        public int MyStage { get; set; }
        
        [PropertyDescription] 
        public UnregisterAgent UnregisterHandle { get; set; }
        
        private static float[] _integrals { get; set; }
        
        private InfectionLayer _infectionLayer;
        
        private int _infectedTime;
        
        private int _meanInteractions;

        private int _infinityTime;

        private float _susceptibility;
        
        private bool _exposedToday;
        
        private LearnableParams _learnableParams;
        
        /// <summary>
        /// Initialization of the agent
        /// </summary>
        /// <param name="layer"></param>
        public void Init(InfectionLayer layer)
        {
            _learnableParams = LearnableParams.Instance;
            _infectionLayer = layer;
            _infectionLayer.ContactEnvironment.Insert(this);
            _susceptibility = Params.Susceptibility[MyAgeGroup];
            _infinityTime = Params.Steps + 1;
            InitStage();
            InitMeanInteractions();
            InitInfectedTime();
        }

        /// <summary>
        /// Is executed every tick
        /// </summary>
        public void Tick()
        {
            _exposedToday = false;
            // infected time initialization missing
            Interact();
            Progress();
            if (MyStage == (int)Stage.Mortality) Die();
        }
        
        /// <summary>
        /// Calculates the probability of infection from every interaction with an infected or exposed host
        /// </summary>
        private void Interact()
        {
            if(MyStage == (int) Stage.Susceptible)
            {
                var rDerivatives = new List<float>();
                var product = 1f;
                var inverseDerivatives = new List<float>();
                foreach (Host host in _infectionLayer.ContactEnvironment.GetNeighbors(Index))
                {
                    if (host.MyStage is (int) Stage.Infected or (int) Stage.Exposed)
                    {
                        var infector = Params.Infector[host.MyStage];
                        var bN = Params.EdgeAttribute;
                        var integral =
                            _integrals[
                                Math.Abs(_infectionLayer.GetCurrentTick() - host._infectedTime)];
                        var rDerivative = _susceptibility * infector * bN * integral / _meanInteractions;
                        rDerivatives.Add(rDerivative);
                        var result = (float)_learnableParams.R0Value.numpy() * rDerivative;
                        product *= result;
                        inverseDerivatives.Add(1 / result);
                        Random random = new Random();
                        if (random.NextDouble() < result)
                        {
                            _infectionLayer.ArrayExposedToday[Index] = 1;
                            _exposedToday = true;
                        }
                    }
                }
                var derivatives = np.array(inverseDerivatives) * product * np.array(rDerivatives);
                _infectionLayer.ArrayExposedTodayDerivatives[Index] = np.sum(derivatives);
            }
        }

        /// <summary>
        /// This method is executed when the agent dies upon reaching the mortality stage
        /// </summary>
        private void Die()
        {
           // UnregisterHandle.Invoke(_infectionLayer, this);
        }
        
        /// <summary>
        /// Progresses the stage of the agent
        /// </summary>
        private void Progress()
        {
            if (_exposedToday)
            {
                MyStage = (int)Stage.Exposed;
                _infectedTime = (int) _infectionLayer.GetCurrentTick();
            }
            else
            {
                MyStage = _infectionLayer.ArrayStages[Index];
            }
        }
        
        /// <summary>
        /// Calculates the mean number of interactions based on the age group of the agent
        /// </summary>
        private void InitMeanInteractions()
        {
            var childAgent = MyAgeGroup <= Params.ChildUpperIndex;
            var adultAgent = MyAgeGroup > Params.ChildUpperIndex && MyAgeGroup <= Params.AdultUpperIndex;
            var elderAgent = MyAgeGroup > Params.AdultUpperIndex;
            
            if (childAgent) _meanInteractions = Params.Mu[0];
            else if (adultAgent) _meanInteractions = Params.Mu[1];
            else if (elderAgent) _meanInteractions = Params.Mu[2];
        }
        
        /// <summary>
        /// Initializes the stage of the agent by setting it to the stage value of the infection layer
        /// </summary>
        private void InitStage()
        {
            MyStage = _infectionLayer.ArrayStages[Index];
        }

        /// <summary>
        /// Initializes the time the agent has been infected based on the stage it is in
        /// </summary>
        private void InitInfectedTime()
        {
            switch (MyStage)
            {   
                case (int) Stage.Susceptible:
                    _infectedTime = _infinityTime;
                    break;
                case (int)Stage.Exposed:
                    _infectedTime = 0;
                    break;
                case (int)Stage.Infected:
                    _infectedTime = 1 - (int)Params.ExposedToInfectedTime.numpy();
                    break;
                case (int)Stage.Recovered:
                    _infectedTime = _infinityTime;
                    break;
                case (int)Stage.Mortality:
                    _infectedTime = _infinityTime;
                    break;
            }
        }
        
        /// <summary>
        /// Initializes the integrals for the gamma distribution needed for the infection probability calculation
        /// </summary>
        public static void InitLamdaGammaIntegrals()
        {
            var scale = 5.15;
            var rate = 2.14;
            var b = rate * rate / scale;
            var a = scale / b;
            var res = new List<float>();

            for (int t = 1; t <= Params.Steps + 10; t++)
            {
                var cdfAtTimeT = Gamma.CDF(a, b, t);
                var cdfAtTimeTMinusOne = Gamma.CDF(a, b, t - 1);
                res.Add((float)(cdfAtTimeT - cdfAtTimeTMinusOne));
            }
            _integrals = res.ToArray();
        }

        public Guid ID { get; set; }
    }
    
    
    
    
}