using System.Linq;
using Mars.Common.IO.Mapped.Arrays;
using Mars.Components.Layers;
using Mars.Core.Data;
using Mars.Interfaces.Data;
using Mars.Interfaces.Layers;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined.Model

{
    public class InfectionLayer : AbstractLayer, ISteppedActiveLayer
    {
        public ContactGraphEnvironment ContactEnvironment { get; private set; }
        
        private Tensor Stages { get; set; }
        
        public Tensor Deaths { get; private set; }
        
        public int[] ArrayStages { get; private set; }
        
        public int[] ArrayExposedToday { get; set; }
        
        public IAgentManager AgentManager { get; private set; }

        private LearnableParams _learnableParams;
        
        private int _infinityTime;

        private Tensor _nextStageTimes;
        


        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            _learnableParams = LearnableParams.Instance;
            _infinityTime = Params.Steps + 1;
            ContactEnvironment = new ContactGraphEnvironment();
            InitStages();
            InitNextStageTimes();
            ArrayExposedToday = new int[Params.AgentCount];
            Host.SetLamdaGammaIntegrals();
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
            ContactEnvironment.ReadCSV();
            Deaths = tf.constant(0f);

            return initiated;
        }

        public void Tick()
        {
            
        }

        public void PreTick()
        { 
            
        }

        public void PostTick()
        {
            var recoveredAndDead= Stages * tf.equal(tf.cast(Stages, TF_DataType.TF_INT32), 
                tf.constant(Stage.Infected, TF_DataType.TF_INT32)) * tf.less_equal(_nextStageTimes, 
                (int) Context.CurrentTick) / (float)Stage.Infected;
            
            Deaths += tf.reduce_sum(recoveredAndDead) * _learnableParams.MortalityRate;
            
            var exposedToday = tf.expand_dims(tf.constant(ArrayExposedToday));
            
            var nextStages = UpdateStages(exposedToday);
            
            _nextStageTimes = UpdateNextStageTimes(exposedToday);
            
            ArrayExposedToday = new int[Params.AgentCount];
            
            Stages = nextStages;
            
            ArrayStages = tf.cast(Stages, TF_DataType.TF_INT32).numpy().ToArray<int>();
        }

        private Tensor UpdateNextStageTimes(Tensor exposedToday)
        {
            var newTransitionTimes = tf.identity(_nextStageTimes);
            var currentStages = tf.cast(Stages, TF_DataType.TF_INT32);
            var conditionInfectedAndTransitionTime = tf.logical_and(tf.equal(currentStages, 
                    tf.constant((int)Stage.Infected)), tf.equal(_nextStageTimes, 
                tf.constant((int)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionInfectedAndTransitionTime, 
                tf.fill(tf.shape(newTransitionTimes), tf.constant(_infinityTime)), newTransitionTimes);
            
            var conditionExposedAndTransitionTime = tf.logical_and(tf.equal(currentStages, 
                    tf.constant((int)Stage.Exposed)), tf.equal(_nextStageTimes, 
                tf.constant((int)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionExposedAndTransitionTime,
                (tf.fill(tf.shape(newTransitionTimes),tf.constant((int)Context.CurrentTick) + 
                                                      Params.InfectedToRecoveredTime)), newTransitionTimes);
            // tf.print(tf.shape(_nextStageTimes));
            var result = exposedToday * (tf.constant((int)Context.CurrentTick + 1) + Params.ExposedToInfectedTime)
                + (tf.fill(tf.shape(exposedToday), tf.constant(1)) - exposedToday) * newTransitionTimes;
            return result;
        }

        private Tensor UpdateStages(Tensor exposedToday)
        {
            var currentStages = tf.cast(Stages, TF_DataType.TF_INT32);
            
            var transitionToInfected = tf.cast(tf.less_equal(_nextStageTimes, (int)Context.CurrentTick),
                    TF_DataType.TF_INT32) * (int)Stage.Infected + tf.cast(tf.greater(_nextStageTimes, 
                (int)Context.CurrentTick), TF_DataType.TF_INT32) * (int)Stage.Exposed;
            
            var transitionToMortalityOrRecovered = tf.cast(tf.less_equal(_nextStageTimes, 
                    (int)Context.CurrentTick), TF_DataType.TF_INT32) * (int)Stage.Recovered + 
                                                   tf.cast(tf.greater(_nextStageTimes, (int)Context.CurrentTick), 
                                                       TF_DataType.TF_INT32) * (int)Stage.Infected;
            
            var probabilityMortality = tf.cast(tf.logical_and(tf.equal(currentStages, 
                    tf.constant((int)Stage.Infected)), tf.less_equal(_nextStageTimes, (int)Context.CurrentTick)), 
                TF_DataType.TF_INT32) * (_learnableParams.MortalityRate);
            
            var p = tf.concat(new [] { probabilityMortality, 1 - probabilityMortality }, axis: 1);
            var mortality = tf.cast(GumbelSoftmax.Execute(p)[Slice.All, 0], dtype: TF_DataType.TF_BOOL);
            
            transitionToMortalityOrRecovered = tf.where(mortality, tf.fill(
                tf.shape(transitionToMortalityOrRecovered), (int)Stage.Mortality), 
                transitionToMortalityOrRecovered);
            
            var stageProgression =
                tf.cast(tf.equal(currentStages, tf.constant((int)Stage.Susceptible)), TF_DataType.TF_INT32) *
                (int)Stage.Susceptible +
                tf.cast(tf.equal(currentStages, tf.constant((int)Stage.Recovered)), TF_DataType.TF_INT32) *
                (int)Stage.Recovered +
                tf.cast(tf.equal(currentStages, tf.constant((int)Stage.Mortality)), TF_DataType.TF_INT32) *
                (int)Stage.Mortality +
                tf.cast(tf.equal(currentStages, tf.constant((int)Stage.Exposed)), TF_DataType.TF_INT32) *
                transitionToInfected * Stages / (int)Stage.Exposed +
                tf.cast(tf.equal(currentStages, tf.constant((int)Stage.Infected)), TF_DataType.TF_INT32) *
                transitionToMortalityOrRecovered * Stages / (int)Stage.Infected;

            var nextStages = exposedToday * (float)Stage.Exposed + stageProgression;
            return nextStages;
        }
        
        private void InitStages()
        {
            var ones = tf.ones(new Shape(Params.AgentCount, 1));
            var predColumn = ones * _learnableParams.InitialInfectionRate;
            var oneMinusPredColumn = ones * (1 - _learnableParams.InitialInfectionRate);
            var p = tf.concat(new [] { predColumn, oneMinusPredColumn }, axis: 1);
            Stages = tf.expand_dims(tf.cast(GumbelSoftmax.Execute(p)[Slice.All,0], dtype: TF_DataType.TF_FLOAT), 
                axis: 1) * 2;
            ArrayStages = tf.cast(Stages, TF_DataType.TF_INT32).numpy().ToArray<int>();
        }

        private void InitNextStageTimes()
        {
            var currentStages = tf.cast(Stages, TF_DataType.TF_INT32);

            _nextStageTimes = _infinityTime * tf.ones(new Shape(Params.AgentCount, 1), 
                dtype: TF_DataType.TF_INT32);
            
            var exposedCondition = tf.equal(currentStages, tf.constant((int)Stage.Exposed));
            var infectedCondition = tf.equal(currentStages, tf.constant((int)Stage.Infected));
            
            _nextStageTimes = tf.where(exposedCondition, tf.fill(tf.shape(_nextStageTimes), 
                Params.ExposedToInfectedTime + tf.constant(1)), _nextStageTimes);
            _nextStageTimes = tf.where(infectedCondition, tf.fill(tf.shape(_nextStageTimes), 
                Params.InfectedToRecoveredTime + tf.constant(1)), _nextStageTimes);
        }
    }
}