using Mars.Components.Layers;
using Mars.Core.Data;
using Mars.Interfaces.Data;
using Mars.Interfaces.Layers;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined.Model
{
    public class TestLayer : AbstractLayer, ISteppedActiveLayer
    {
        public IAgentManager AgentManager { get; private set; }
        
        public Tensor Deaths { get; private set; }
        
        private LearnableParams _learnableParams;

        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            _learnableParams = LearnableParams.Instance;
            Deaths = tf.constant(0);
            return initiated;
        }



        public void Tick()
        {
            // var probability = tf.expand_dims(_learnableParams.InitialInfectionRate / _learnableParams.InitialInfectionRate * 10, axis: 0);
            // probability = tf.expand_dims(probability, axis: 1);
            // var p = tf.concat(new [] { probability, 1 - probability }, axis: 1);
            // var mortality = tf.cast(GumbelSoftmax.Execute(p)[Slice.All, 0], dtype: TF_DataType.TF_BOOL);
            //
            // var equalStagesMortality = tf.equal(_learnableParams.InitialInfectionRate, tf.constant(_learnableParams.InitialInfectionRate, _learnableParams.InitialInfectionRate.dtype));
            // var castEqualStagesMortality = tf.cast(equalStagesMortality, TF_DataType.TF_FLOAT);
            // var deaths = tf.reduce_sum(castEqualStagesMortality);
            
            Deaths += _learnableParams.InitialInfectionRate * 150;
            // Deaths += tf.constant(_learnableParams.InitialInfectionRate * 10 + mortality) + tf.cast(tf.equal(_learnableParams.InitialInfectionRate, tf.constant(_learnableParams.InitialInfectionRate, _learnableParams.InitialInfectionRate.dtype)), TF_DataType.TF_FLOAT);
        }

        public void PreTick()
        {
            
        }

        public void PostTick()
        {
            
        }
    }
}