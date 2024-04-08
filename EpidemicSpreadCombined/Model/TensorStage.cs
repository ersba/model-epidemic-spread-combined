using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined.Model
{
    /// <summary>
    /// The TensorStage static class provides constant Tensor values representing different stages of an epidemic.
    /// These stages include Susceptible, Exposed, Infected, Recovered, and Mortality. 
    /// </summary>
    public static class TensorStage
    {
        public static Tensor Susceptible = tf.constant(0);
        
        public static Tensor Exposed = tf.constant(1);
        
        public static Tensor Infected = tf.constant(2);
        
        public static Tensor Recovered = tf.constant(3);
        
        public static Tensor Mortality = tf.constant(4);
    }
}