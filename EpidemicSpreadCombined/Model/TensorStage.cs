using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined.Model
{
    public static class TensorStage
    {
        public static Tensor Susceptible = tf.constant(0);
        
        public static Tensor Exposed = tf.constant(1);
        
        public static Tensor Infected = tf.constant(2);
        
        public static Tensor Recovered = tf.constant(3);
        
        public static Tensor Mortality = tf.constant(4);
    }
}