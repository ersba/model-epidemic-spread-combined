using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined
{
    /// <summary>
    /// Implementation of the Gumbel-Softmax function.
    /// </summary>
    public static class GumbelSoftmax
    {
        /// <summary>
        /// This method is used to perform the Gumbel-Softmax trick, which is a method for sampling from a discrete
        /// probability distribution that also provides differentiable samples.
        /// </summary>
        /// <param name="probabilities"></param>
        /// <param name="temperature"></param>
        /// <returns></returns>
        public static Tensor Execute(Tensor probabilities, double temperature = 1.0)
        {
            var gumbelNoise = -tf.math.log(-tf.math.log(tf.random.uniform(probabilities.shape)));
            var softSample = tf.nn.softmax((tf.math.log(probabilities + 1e-9) + gumbelNoise) / temperature);
            var cutSoftSample = tf.constant(softSample.numpy());
            var hardSample = tf.cast(tf.equal(softSample, tf.reduce_max(softSample, axis: 1, keepdims: true)),TF_DataType.TF_INT32);
            softSample = tf.stop_gradient(hardSample - cutSoftSample) + softSample;
            return softSample;
        }
    }
}

