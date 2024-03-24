using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined.Model;

public static class SimpleSimulation
{
    public static Tensor Execute()
    {
        var death = tf.constant(0f);
        var learnparam = LearnableParams.Instance;
        death += tf.constant(1000) * learnparam.InitialInfectionRate * 20;
        return death;
    }
}