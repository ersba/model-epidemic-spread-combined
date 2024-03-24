using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined
{
    public static class Params
    {
        public static readonly int ChildUpperIndex = 1;
        
        public static readonly int AdultUpperIndex = 6;
        
        public static readonly int[] Mu = { 2, 4, 3 };

        public static int Steps = 0;

        public static int AgentCount = 0;

        public static string ContactEdgesPath = "Resources/contact_edges.csv";

        public static string OptimizedParametersPath = "Resources/optimized_parameters.csv";
        
        public static readonly float EdgeAttribute = 1f;
        
        public static readonly float[] Susceptibility = {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f};
        
        public static readonly float[] Infector = {0.0f, 0.33f, 0.72f, 0.0f, 0.0f};

        public static readonly float R0Value = 5.18f;
        
        public static Tensor ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_INT32);
            
        public static Tensor InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_INT32);
    } 
}