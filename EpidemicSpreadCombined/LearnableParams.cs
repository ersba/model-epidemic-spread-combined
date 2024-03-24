using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpreadCombined
{
    public class LearnableParams
    {
        private static LearnableParams _instance;
        public Tensor InitialInfectionRate { get; set; }
        public Tensor MortalityRate { get; set; }

        private LearnableParams()
        {
            // InitialInfectionRate = tf.constant(0.05, dtype: TF_DataType.TF_FLOAT);
            // InitialInfectionRate = tf.constant(0.5, dtype: TF_DataType.TF_FLOAT);
            InitialInfectionRate = tf.constant(0.05, dtype: TF_DataType.TF_FLOAT);
            // InitialInfectionRate = tf.constant(0.99, dtype: TF_DataType.TF_FLOAT);
            MortalityRate = tf.constant(0.1, dtype: TF_DataType.TF_FLOAT);
        }
        
        public static LearnableParams Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new LearnableParams();
                }
                return _instance;
            }
        }
    }
}