using System.Data;
using static System.Net.Mime.MediaTypeNames;

namespace ByeByeNeuralNet
{
    internal class Program
    {

        static float[][] newRandomWeights(int[] layerSizes, Random rnd)
        {
            
            List<float[]> weightsList = new List<float[]>();
            for (int i = 0; i < layerSizes.Length; i++)
            {
                int weightsSize = layerSizes[i];
                if (i != layerSizes.Length - 1)
                {
                    weightsSize = layerSizes[i] * layerSizes[i + 1];
                }

                float[] tempWeights = new float[weightsSize];
                for (int j = 0; j < weightsSize; j++)
                {
                    
                    tempWeights[j] = ((float)rnd.Next(0, 256) /255f);

                }
                weightsList.Add(tempWeights);
            }

            return weightsList.ToArray();
        }

        static void addWeights(ref float[][] origin, float[][] toadd)
        {
            for (int i = 0; i < origin.Length;i++)
            {
                for (int j = 0;j < origin[i].Length;j++)
                {
                    origin[i][j] *= toadd[i][j];
                }
            }
        }
        static void Main(string[] args)
        {
            Random rnd = new Random();
            List<Image> trainSet = MnistReader.ReadTrainingData().ToList();
            //IEnumerable<Image> testSet = MnistReader.ReadTestData();

            int[] layerSizes = { 784, 20, 20, 10 };

            List<float[]> valuesList = new List<float[]>();
            List<float[]> weightsList = new List<float[]>();
            for (int i = 0; i < layerSizes.Length; i++)
            {
                valuesList.Add(new float[layerSizes[i]]);
            }

            float[][] values = valuesList.ToArray();
            float[][] weights = newRandomWeights(layerSizes, rnd);

            int correctGuesses = 0;
            int guesses = 0;
            int overallGuesses = 0;
            int overallCorrect = 0;
            foreach(Image image in trainSet)
            {
                //Console.WriteLine("OG MODEL GUESS = " + Guess(layerSizes, values, weights, image).ToString() + " LABEL = " + image.Label.ToString());
                if (Guess(layerSizes, values, weights, image) == image.Label)
                {
                    correctGuesses++;
                    overallCorrect++;
                    
                }
                guesses++;
                overallGuesses++;

                if (guesses == 1000)
                {
                    Console.WriteLine("ERROR RATE = " + ((double)correctGuesses / (double)guesses).ToString() + " " + weights[0][0].ToString());
                    guesses = 0;
                    correctGuesses = 0;

                }

                
            }

            Console.WriteLine("Overall Error Rate = " + ((double)overallCorrect / (double)overallGuesses).ToString());

        }

        static int Guess(int[] layerSizes,  float[][] values,  float[][] weights, Image image)
        {
            Random rnd = new Random();
            for (int i = 0; i < values.Length; i++)
            {
                for (int j = 0; j < values[i].Length; j++)
                {
                    values[i][j] = 0f;
                }

            }

            // Load image
            for (int i = 0; i < values[0].Length; i++)
            {
                values[0][i] = (float)image.Data[i] / 255f;
            }

            // foreach layer
            for (int layNum = 0; layNum < layerSizes.Length - 1; layNum++)
            {
                int layerSize = layerSizes[layNum];
                int nextLayerSize = layerSizes[layNum + 1];
                int numWeights = weights[layNum].Length;

                // foreach value in layer
                for (int valNum = 0; valNum < values[layNum].Length; valNum++)
                {
                    // foreach weight assigned to that value
                    for(int weightNum = 0; weightNum < nextLayerSize; weightNum++)
                    {
                        float weightValue = weights[layNum][weightNum * valNum];
                        values[layNum + 1][weightNum] += ((values[layNum][valNum] * weightValue)  / (float)layerSize);
                        
                    }
                }
            }

            //Console.Write("[");
            int maxIndex = 0;
            for(int i = 0; i < values[layerSizes.Length - 1].Length; i++)
            {
                if (values[layerSizes.Length - 1][i] > values[layerSizes.Length - 1][maxIndex])
                {
                    maxIndex = i;
                }

                //Console.Write(values[layerSizes.Length - 1][i].ToString() + " ");
            }
            //Console.Write("]");
            bool pushFromCenter = false;
            bool isCorrect = false;
            if (maxIndex == (int)image.Label)
            {
                pushFromCenter = true;
                isCorrect = true;
            }

            // foreach layer
            for (int layNum = 0; layNum < layerSizes.Length - 1; layNum++)
            {
                // foreach weight
                for (int weightNum = 0; weightNum < weights[layNum].Length; weightNum++)
                {

                    float currentWeight = weights[layNum][weightNum];
                    float bubble = 0;


                    if(isCorrect)
                    {
                        weights[layNum][weightNum] = weights[layNum][weightNum] / 2f; 
                    }
                    else
                    {
                        weights[layNum][weightNum] = weights[layNum][weightNum] * 2f;
                    }

                    if(weights[layNum][weightNum] > 0.999f)
                    {
                        weights[layNum][weightNum] = 0.998f;
                    }
                    if (weights[layNum][weightNum] < 0.001f)
                    {
                        weights[layNum][weightNum] = 0.001f;
                    }
                }
            }


            return maxIndex;



        }

        static float increaseClamp(float toIncrease)
        {
            double closeness = Math.Truncate(toIncrease);

            if(closeness > 0.95f)
            {
                return toIncrease + 0.051f;
            }
            if (closeness > 0.90f)
            {
                return toIncrease + 0.005f;
            }
            if (closeness > 0.80f)
            {
                return toIncrease + 0.01f;
            }
            if (closeness > 0.70f)
            {
                return toIncrease + 0.04f;
            }
            if (closeness > 0.60f)
            {
                return toIncrease + 0.05f;
            }
            return toIncrease + 0.06f;
        }

        static float decreaseClamp(float toDecrease)
        {
            double closeness = Math.Truncate(toDecrease);

            if (closeness > 0.95f)
            {
                return toDecrease - 0.005f;
            }
            if (closeness > 0.90f)
            {
                return toDecrease - 0.01f;
            }
            if (closeness > 0.80f)
            {
                return toDecrease - 0.02f;
            }
            if (closeness > 0.70f)
            {
                return toDecrease - 0.04f;
            }
            if (closeness > 0.60f)
            {
                return toDecrease - 0.05f;
            }
            if (closeness > 0.5f)
            {
                return toDecrease - 0.05f;
            }
            if(closeness < 0.1f)
            {
                return toDecrease - 0.006f;
            }
            return toDecrease - 0.03f;
        }

    }

    // Classes MnistReader, Image, and Extensions pulled from
    // https://stackoverflow.com/questions/49407772/reading-mnist-database
    // https://stackoverflow.com/users/3715778/koryakinp
    // https://stackoverflow.blog/2009/06/25/attribution-required/
    // *Has been modified*
    public static class MnistReader
    {
        private const string TrainImages = "mnist/train-images-idx3-ubyte";
        private const string TrainLabels = "mnist/train-labels-idx1-ubyte";
        private const string TestImages = "mnist/t10k-images-idx3-ubyte";
        private const string TestLabels = "mnist/t10k-labels-idx1-ubyte";
        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                byte[] bytes = images.ReadBytes(width * height);
                byte[,] arrFolded = new byte[height, width];
                
                arrFolded.ForEach((j, k) => arrFolded[j, k] = bytes[j * height + k]);

                // Sketch .ReadByte() usage 
                yield return new Image()
                {
                    Label = labels.ReadByte(),
                    Data = bytes,
                    DataFolded = arrFolded,
                };
            }
        }
    }
    public class Image
    {
        public byte Label { get; set; }
        public byte[] Data { get; set; }
        public byte[,] DataFolded { get; set; }

    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}