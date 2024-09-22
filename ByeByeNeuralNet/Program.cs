using System.Data;
using static System.Net.Mime.MediaTypeNames;

namespace ByeByeNeuralNet
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Random rnd = new Random();
            List<Image> trainSet = MnistReader.ReadTrainingData().ToList();
            //IEnumerable<Image> testSet = MnistReader.ReadTestData();

            int[] layerSizes = { 784, 20, 20, 10 };

            List<float[]> valuesList = new List<float[]>();
            List<byte[]> weightsList = new List<byte[]>();
            for (int i = 0; i < layerSizes.Length; i++)
            {
                int weightsSize = layerSizes[i];
                if (i != layerSizes.Length - 1)
                {
                    weightsSize = layerSizes[i] * layerSizes[i + 1];
                }

                byte[] tempWeights = new byte[weightsSize];
                for (int j = 0; j < weightsSize; j++)
                {
                    tempWeights[j] = (byte)rnd.Next(0, 256);
                }
                weightsList.Add(tempWeights);

                valuesList.Add(new float[layerSizes[i]]);

            }

            float[][] values = valuesList.ToArray();
            byte[][] weights = weightsList.ToArray();


            int correctGuesses = 0;
            int guesses = 0;
            foreach(Image image in trainSet)
            {
                if (Guess(layerSizes, values, weights, image) == image.Label)
                {
                    correctGuesses++;
                }
                guesses++;
                
            }

            Console.WriteLine("ERROR RATE = " + ((double) correctGuesses / (double)guesses).ToString());


        }

        static int Guess(int[] layerSizes,  float[][] values, byte[][] weights, Image image)
        {
            // Load image
            for(int i = 0; i < values[0].Length; i++)
            {
                values[0][i] = image.Data[i] / 255.00f;
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
                        byte weightValue = weights[layNum][weightNum * valNum];
                        values[layNum + 1][weightNum] += (values[layNum][valNum] * (weightValue / 255.00f)) / (float)layerSize;
                    }
                }
            }

            int maxIndex = 0;
            for(int i = 0; i < values[layerSizes.Length - 1].Length; i++)
            {
                if (values[layerSizes.Length - 1][i] > values[layerSizes.Length - 1][maxIndex])
                {
                    maxIndex = i;
                }
            }

            return maxIndex;
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