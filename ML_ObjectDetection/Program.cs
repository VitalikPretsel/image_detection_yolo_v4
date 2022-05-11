using System;
using System.Drawing;
using System.IO;
using ML_ObjectDetection.Drawer;
using ML_ObjectDetection.MachineLearning;

namespace ML_ObjectDetection
{
    class Program
    {
        private const string _modelPath = @"..\..\..\Assets\Model\yolov4.onnx";
        private const string _imageFolder = @"..\..\..\Assets\Data";
        private const string _imageOutputFolder = @"..\..\..\Assets\Output";
        private static readonly string[] _classesNames = new string[] {
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        static void Main()
        {
            Directory.CreateDirectory(_imageOutputFolder);
            var trainer = new Trainer();

            Console.WriteLine("Build and train YOLO V4 model...");
            var trainedModel = trainer.BuildAndTrain(_modelPath);

            Console.WriteLine("Create predictor...");
            var predictor = new Predictor(trainedModel);

            Console.WriteLine("Run predictions on images...");

            DirectoryInfo directoryInfo = new DirectoryInfo(_imageFolder);
            FileInfo[] files = directoryInfo.GetFiles("*.jpg");

            foreach (FileInfo file in files)
            {
                using (var image = new Bitmap(Image.FromFile(Path.Combine(_imageFolder, file.Name))))
                {
                    var predict = predictor.Predict(image);
                    var results = predict.GetResults(_classesNames);

                    DrawResults.DrawAndStore(_imageOutputFolder, file.Name, results, image);
                }
            }

            Console.WriteLine($"Check images in the output folder {_imageOutputFolder}...");
        }
    }
}
