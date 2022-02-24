using Emgu.CV;
using Emgu.CV.Structure;
using static Emgu.CV.CvInvoke;

var faceCascade = new CascadeClassifier("./haarcascade_frontalface_alt2.xml");

string[] files =
{
    "f1.jpg",
    "f2.jpg",
    "f3.png",
    "f4.jpg",
    "f5.jpg"
};

foreach(var filename in files)
{
    if (filename == "exit") return;
    var image = Imread(filename);
    var rectangles = faceCascade.DetectMultiScale(image, 1.05, 6, new System.Drawing.Size(10, 10));
    foreach (var rect in rectangles)
        Rectangle(image, rect, new MCvScalar(0, 0, 255), 3);
    Imwrite(filename + ".result.jpg", image);
}
Console.WriteLine("Successful");
