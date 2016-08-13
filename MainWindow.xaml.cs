namespace KinectReferee
{
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.Windows;
    using System.Windows.Media;
    using Microsoft.Kinect;
    using System.Speech.Recognition;
    using System;
    using System.Linq;
    using System.Windows.Media.Imaging;

    /// <summary>
    /// Interaction logic for MainWindow
    /// </summary>
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Microsoft.Design", "CA1001:TypesThatOwnDisposableFieldsShouldBeDisposable",
       Justification = "In a full-fledged application, the SpeechRecognitionEngine object should be properly disposed. For the sake of simplicity, we're omitting that code in this sample.")]
    public partial class MainWindow : Window
    {
        /// <summary>
        /// Active Kinect sensor.
        /// </summary>
        private KinectSensor kinectSensor;

        protected bool game_started = false;

        /// <summary>
        /// Speech recognition engine using audio data from Kinect.
        /// </summary>
        private SpeechRecognitionEngine speechEngine;
        // Speech utterance confidence below which we treat speech as if it hadn't been heard
        const double ConfidenceThreshold = 0.5;

        private MultiSourceFrameReader MultiFrameReader;
        private Body[] bodies;
        private Dictionary<ulong, Body> players;
        private OpenCvSharp.Window WindowMat;
        private byte[] bodyIndexPixels;
        private byte[] depthPixels;

        // convert 0 - 8000mm to 0 - 255
        const double MapDepthToByte = 255.0 / 8000.0;
        const double MapByteToDepth = 1 / MapDepthToByte;

        private WriteableBitmap bodyIndexBitmap;

        /// <summary>
        /// Gets the bitmap to display
        /// </summary>
        public ImageSource ImageSource
        {
            get
            {
                return this.bodyIndexBitmap;
            }
        }

        /// <summary>
        /// Initializes a new instance of the MainWindow class.
        /// </summary>
        public MainWindow()
        {
            // Only one sensor is supported
            this.kinectSensor = KinectSensor.GetDefault();
            if (this.kinectSensor == null)
                throw new PlatformNotSupportedException(Properties.Resources.NoKinectReady);

            // open the sensor
            this.kinectSensor.Open();
            this.MultiFrameReader = this.kinectSensor.OpenMultiSourceFrameReader(
                FrameSourceTypes.Body | FrameSourceTypes.BodyIndex
                | FrameSourceTypes.Depth
                );
            this.MultiFrameReader.MultiSourceFrameArrived += this.MultiReader_FrameArrived;

            this.speechEngine = new SpeechRecognitionEngine(new System.Globalization.CultureInfo("ja-JP"));
            // Create a grammar from grammar definition XML file.
            this.speechEngine.LoadGrammar(new Grammar("SpeechGrammar.xml"));
            this.speechEngine.SpeechRecognized += this.SpeechRecognized;

            // For long recognition sessions (a few hours or more), it may be beneficial to turn off adaptation of the acoustic model. 
            // This will prevent recognition accuracy from degrading over time.
            //speechEngine.UpdateRecognizerSetting("AdaptationOn", 0);

            // set KINECT to default mic before running application
            this.speechEngine.SetInputToDefaultAudioDevice();
            this.speechEngine.RecognizeAsync(RecognizeMode.Multiple);

            {
                var des = this.kinectSensor.DepthFrameSource.FrameDescription;
                this.depthPixels = new byte[des.Width * des.Height];
                this.bodyIndexBitmap = new WriteableBitmap(
                    des.Width, des.Height,
                    96.0, 96.0, PixelFormats.Bgr32, null);
            }
            {
                var des = this.kinectSensor.BodyIndexFrameSource.FrameDescription;
                this.bodyIndexPixels = new byte[des.Width * des.Height];
            }

            this.players = new Dictionary<ulong, Body>();
            this.bodies = new Body[this.kinectSensor.BodyFrameSource.BodyCount];
            this.WindowMat = new OpenCvSharp.Window("Test1");

            using (var sp = new System.Media.SoundPlayer(@"c:\Windows\Media\notify.wav"))
                sp.Play();

            this.DataContext = this;
            this.InitializeComponent();
        }

        /// <summary>
        /// Execute initialization tasks.
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
        }

        private void MultiReader_FrameArrived(object sender, MultiSourceFrameArrivedEventArgs e)
        {
            if (!this.game_started) return;

            var frame = e.FrameReference.AcquireFrame();
            if (frame == null) return;

            var depthFrame = frame.DepthFrameReference.AcquireFrame();
            if (depthFrame != null)
                using (depthFrame)
                using (var buffer = depthFrame.LockImageBuffer())
                {
                    var des = depthFrame.FrameDescription;
                    this.ProcessDepthFrameData(
                        buffer.UnderlyingBuffer,
                        buffer.Size,
                        depthFrame.DepthMinReliableDistance,
                        ushort.MaxValue);
                }

            var bodyFrame = frame.BodyFrameReference.AcquireFrame();
            var bodyIndexFrame = frame.BodyIndexFrameReference.AcquireFrame();
            if (bodyFrame != null && bodyIndexFrame != null)
                using (bodyFrame) using (bodyIndexFrame)
                {
                    bodyFrame.GetAndRefreshBodyData(this.bodies);
                    var count = this.bodies.Where(b => b.IsTracked).Count();
                    if (count > 0)
                    {
                        this.players = new Dictionary<ulong, Body>();
                        foreach (var body in this.bodies)
                        {
                            if (body.IsTracked
                                // && body.HandRightConfidence == TrackingConfidence.High
                                )
                                this.players[body.TrackingId] = body;
                        }
                    }
                    bodyIndexFrame.CopyFrameDataToArray(this.bodyIndexPixels);

                    var bodyindex = new OpenCvSharp.Mat(
                                bodyIndexFrame.FrameDescription.Height,
                                bodyIndexFrame.FrameDescription.Width,
                                OpenCvSharp.MatType.CV_8UC1,
                                this.bodyIndexPixels);
                    /*
                    // LINQ is slow...
                    var nobodyPixels = this.bodyIndexPixels.Select(b => b == 0xff ? (byte)1 : (byte)0).ToArray();
                    var nobody = new OpenCvSharp.Mat(
                                bodyIndexFrame.FrameDescription.Height,
                                bodyIndexFrame.FrameDescription.Width,
                                OpenCvSharp.MatType.CV_8UC1,
                                nobodyPixels);
                    var depth = new OpenCvSharp.Mat(
                                bodyIndexFrame.FrameDescription.Height,
                                bodyIndexFrame.FrameDescription.Width,
                                OpenCvSharp.MatType.CV_8UC1,
                                this.depthPixels);
                    */
                    var result = new OpenCvSharp.Mat(
                        new OpenCvSharp.Size(
                            bodyIndexFrame.FrameDescription.Width,
                            bodyIndexFrame.FrameDescription.Height),
                        OpenCvSharp.MatType.CV_8UC4);
                    /*
                    // Mul is slow...
                    using (var tmp = nobody.Mul(depth))
                        OpenCvSharp.Cv2.CvtColor(tmp, result, OpenCvSharp.ColorConversionCodes.GRAY2BGRA);
                        */
                    OpenCvSharp.Cv2.CvtColor(bodyindex, result, OpenCvSharp.ColorConversionCodes.GRAY2BGRA);
                    OpenCvSharp.LineSegmentPoint[] lsps;
                    using (var cannied = new OpenCvSharp.Mat())
                    {
                        // OpenCvSharp.Cv2.Canny(nobody, cannied, 50.0, 200.0, 3, false);
                        OpenCvSharp.Cv2.Canny(bodyindex, cannied, 50.0, 200.0, 3, false);
                        lsps = OpenCvSharp.Cv2.HoughLinesP(cannied, 50, 3 * Math.PI / 180, 100, 100, 30);
                    }

                    const int r = 25;
                    const double r2 = r * r;

                    // （持ってる得物を含む）フレーム中の直線が
                    // 25pxより小さいものしかなかったら処理をやめる
                    if (lsps.Where(l => Math.Pow(l.P1.X - l.P2.X, 2) + Math.Pow(l.P1.Y - l.P2.Y, 2) > r2).
                        Count() == 0) return;
                    int i = 0;
                    foreach (KeyValuePair<ulong, Body> player in this.players)
                    {
                        var joints = player.Value.Joints;
                        var rhand = this.kinectSensor.CoordinateMapper.MapCameraPointToDepthSpace(joints[JointType.HandRight].Position);
                        var des = bodyIndexFrame.FrameDescription;
                        var hand = new OpenCvSharp.Point(
                            rhand.X <= des.Width ? rhand.X + 1 : des.Width,
                            rhand.Y <= des.Height ? rhand.Y + 1 : des.Height);

                        var lsp = lsps.OrderBy(
                            l => Math.Min(
                                Math.Pow(l.P1.X - hand.X, 2) + Math.Pow(l.P1.Y - hand.Y, 2),
                                Math.Pow(l.P2.X - hand.X, 2) + Math.Pow(l.P2.Y - hand.Y, 2))
                        ).Take(10).OrderByDescending(
                            l => Math.Pow(l.P1.X - l.P2.X, 2) + Math.Pow(l.P1.Y - l.P2.Y, 2)
                        ).First();

                        result.Circle(hand, r, OpenCvSharp.Scalar.Blue, 10);
                        result.PutText(player.Key.ToString(), hand, OpenCvSharp.HersheyFonts.HersheyComplex, 1, OpenCvSharp.Scalar.Violet);

                        var find_r2 = 2500;
                        var p1r2 = Math.Pow(lsp.P1.X - hand.X, 2) +
                            Math.Pow(lsp.P1.Y - hand.Y, 2);
                        var p2r2 = Math.Pow(lsp.P2.X - hand.X, 2) +
                            Math.Pow(lsp.P2.Y - hand.Y, 2);
                        {
                            // 手に一番近い直線の
                            // 手から遠い点を得物の先端の座標とする
                            var tip = (p1r2 <= find_r2) ? lsp.P2 : lsp.P1;
                            if (tip.X > des.Width) tip.X = des.Width;
                            if (tip.X == 0) tip.X = 1;
                            if (tip.Y > des.Height) tip.Y = des.Height;
                            if (tip.Y == 0) tip.Y = 1;
                            var helve = (p1r2 <= find_r2) ? lsp.P1 : lsp.P2;
                            System.Console.WriteLine("{0} 得物発見！", DateTime.Now);
                            result.Circle(helve, 20, OpenCvSharp.Scalar.Red, 10);
                            result.Circle(tip, 20, OpenCvSharp.Scalar.Green, 10);

                            if (i < 2) ++i;
                            else
                            {
                                System.Console.WriteLine("Find 2people+");
                            }

                            // 得物の先端が自分と違う人物がいる領域にあれば勝負がついたと判断する
                            var opposition = this.bodyIndexPixels[bodyIndexFrame.FrameDescription.Width * (tip.Y - 1) + tip.X - 1];
                            var self = this.bodyIndexPixels[bodyIndexFrame.FrameDescription.Width * (hand.Y - 1) + hand.X - 1];
                            const byte _nobody = 0xff;
                            if (opposition != _nobody && self != _nobody &&
                                opposition != self)
                            {
                                System.Console.WriteLine("{0} 勝負あり！！", DateTime.Now);
                                using (var sp = new System.Media.SoundPlayer(@"c:\Windows\Media\notify.wav"))
                                    sp.Play();
                            }
                        }
                    }

                    var bmp = OpenCvSharp.Extensions.WriteableBitmapConverter.ToWriteableBitmap(
                        result, PixelFormats.Bgr32);
                    RenderBitMap(bmp);
                    // this.WindowMat.ShowImage(result);
                    bodyindex.Dispose();
                    // depth.Dispose();
                    result.Dispose();
                }
        }

        /// <summary>
        /// Execute un-initialization tasks.
        /// </summary>
        /// <param name="sender">object sending the event.</param>
        /// <param name="e">event arguments.</param>
        private void WindowClosing(object sender, CancelEventArgs e)
        {
            if (this.speechEngine != null)
            {
                this.speechEngine.SpeechRecognized -= this.SpeechRecognized;
                this.speechEngine.RecognizeAsyncStop();
            }

            if (this.MultiFrameReader != null)
            {
                this.MultiFrameReader.MultiSourceFrameArrived -= this.MultiReader_FrameArrived;
                this.MultiFrameReader.Dispose();
                this.MultiFrameReader = null;
            }

            if (this.kinectSensor != null)
            {
                this.kinectSensor.Close();
                this.kinectSensor = null;
            }
        }

        /// <summary>
        /// Handler for recognized speech events.
        /// </summary>
        /// <param name="sender">object sending the event.</param>
        /// <param name="e">event arguments.</param>
        private void SpeechRecognized(object sender, SpeechRecognizedEventArgs e)
        {
            if (e.Result.Confidence < ConfidenceThreshold) return;

            System.Console.WriteLine("{0}, {1}", e.Result.Text, e.Result.Confidence);
            switch (e.Result.Text)
            {
                // throw InvocationTargetException to access e.Result.Semantics...
                case "よし":
                case "ようし":
                case "しょうぶあり":
                case "やめ":
                    this.game_started = false;
                    using (var sp = new System.Media.SoundPlayer(@"c:\Windows\Media\Alarm01.wav"))
                        sp.Play();
                    break;

                case "はじめ":
                    this.game_started = true;
                    using (var sp = new System.Media.SoundPlayer(@"c:\Windows\Media\chimes.wav"))
                        sp.Play();
                    break;
            }
        }

        private unsafe void ProcessDepthFrameData(IntPtr depthFrameData, uint depthFrameDataSize, ushort minDepth, ushort maxDepth)
        {
            // depth frame data is a 16 bit value
            ushort* frameData = (ushort*)depthFrameData;
            // convert depth to a visual representation
            var bpp = this.kinectSensor.DepthFrameSource.FrameDescription.BytesPerPixel;
            for (int i = 0; i < (depthFrameDataSize / bpp); ++i)
            {
                // Get the depth for this pixel
                ushort depth = frameData[i];

                // To convert to a byte, we're mapping the depth value to the byte range.
                // Values outside the reliable depth range are mapped to 0 (black).
                this.depthPixels[i] = (byte)(depth >= minDepth && depth <= maxDepth ? (depth * MapDepthToByte) : 0);
            }
        }

        /// <summary>
        /// Renders color pixels into the writeableBitmap.
        /// </summary>
        private unsafe void RenderBitMap(WriteableBitmap image)
        {
            this.ImageBinding.Source = image;
        }
    }
}
