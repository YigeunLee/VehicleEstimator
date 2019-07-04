
    public class VehicleSpeedEstimator {

        OpenCvSharp.ML.SVM classifier; // for hog
        HOGDescriptor hog;
        Rect prev_rect;
        
        Mat prev_img = new Mat(1080 / 2,1920 /2 ,MatType.CV_8UC1);
        Mat curr_img_resize = new Mat(1080 / 2, 1920 / 2, MatType.CV_8UC1);
        AnomalySpeedDetector anomaly;
        readonly int between_frame = 6; // per 5frame
        readonly int STAT_OK = 1;

        /// <summary>

        /// </summary>
        public event Action OnSurging;
        /// <summary>

        /// </summary>
        public int SurgingCount {
            get;set;
        }
        /// <summary>

        /// </summary>
        public int SurgingCountLimit {
            get;
            set;
        }
        /// <summary>

        /// </summary>
        public float SurgingSpeedMin {
            get;
            set;
        }
        /// <summary>
        /// 에러레이트보다 이하이면 피쳐의 속도값을 계산합니다.
        /// </summary>
        public float AllowErrorRate {
            get;
            set;
        }
        /// <summary>
        /// 추출된 피쳐의 위치들입니다.
        /// </summary>
        public Point2f[] FeaturePoints {
            get;
            private set;
        }
        /// <summary>
        /// Hog로 추출된 위치입니다.
        /// </summary>
        public Rect DetectedRect {
            get;
            private set;
        }
        /// <summary>
        /// for training
        /// </summary>
        public VehicleSpeedEstimator(System.Drawing.Size hog_training_img_size) {
            anomaly = new AnomalySpeedDetector();
              hog = new HOGDescriptor(new OpenCvSharp.Size(hog_training_img_size.Width, hog_training_img_size.Height), new OpenCvSharp.Size(16, 16), 
                new OpenCvSharp.Size(8, 8), new OpenCvSharp.Size(8, 8));
            classifier = OpenCvSharp.ML.SVM.Create();
            AllowErrorRate = 1.5f;
            SurgingSpeedMin = 0.45f;
            SurgingCountLimit = 7;
        }
        /// <summary>
        /// for estimation
        /// </summary>
        /// <param name="svm_path"></param>
        public VehicleSpeedEstimator(System.Drawing.Size hog_training_img_size,string svm_path) {
            try {
                anomaly = new AnomalySpeedDetector();
                AllowErrorRate = 1.5f;
                SurgingSpeedMin = 0.45f;
                SurgingCountLimit = 7;
                classifier = OpenCvSharp.ML.SVM.Load(svm_path);
                hog = new HOGDescriptor(new OpenCvSharp.Size(hog_training_img_size.Width, hog_training_img_size.Height), new OpenCvSharp.Size(16, 16),
              new OpenCvSharp.Size(8, 8), new OpenCvSharp.Size(8, 8));
                var support_vectors = classifier.GetSupportVectors();
                var vectors = new float[support_vectors.Width * support_vectors.Height];
                Marshal.Copy(support_vectors.Data, vectors, 0, support_vectors.Width * support_vectors.Height);
                hog.SetSVMDetector(vectors);

            }
            catch(Exception) {
                classifier = OpenCvSharp.ML.SVM.Create();
            }
        }
        /// <summary>
        /// for hog training
        /// </summary>
        /// <param name="positive_path"></param>
        /// <param name="negative_path"></param>
        /// <param name="svm_save_path"></param>
        /// <returns></returns>
        public bool Training(string root_path, string svm_save_path) {
            try {
                var computes = new List<float[]>();
                var response = new List<int>();
                var color_folders = Directory.GetDirectories(root_path);

                if(color_folders.Length == 0)
                    return false;

                foreach(var color in color_folders) {
                    var positive_path = string.Format(color + "\\" + "positive");
                    var negative_path = string.Format(color + "\\" + "negative");
                    var positives = Directory.GetFiles(positive_path);
                    var negatives = Directory.GetFiles(negative_path);

                    foreach(var file in positives) {
                        computes.Add(hog.Compute(Cv2.ImRead(file, ImreadModes.GrayScale).Resize(hog.WinSize)));
                        response.Add(1);
                    }

                    foreach(var file in negatives) {
                        computes.Add(hog.Compute(Cv2.ImRead(file, ImreadModes.GrayScale).Resize(hog.WinSize)));
                        response.Add(-1);
                    }
                }
                var respon_mat = new Mat(response.Count, 1, MatType.CV_32SC1, response.ToArray());
                var samples = new float[computes[0].Length * computes.Count];
                var sample_mat = new Mat(computes.Count, computes[0].Length, MatType.CV_32FC1, samples);

                for(int idx = 0; idx < computes[0].Length * computes.Count; idx += computes[0].Length) {
                    for(int col = 0; col < computes[0].Length; col++) {
                        samples[idx + col] = computes[idx / computes[0].Length][col];
                    }
                }

                classifier.KernelType = OpenCvSharp.ML.SVM.KernelTypes.Linear;
                classifier.Type = OpenCvSharp.ML.SVM.Types.EpsSvr;
                classifier.P = 0.1;
                classifier.Nu = 0.1;
                classifier.C = 0.01;
                classifier.Gamma = 0.1;
                classifier.Train(sample_mat, OpenCvSharp.ML.SampleTypes.RowSample, respon_mat);
                classifier.Save(svm_save_path);

                var support_vectors = classifier.GetSupportVectors();
                var vectors = new float[support_vectors.Width * support_vectors.Height];
                Marshal.Copy(support_vectors.Data, vectors, 0, support_vectors.Width * support_vectors.Height);
                hog.SetSVMDetector(vectors);
                return true;
            }
            catch(Exception e) {
                Trace.WriteLine(e.StackTrace + e.Message);
                return false;
            }
        }

        /// <summary>
        /// for hog training
        /// </summary>
        /// <param name="positive_path"></param>
        /// <param name="negative_path"></param>
        /// <param name="svm_save_path"></param>
        /// <returns></returns>
        public bool Training(string positive_path,string negative_path,string svm_save_path) {
            var positives = Directory.GetFiles(positive_path);
            var negatives = Directory.GetFiles(negative_path);
            var computes = new List<float[]>();
            var response = new List<int>();

            foreach(var file in positives) {
                computes.Add(hog.Compute(Cv2.ImRead(file, ImreadModes.GrayScale).Resize(hog.WinSize)));
                response.Add(1);
            }

            foreach(var file in negatives) {
                computes.Add(hog.Compute(Cv2.ImRead(file, ImreadModes.GrayScale).Resize(hog.WinSize)));
                response.Add(-1);
            }
            var respon_mat = new Mat(response.Count, 1, MatType.CV_32SC1, response.ToArray());
            var samples = new float[computes[0].Length * computes.Count];

            for(int idx = 0; idx < computes[0].Length * computes.Count; idx += computes[0].Length) {
                for(int col = 0; col < computes[0].Length; col++) {
                    samples[idx + col] = computes[idx / computes[0].Length][col];
                }
            }

            var sample_mat = new Mat(computes.Count, computes[0].Length, MatType.CV_32FC1, samples);
          
            classifier.KernelType = OpenCvSharp.ML.SVM.KernelTypes.Linear;
            classifier.Type = OpenCvSharp.ML.SVM.Types.EpsSvr;
            classifier.P = 0.1;
            classifier.Nu = 0.1;
            classifier.C = 0.01;
            classifier.Gamma = 0.1;
            classifier.Train(sample_mat, OpenCvSharp.ML.SampleTypes.RowSample, respon_mat);
            classifier.Save(svm_save_path);
            var support_vectors = classifier.GetSupportVectors();
            var vectors = new float[support_vectors.Width * support_vectors.Height];
            Marshal.Copy(support_vectors.Data, vectors, 0, support_vectors.Width * support_vectors.Height);
            hog.SetSVMDetector(vectors);
            return true;
        }
        public bool IsInitialize = false;
        public void Init(Mat img,int corners = 500) {

            if(!IsInitialize) {
                Cv2.PyrDown(img, prev_img);
                prev_rect = NMS(hog.Detect(prev_img));
                DetectedRect = new Rect(prev_rect.X - ( prev_rect.Width / 2 ), prev_rect.Y - ( prev_rect.Height / 2 ), prev_rect.Width, prev_rect.Height);
                var mask = new Mat(prev_img.Size(), MatType.CV_8UC1);

                for(int y = DetectedRect.Y; y < DetectedRect.Y + DetectedRect.Height; y++) {
                    for(int x = DetectedRect.X; x < DetectedRect.X + DetectedRect.Width; x++) {
                        mask.Set<byte>(y, x, 1);
                    }
                }

                FeaturePoints = Cv2.GoodFeaturesToTrack(prev_img, corners, 0.01, 15, mask, 3, false, 0.04);
                IsInitialize = true;
            }
        }
        float prev_speed;
        /// <summary>
        /// use 1920x1080
        /// </summary>
        /// <param name="curr_img"></param>
        /// <returns></returns>
        public float GetSpeed(Mat curr_img) {
            Point2f[] curr_feature_pts = new Point2f[100];
            byte[] status;
            float[] err;

            Cv2.PyrDown(curr_img, curr_img_resize, curr_img_resize.Size());
            var feature_mask = NMS(hog.Detect(curr_img_resize), 0.8f);

            if(feature_mask == Rect.Empty)
                return -1.0f;

            feature_mask = new Rect(feature_mask.X - ( feature_mask.Width / 2 ), feature_mask.Y - ( feature_mask.Height / 2 ), feature_mask.Width, feature_mask.Height);

            Cv2.CalcOpticalFlowPyrLK(prev_img, curr_img_resize, FeaturePoints, ref curr_feature_pts, out status, out err);
            var speed = CalcSpeed(FeaturePoints, curr_feature_pts, feature_mask, status, err, between_frame);
            var acc = Math.Abs(speed - prev_speed);
            FeaturePoints = curr_feature_pts;
            curr_img_resize.CopyTo(prev_img);
            DetectedRect = feature_mask;
            //var result = anomaly.DataRecorded(speed, 0.03f * 1);
            //Trace.WriteLine(msg); // test
            var estimation = new SpeedEstimation();
            estimation.speed = speed;
            estimation.accelation = acc;
            estimation.result = 1;
            prev_speed = speed;
            return estimation.speed;

        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="prev_loc">optical feature 1</param>
        /// <param name="curr_loc">optical feature 2</param>
        /// <param name="between_frame">5번 카운팅 때 하나씩 받는것 = 6fps 와 같음 30 / 6 = 5 </param>
        /// <returns></returns>
        private float CalcSpeed(Point2f[] prev_loc, Point2f[] curr_loc,Rect feature_mask,byte[] status,float[] err, int between_frame = 2) {
            double distance_sum = 0.0f;
            int ok_len = 0;
            for(int idx = 0; idx < curr_loc.Length; idx++) {
                if(feature_mask.X < curr_loc[idx].X && feature_mask.Y < curr_loc[idx].Y &&
                        ( feature_mask.Width + feature_mask.X ) > curr_loc[idx].X && ( feature_mask.Height + feature_mask.Y ) > curr_loc[idx].Y && status[idx] == STAT_OK && err[idx] <= AllowErrorRate) {
                    var distance = curr_loc[idx].DistanceTo(prev_loc[idx]);
                    distance_sum += distance;
                    ok_len++;
                }
            }
            var speed = (float)( ( ( distance_sum / ok_len ) / ( 1.0f / between_frame ) ) / 3.6f );

            if(speed <= SurgingSpeedMin && ok_len > 0) {
                SurgingCount++; 
            }
            else if(ok_len == 0) {
                return -1.0f; // 비정상인 경우..
            }

            if(SurgingCount >= SurgingCountLimit) {
                if(OnSurging != null)
                    OnSurging();

                InitializeSurgingCount();
            }

            return speed;
        }
        public void InitializeSurgingCount() {
            SurgingCount = 0;
        }
        /// <summary>
        /// use non maxima sup
        /// </summary>
        /// <param name="rects"></param>
        /// <returns></returns>
        private Rect NMS(Point[] rects, float threshold = 0.7f) {
            if(rects == null || rects.Length == 0)
                return Rect.Empty;

            var convert_rect = new Rect[rects.Length];

            for(int idx = 0; idx < rects.Length; idx++) {
                convert_rect[idx] = new Rect(rects[idx].X + (hog.WinSize.Width / 2), rects[idx].Y + (hog.WinSize.Height / 2), hog.WinSize.Width, hog.WinSize.Height);
            }
            var result = NMS(convert_rect, threshold);
            return result;
        }
        /// <summary>
        /// use non maxima sup
        /// </summary>
        /// <param name="rects"></param>
        /// <returns></returns>
        private Rect NMS(Rect[] rects, float threshold = 0.7f) {
            var rect_ranges = new List<Rect>(rects);
            var rect_result = new List<Rect>();
            var keep_result = new bool[rects.Length];

            for(int idx = 0; idx < rects.Length - 1; idx++) {
                var order = rect_ranges.GetRange(idx + 1, ( rects.Length - idx ) - 1);
                var iou = Get_iou(order.ToArray(), rects[idx]);
                for(int col = 0; col < iou.Count; col++) {
                    if(iou[col] > threshold) {
                        /* get rect */
                        keep_result[idx + col] = false;
                    }
                    else {
                        keep_result[idx + col] = true;
                        rect_result.Add(rects[idx]);
                    }
                }
            }

            for(int idx = 0; idx < keep_result.Length; idx++) {
                if(keep_result[idx]) {
                    return rects[idx];
                }
            }

            return Rect.Empty;

        }
        private List<double> Get_iou(Rect[] rects, Rect box) {
            var rect_len = rects.Length;
            var xx1 = new List<double>();
            var yy1 = new List<double>();
            var union = new List<double>();

            for(int idx = 0; idx < rect_len; idx++) {
                xx1.Add(Math.Max(Math.Min(rects[idx].X + 0.5 * rects[idx].Width, box.X + 0.5 * box.Width) - Math.Max(rects[idx].X - 0.5 * rects[idx].Width, box.X - 0.5 * box.Width), 0));
                yy1.Add(Math.Max(Math.Min(rects[idx].Y + 0.5 * rects[idx].Height, box.Y + 0.5 * box.Height) - Math.Max(rects[idx].Y - 0.5 * rects[idx].Height, box.Y - 0.5 * box.Height), 0));
            }

            for(int idx = 0; idx < rect_len; idx++) {
                //inter.Add(xx1[idx] * yy1[idx]);
                var inter = (double)( xx1[idx] * yy1[idx] );
                union.Add(inter / (double)( ( rects[idx].Width * rects[idx].Height ) + ( box.Width * box.Height ) - inter ));
            }

            return union;
        }
    }
