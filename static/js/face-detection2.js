// Initialize DOM selectors
let warn = document.getElementById("model_log");
let emotionText = document.getElementById("emotion");

// Initialize emotions, and tflite model.
const emotions = ["Angry", "Happy", "Sad", "Surprise"];
let tfliteModel = undefined;

// Load the (tflite) model.
// Program would await until model is loaded before continuing.
async function start() {
    await tflite.loadTFLiteModel(
        "static/model.tflite"
    ).then((loadedModel) => { 
        tfliteModel = loadedModel;
        warn.innerHTML = "Model has successfully loaded! Your camera should be displayed soon."
        // enableWebcamButton.classList.remove("invisible");
    });
}
start();

// Initialize OpenCV and webcams.
function openCvReady() {
    cv['onRuntimeInitialized'] = () => {
        let video = document.getElementById("cam_input"); // video is the id of video tag
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function (stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function (err) {
                console.log("An error occurred! " + err);
            });

        // Initialize all neccesaries mats and utils.
        let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let gray = new cv.Mat();
        let gray_roi = new cv.Mat();
        let cap = new cv.VideoCapture(cam_input);
        let faces = new cv.RectVector();
        let utils = new Utils('errorMessage');

        // Load (haas) face classifier
        let classifier = new cv.CascadeClassifier();
        let faceCascadeFile = 'haarcascade_frontalface_default.xml';
        utils.createFileFromUrl(faceCascadeFile,
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", () => {
                classifier.load(faceCascadeFile); // in the callback, load the cascade from file 
            });


        // FPS would determine how many frame per second.
        const FPS = 30;

        // Start processing and rendering the video
        function processVideo() {
            // Save webcam data
            let begin = Date.now();
            cap.read(src);
            src.copyTo(dst);
            cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
            
            // Detect face(s) using previously intialized classifier.
            try {
                classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
            } catch (err) {
                console.log(err);
            }
            
            // For each faces detected, predict the emotions
            for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                
                // Rect for gray roi
                let rect = new cv.Rect(face.x, face.y, face.width, face.height);
                
                // Mat for resize gray roi
                gray_roi = gray.roi(rect);
                gray_roi_resize = new cv.Mat();
                cv.resize(gray_roi, gray_roi_resize, new cv.Size(244, 244))
                cv.imshow("canvas_roi", gray_roi_resize);
                
                // Predict using model
                const outputTensor = tf.tidy(() => {
                    // Transform the image data into Array pixels.
                    let img = tf.browser.fromPixels(document.getElementById("canvas_roi"));

                    // Resize, normalize, expand dimensions of image pixels by 0 axis.:
                    img = tf.image.resizeBilinear(img, [48, 48]);
                    img = tf.div(tf.expandDims(img, 0), 255);
                    
                    // Predict the emotions.
                    let outputTensor = tfliteModel.predict(img);
                    return outputTensor;
                });

                // Convert to array and take prediction index with highest value
                let output = outputTensor.arraySync();
                let index = output[0].indexOf(Math.max(...output[0]));
                
                // Render rectangles and text
                cv.rectangle(dst, point1, point2, [255, 0, 0, 255], 2);
                cv.putText(dst, emotions[index], new cv.Point(face.x, face.y),
                cv.FONT_HERSHEY_SIMPLEX, 1, new cv.Scalar(0, 0, 255), 2);
                cv.imshow("canvas_output", dst);
            }
            // Schedule next frame.
            let delay = 1000 / FPS - (Date.now() - begin);
            setTimeout(processVideo, delay);
        }
        // Schedule first frame
        setTimeout(processVideo, 0);
    };
}
