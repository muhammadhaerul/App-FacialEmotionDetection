const videoElement = document.getElementById('cam_input');
const canvasElement = document.getElementById('canvas_output');
const canvasRoi = document.getElementById('canvas_roi');
const canvasCtx = canvasElement.getContext('2d');
const roiCtx = canvasRoi.getContext('2d');
const noFaceFoundElement = document.getElementById('no-face-found');

const drawingUtils = window;
const emotions = ["Angry", "Happy", "Sad", "Surprise"];
let tfliteModel;
let isModelLoaded = false;

async function loadModel() {
    try {
        tfliteModel = await tf.loadLayersModel("./static/model/uint8/model.json");
        isModelLoaded = true;
        console.log("Model loaded successfully!");
    } catch (error) {
        console.error("Failed to load the model:", error);
    }
}

async function start() {
    await loadModel();
}

function openCvReady() {
    start();

    cv['onRuntimeInitialized'] = () => {
        function onResults(results) {
            try {
                // Draw the overlays.
                canvasCtx.save();
                roiCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);
                canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
                if (results.detections.length > 0) {
                    drawingUtils.drawRectangle(
                        canvasCtx, results.detections[0].boundingBox,
                        { color: 'blue', lineWidth: 4, fillColor: '#00000000' });
                    let width = results.detections[0].boundingBox.width * canvasElement.width;
                    let height = results.detections[0].boundingBox.height * canvasElement.height;
                    let sx = results.detections[0].boundingBox.xCenter * canvasElement.width - (width / 2);
                    let sy = results.detections[0].boundingBox.yCenter * canvasElement.height - (height / 2);
                    let center = sx + (width / 2);

                    let imgData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
                    let gray_roi = cv.matFromImageData(imgData);
                    let rect = new cv.Rect(sx, sy, width, height);
                    gray_roi = gray_roi.roi(rect);

                    cv.cvtColor(gray_roi, gray_roi, cv.COLOR_RGBA2GRAY, 0);
                    cv.imshow('canvas_roi', gray_roi);
                    if (tfliteModel && isModelLoaded) {
                        const outputTensor = tf.tidy(() => {
                            // Transform the image data into Array pixels.
                            let img = tf.browser.fromPixels(canvasRoi);

                            // Resize, normalize, expand dimensions of image pixels by 0 axis.:
                            img = tf.image.resizeBilinear(img, [48, 48]);
                            img = tf.div(tf.expandDims(img, 0), 255);

                            // Predict the emotions.
                            let outputTensor = tfliteModel.predict(img).arraySync();
                            return outputTensor;
                        });
                        // Convert to array and take prediction index with highest value
                        let index = outputTensor[0].indexOf(Math.max(...outputTensor[0]));
                        console.log(index)

                        canvasCtx.font = "100px Arial";
                        canvasCtx.fillStyle = "red";
                        canvasCtx.textAlign = "center";

                        canvasCtx.fillText(emotions[index], center, sy - 10);

                        // Hide the "No face found" message
                        noFaceFoundElement.style.display = 'none';
                    } else {
                        canvasCtx.fillText("Loading the model", center, sy - 50);
                    }
                } else {
                    // No face detected
                    roiCtx.clearRect(0, 0, canvasRoi.width, canvasRoi.height);
                    noFaceFoundElement.style.display = 'block';
                }
                canvasCtx.restore();
                roiCtx.restore();
            } catch (err) {
                console.log(err.message);
            }
        }

        const faceDetection = new FaceDetection({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
            }
        });

        faceDetection.setOptions({
            selfieMode: true,
            model: 'short',
            minDetectionConfidence: 0.1
        });

        faceDetection.onResults(onResults);

        const camera = new Camera(videoElement, {
            onFrame: async () => {
                await faceDetection.send({ image: videoElement });
            },
            width: 854,
            height: 480
        });

        camera.start()
    }
}
