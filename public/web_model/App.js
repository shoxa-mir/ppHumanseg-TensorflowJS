import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Function to load and run the deeplab model
  const runSegment = async () => {
    const MODEL_PATH = 'http://localhost:3000/web_model/model.json';
    const model = await tf.loadGraphModel(MODEL_PATH);
    console.log("PPHumanSeg model loaded.");
    
    // Setting interval for detection
    setInterval(() => {
      infer(model);
    }, 100);
  };

  // Function to detect objects and draw the segmentation map
  const infer = async (model) => {
    if (
      webcamRef.current &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      //Preprocessing
      const videoFrame = tf.browser.fromPixels(video);
      const resizedFrame = tf.image.resizeBilinear(videoFrame, [192, 192]);
      const PREPROCESS_DIVISOR = tf.scalar(255 / 2);
      const preprocessedInput = tf.div(tf.sub(resizedFrame.asType('float32'), PREPROCESS_DIVISOR),PREPROCESS_DIVISOR);
      const expandedInput = preprocessedInput.expandDims(0);
      const reshapedInput = expandedInput.transpose([0, 3, 1, 2]);

      // Inference
      const segmap = model.execute({["x"]: reshapedInput}, "save_infer_model/scale_0.tmp_1");

      // Postprocessing
      const squeezed = segmap.squeeze();
      const transposed = squeezed.transpose([1, 2, 0]);
      const resized = tf.image.resizeBilinear(transposed, [videoHeight, videoWidth]);
      const post = resized.transpose([2, 0, 1]).expandDims(0);
      const segm = post.argMax(1).asType('int32');
      const maxValue = segm.max();
      // maxValue.data().then(maxVal => {
      //   console.log('Max value:', maxVal[0]);
      // });
      const colorMap = tf.tensor([[0, 255, 0, 255], [255, 255, 255, 255]]);
      const segmColor = tf.gather(colorMap, segm, 0).squeeze();
      console.log(segmColor);

      // Draw the segmentation map on canvas
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, videoWidth, videoHeight);
  
      segmColor.data().then(data => {
        const height = segmColor.shape[0];
        const width = segmColor.shape[1];
        var imageData = new ImageData(new Uint8ClampedArray(data), width, height);

        ctx.putImageData(imageData, 0, 0);
        imageData = null;
    });

    tf.dispose([videoFrame, resizedFrame, preprocessedInput, expandedInput, reshapedInput, segmap, squeezed, transposed, resized, post, segm, maxValue, segmColor]);
    // console.log(tf.memory());

    }
  };

  // Run the deeplab model when component mounts
  useEffect(()=>{runSegment()},[]);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 8,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
