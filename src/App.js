import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
import { drawMask, preprocessVideoFrame, postprocResult, postproc, preproc } from "./utilities";

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
      const deviceWidth = window.innerWidth;
      const deviceHeight = window.innerHeight;
      const video = webcamRef.current.video;
      const videoWidth = Math.min(webcamRef.current.video.videoWidth, deviceWidth);
      const videoHeight = Math.min(webcamRef.current.video.videoHeight, deviceHeight);

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      
      //Preprocessing
      const input = preprocessVideoFrame(video);

      // Inference
      const segmap = model.execute({["x"]: input}, "save_infer_model/scale_0.tmp_1");

      // Postprocessing
      const segm = postprocResult(segmap, videoHeight, videoWidth);
      // const segm = postproc(segmap, videoHeight, videoWidth);

      // const colorMap = tf.tensor([[0, 0, 0, 255], [255, 255, 255, 100]]);
      // const segmColor = tf.gather(colorMap, segm, 0).squeeze();
      // console.log(segmColor)

      // Draw the segmentation map on canvas
      const ctx = canvasRef.current.getContext("2d");
      drawMask(ctx, segm);
  
    //   segmColor.data().then(data => {
    //     const height = segmColor.shape[0];
    //     const width = segmColor.shape[1];
    //     console.log(new Uint8ClampedArray(data));
    //     var imageData = new ImageData(new Uint8ClampedArray(data), width, height);

    //     ctx.putImageData(imageData, 0, 0);
    //     imageData = null;
    // });

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
