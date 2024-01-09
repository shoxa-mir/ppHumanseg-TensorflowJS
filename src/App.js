import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
import { drawMask, preprocessVideoFrame, postprocResult } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  // Function to load and run the PPHumanSeg model
  const runSegment = async () => {
    const MODEL_PATH = 'https://192.168.1.30:3000/web_model/model.json';
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
      // console.log("deviceWidth: ", deviceWidth, "deviceHeight: ", deviceHeight);
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

      // Draw the segmentation map on canvas
      const ctx = canvasRef.current.getContext("2d");
      drawMask(ctx, segm);
      
      // Clean up
      tf.dispose(input, segmap, segm);
      
    }
  };

  // Run the segmentation model when component mounts
  useEffect(()=>{runSegment()},[]);

  if (window.innerWidth < window.innerHeight) {
    // Portrait ratio is 9:16
    const w = window.innerHeight/16*9;
    const h = window.innerHeight;
  }
  else {
    // Landscape ratio is 16:9
    const h = window.innerWidth/9*16;
    const w = window.innerWidth;
  }


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
            height: 640,
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
            zIndex: 10,
            height: 600,
          }}
        />
      </header>
    </div>
  );
}

export default App;
