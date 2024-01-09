import * as tf from "@tensorflow/tfjs";

const colors = [
  [0, 0, 0, 0],
  [255, 255, 255, 100]
];

export const drawMask = (ctx, mask, color=colors) => {
    // input is a 2d tensor with [0, 1] values representing the mask
    // ctx is the canvas context
    // color is an array of rgba values for the mask and background
    const [height, width] = mask.shape;
    // console.log("height: ", height, "width: ", width);
    const imageData = new ImageData(width, height);
    const data = imageData.data;
    
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const k = mask.dataSync()[i];
        data[j + 0] = color[k][0];
        data[j + 1] = color[k][1];
        data[j + 2] = color[k][2];
        data[j + 3] = color[k][3];
    }
    // console.log(data);
    ctx.putImageData(imageData, 0, 0);

}

export const preprocessVideoFrame = (video) => {
    const PREPROCESS_DIVISOR = tf.scalar(255 / 2);
    return tf.browser.fromPixels(video)
        .resizeBilinear([192, 192])
        .asType('float32')
        .sub(PREPROCESS_DIVISOR)
        .div(PREPROCESS_DIVISOR)
        .expandDims(0)
        .transpose([0, 3, 1, 2]);
};

export const postprocResult = (segmap, videoHeight, videoWidth) => {
    return segmap.squeeze(0)
        .transpose([1, 2, 0])
        .resizeBilinear([videoHeight, videoWidth])
        .transpose([2, 0, 1])
        .argMax(0)
        .asType('int32');
};

// export const preproc = (video) => {
//     const videoFrame = tf.browser.fromPixels(video);
//     const resizedFrame = tf.image.resizeBilinear(videoFrame, [192, 192]);
//     const PREPROCESS_DIVISOR = tf.scalar(255 / 2);
//     const preprocessedInput = tf.div(tf.sub(resizedFrame.asType('float32'), PREPROCESS_DIVISOR),PREPROCESS_DIVISOR);
//     const expandedInput = preprocessedInput.expandDims(0);
//     return expandedInput.transpose([0, 3, 1, 2]);
// }

// export const postproc = (segmap, videoHeight, videoWidth) => {
//     const squeezed = segmap.squeeze(0);
//     const transposed = squeezed.transpose([1, 2, 0]);
//     const resized = tf.image.resizeBilinear(transposed, [videoHeight, videoWidth]);
//     const post = resized.transpose([2, 0, 1]);
//     return post.argMax(0).asType('int32');
// }