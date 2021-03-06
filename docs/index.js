import * as tf from '@tensorflow/tfjs';
import {CLASSES} from './classes';
import imageURL from './waldo.jpg';
import image2URL from './waldo2.jpg';

const GOOGLE_CLOUD_STORAGE_DIR =
    'http://localhost:8000/web_model/';
const MODEL_URL =
    GOOGLE_CLOUD_STORAGE_DIR + 'tensorflowjs_model.pb';
const WEIGHTS_URL =
    GOOGLE_CLOUD_STORAGE_DIR + 'weights_manifest.json';

const IMAGE_HEIGHT = 640;
const IMAGE_WIDTH = 480;

let modelPromise;

window.onload = () => modelPromise = tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);

const button = document.getElementById('toggle');
button.onclick = () => {
  image.src = image.src.endsWith(imageURL) ? image2URL : imageURL;
};

const image = document.getElementById('image');
image.src = imageURL;

const runButton = document.getElementById('run');
runButton.onclick = async () => {
  const model = await modelPromise;
  // const pixels = tf.fromPixels(image);
  const pixels = tf.fromPixels(image).toFloat().expandDims(0);
  console.log('model loaded');
  console.time('predict1');

  const res = await model.executeAsync(pixels);
  const boxes = await res[0].data();
  const scores = await res[1].data();
  // const classes = await res[2].data();
  // const numDetect = await res[3].data();

  console.log(boxes)
  console.log(scores)
  // console.log(classes)
  // console.log(numDetect)

  const c = document.getElementById('canvas');
  const context = c.getContext('2d');
  context.drawImage(image, 0, 0);
  context.font = '10px Arial';

  scores.forEach(function(score, index) {
    if (score > 0.98) {
      const top = boxes[index];
      const left = boxes[index+1];
      const bottom = boxes[index+2];
      const right = boxes[index+3];

      const minY = top * IMAGE_HEIGHT;
      const minX = left * IMAGE_WIDTH;
      const maxY = bottom * IMAGE_HEIGHT;
      const maxX = right * IMAGE_WIDTH;

      context.beginPath();
      context.rect(minX, minY, maxX - minX, maxY - minY);
      context.lineWidth = 2;
      context.strokeStyle = 'blue';
      context.stroke();
    }
  }); 
};

