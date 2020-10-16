const express = require('express');
const bodyParser = require('body-parser');
const cors = require("cors");
const fs = require('fs');

const app = express();
const port = process.env.PORT || 6001;

// Tensorflow library
const tf = require('@tensorflow/tfjs');
const { response } = require('express');
const { model } = require('@tensorflow/tfjs');

app.use(cors());
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));

let has_resumes = false;
let resumes = [];

// List of resumes used in training
let Xs_A = [];
let Xs_B = [];

// List of preferred resumes for the given pairs above
let ys   = [];

// New resumes to be added to the training dataset
let new_xs_A = [];
let new_xs_B = [];

// New preferred resumes to be added to the training dataset
let new_ys   = [];

// The pair of resumes currently being examined by the human
let current_question = [];

app.post('/api/world', (req, res) => {
  console.log("req.body", req.body);

  let data = JSON.stringify(req.body);

  // If the script does not have any resumes, assume that the front end is sending those
  if (has_resumes = false) {

    // Extract all the resumes into a list
    raw_resumes = [];
    for (let i in data) { raw_resumes.push(i); }

    // Remove the first one because it is just a template  
    raw_resumes.splice(0, 1);

    // Extract the actual resumes from each of the list items
    resumes = [];
    for (let i in raw_resumes) { resumes.push(i./* Insert name of item here --------------------------------------------------------------------------------------------------------------------------------------- */) }

  } 
  else 
  {
    
    // If the first resume is preferred, expect a 0.  If the second is preferred expect a 1
    response = data.preferred;

    // Add to the new_xs and new_ys vars
    new_xs_A.append(resumes[current_question[0]]);
    new_cs_B.append(resumes[current_question[1]]);
    new_ys.append([1 - response, response]);

  }

  // Select two random resumes and ask the human about them
  let r1 = Math.random(resumes.length);
  let r2 = Math.random(resumes.length);

  // Set the current question var to these two ids
  current_question = [r1, r2];

  res.send(
    {
      "goal": "find_preferred", // also will be model_completed
      "ids": [r1, r2]
    }
  );
  
});

// Tensorflow model
const input_dim     = 100;  // amount of words in the preprocessed resumes
const embedding_dim = 64;   // dimensionality of the embedding

const min_batch_size = 8;   // Minimum amount of data points required to train the model
const max_batch_size = 32;  // Max batch size used by model

const required_loss = 0.1;  // How good the model has to be before it's considered
let model_loss = 10;        // Must he larger than the required_loss

function createEncoder() {
  const input = tf.input({shape: [input_dim]});

  const embedding = tf.layers.embedding({outputDim : embedding_dim})

  const flatten = tf.layers.flatten();

  const dense1 = tf.layers.dense({units: 2048, activation="relu"});
  const dense2 = tf.layers.dense({units: 1024, activation="relu"});

  const output = dense2.apply( dense1.apply( flatten.apply( embedding.apply( input ) ) ) );

  return [input, output]
}

function createModel() {
  let encoder1 = createEncoder();
  let encoder2 = createEncoder();

  let i1 = encoder1[0];
  let o1 = encoder1[1];

  let i2 = encoder2[0];
  let o2 = encoder2[1];

  const combiner = tf.layers.concatenate();

  const dense1 = tf.layers.dense({units: 1024, activation="relu"});
  const dense2 = tf.layers.dense({units: 256 , activation="relu"});
  const dense3 = tf.layers.dense({units: 64  , activation="relu"});
  const dense4 = tf.layers.dense({units: 2   , activation="sigmoid"});

  const output = dense4.apply( dense3.apply( dense2.apply( dense1.apply(
    combiner.apply([o1, o2])
  ) ) ) );

  const model = tf.model({inputs: [i1, i2], outputs: output});

  model.compile({optimizer: "adam", loss: "categoricalCrossentropy"})

  return model;
}

function trainModel () {

  // Look for new data points
  if (new_xs_A.length != 0) {

    Xs_A = Xs_A.concat(new_xs_A);
    Xs_B = Xs_B.concat(new_xs_B);
    ys   = ys.concat(new_ys);

    new_xs_A = [];
    new_xs_B = [];
    new_ys   = [];

  }

  // See if there are enough data points to train the model
  if (Xs_A.length >= min_batch_size) {

    // Set the batch size to the correct value
    const batch_size = Math.min(max_batch_size, Xs_A.length);

    // Train the model
    const history = await model.fit([Xs_A, Xs_B], ys, {
      epochs: 1, batchSize: batch_size
    });

    model_loss = history.history.loss;
  }

}

// Debug statement, app is running
app.listen(port, () => console.log(`Listening on port ${port}`));