const express = require('express');
const bodyParser = require('body-parser');
const cors = require("cors");
const fs = require('fs');

const app = express();
const port = process.env.PORT || 6001;

// Tensorflow library
const tf = require('@tensorflow/tfjs');
const { response } = require('express');
const { model, train } = require('@tensorflow/tfjs');

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
    for (let i in raw_resumes.data) { 
      let listed_resume = i.data;
      let concat_resume = "";

      i = 3;
      while (i < listed_resume.length) {
        concat_resume += listed_resume[i];
        i ++;
      }

      resumes.push(concat_resume) ;
    }

    // Preprocess the data
      // Create a tokenizer for all the words
    let textTokenizer = [];
    for (let r in resumes) {

      let words = r.replace(/\s{2,}/g," ");
      words = words.split(" ");

      for (let word in words) {

        textTokenizer.push(word);

      }

    }
      // Convert the resumes (strings) into arrays of ints

    let processedResumes = [];
    let maxLength = 0;

    for (let r in resumes) {

      let convertedResume = [];

      let words = r.replace(/\s{2,}/g," ");
      words = words.split(" ");
      
      for (let word in words) {

        convertedResume.push(textTokenizer.indexOf(word));

      }

      if (convertedResume.length > maxLength) {
        maxLength = convertedResume.length;
      }

      processedResumes.push(convertedResume);

    }

      // Make sure all of these are the same length
    for (let r in processedResumes) {

      while (r.length < maxLength + 1) {

        r.push(-1);

      }

    }

      // All the resumes should now be preprocessed
    resumes = processedResumes;
    has_resumes = true;

  } 
  else 
  {
    
    // If the first resume is preferred, expect a 0.  If the second is preferred expect a 1
    response = data.preferred;

    // Add to the new_xs and new_ys vars
    new_xs_A.push(resumes[current_question[0]]);
    new_cs_B.push(resumes[current_question[1]]);
    new_ys.push([1 - response, response]);

  }

  if (model_loss > required_loss) { // Model is not yet done training

  // Select two random resumes and ask the human about them
  let r1 = Math.random(resumes.length);
  let r2 = Math.random(resumes.length);

  // Set the current question var to these two ids
  current_question = [r1, r2];

  // Send it to the front end
  res.send(
    {
      "task": "find_preferred", // also will be model_completed
      "ids": [r1, r2]
    }
  );

  } 
  else { // Model is done training

    // Get a list containing indices of all resumes
    sorted_resumes = [...Array(resumes.length).keys()];
    
    // Sort the resumes
    sorted_resumes.sort((x, y) => {

      model_prediction = compiled_model.predict(resumes[x], resumes[y]);

      if (model_prediction[0] > model_prediction[1]) {
        return 1;
      }
      return -1;

    });


    // Send it to the front end
    res.send(
      {
        "task": "model_completed",
        "resumes": sorted_resumes
      }
    );

  }
  
});

// Tensorflow model
const input_dim     = 100;  // amount of words in the preprocessed resumes
const embedding_dim = 64;   // dimensionality of the embedding

const min_batch_size = 8;   // Minimum amount of data points required to train the model
const max_batch_size = 32;  // Max batch size used by model

const required_loss = 0.1;  // How good the model has to be before it's considered
let model_loss = 10;        // Must he larger than the required_loss

const compiled_model = createModel();

while (model_loss > required_loss) {

  trainModel();

}

function createEncoder() {
  const input = tf.input({shape: [input_dim]});

  const embedding = tf.layers.embedding({inputDim: input_dim, outputDim: embedding_dim})

  const flatten = tf.layers.flatten();

  const dense1 = tf.layers.dense({units: 2048, activation: "relu"});
  const dense2 = tf.layers.dense({units: 1024, activation: "relu"});

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

  const dense1 = tf.layers.dense({units: 1024, activation: "relu"});
  const dense2 = tf.layers.dense({units: 256 , activation: "relu"});
  const dense3 = tf.layers.dense({units: 64  , activation: "relu"});
  const dense4 = tf.layers.dense({units: 2   , activation: "sigmoid"});

  const output = dense4.apply( dense3.apply( dense2.apply( dense1.apply(
    combiner.apply([o1, o2])
  ) ) ) );

  const precompiled_model = tf.model({inputs: [i1, i2], outputs: output});

  precompiled_model.compile({optimizer: "adam", loss: "categoricalCrossentropy"})

  return precompiled_model;
}

function trainModel () {

  if (!has_resumes) { return; }

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
    const history = compiled_model.fit([Xs_A, Xs_B], ys, {
      epochs: 1, batchSize: batch_size
    });

    model_loss = history.history.loss;
  }

}

// Debug statement, app is running
app.listen(port, () => console.log(`Listening on port ${port}`));