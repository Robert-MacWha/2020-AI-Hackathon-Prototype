const express = require('express');
const bodyParser = require('body-parser');
const cors = require("cors");
const fs = require('fs');
const app = express();
const port = process.env.PORT || 6001;

app.use(cors());
app.use(bodyParser.json({limit: '50mb'}));
app.use(bodyParser.urlencoded({limit: '50mb', extended: true}));

// New resumes to be added to the training dataset
let new_xs_A = [];
let new_cs_B = [];
let resumes = [];
// The pair of resumes currently being examined by the human
let current_question = [];
let new_ys = [];
let model_loss = 10;        // Must he larger than the required_loss
const required_loss = 0.1;  // How good the model has to be before it's considered

app.post('/api/world', cors(), (req, res) => {
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
    res.json(
      {
        task: "find_preferred", // also will be model_completed
        ids: [r1, r2]
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
      res.json(
        {
          task: "model_completed",
          resumes: sorted_resumes
        }
      );
    }
    
  });

  // Debug statement, app is running
  app.listen(6001, () => console.log(`Listening on port 6001`));