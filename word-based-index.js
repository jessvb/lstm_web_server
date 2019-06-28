
/**
 * TO-DO: Allow for multiple requests to be processed in parallel in preparation for upscaling.
 *
 * Run the command `node word-index.js` to start this server, which hosts LSTM text generation
 * models. These text generation models generate words at a time rather than characters, which
 * is what the previous models have been.
 *
 * For a speed comparison, the original character-based narnia_20 model took about 31,000 ms to gen
 * a string of 400 characters. In contrast, this new word-based model takes around 25,000 ms to
 * gen a string of about 420 characters.
 *
 * Next, I will try a new approach to character-based text generation which Hal suggested I look
 * into called "Unsmoothed Maximum Likelihood Character Level Language Model" in the following link:
 *
 * https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139?utm_content=bufferefcf2&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer
 */

console.log("\n\n");

const tf = require('@tensorflow/tfjs-node');
const spellcheck = require('nodehun-sentences');
const fs = require("fs");

// for simple_server:
const url  = require('url');
const http = require('http');
const port = 1234;

const LOG_QUERY_INPUTS  = true;     // Logs all of the query stuff
const LOG_TEXT_GEN_PROGRESS = true; // Logs the progress of generating text every 20%
const LOG_WORD_TOKENIZING = false;  // Logs the number that each word is mapped to

const DEFAULT_MODEL = 'newModel';
const DEFAULT_OUTPUT_LEN = 10;

const modelNamePrefixes = ["aliceInWonderland_", "drSeuss_", "harryPotter_", "nancy_", "narnia_"]


// will get filled by the tensorflowjs model instances.
const models = {};

// a dictionary pairing the model names to model file name
const modelFileNames = {
  newModel: "narnia-0.json",

  narnia_25: "narnia-25.json",
  narnia_100: "narnia-100.json",
  narnia_500: "narnia-500.json",
}



const charsets = JSON.parse(fs.readFileSync('word-based-charsets.json'));



/**
 * encode - Given a list of words, it attempts to encode the words based on the tokenizer
 * It will default to a 0 if the word does not already exist.
 *
 * @param  {String}   inputText   The user's input text
 * @param  {String[]} charset     The character set
 * @return {Number[]}             An array of length 50, where the last n numbers are the n words in inputText.
 */
function encode (inputText, charset) {

  console.log(`- - Beginning to encode the text: ${inputText}`);
  // A regex expression that will apply some filter to each of these punctuation marks
  const filteredPunctuation = /!|"|#|\$|&|\\|'|\(|\)|\*|\/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|}|~/g;

  inputText = inputText
    // Remove the filtered punctuation
    .replace(filteredPunctuation, '')
    // Treat the periods and commas like separate words
    .replace(/,/g, ' , ')
    .replace(/\./g, ' . ')
    .toLowerCase()
    // Split into words
    .split(' ')
    // Remove empty strings from the inputText
    .filter((e) => e.length > 0)
    // Removes trailing white spaces
    .map((word) => word.trim());

  let encoded = (new Array (50)).fill(0);
  for (let w of inputText) {
    if (LOG_WORD_TOKENIZING) {
      console.log(`The word, ${w}, mapped to the number ${charset.indexOf(w)+1}`);
    }
    encoded.push(charset.indexOf(w) + 1);
  }
  return encoded.slice(-50);
}



/**
 * Loads every model in the modelFileNames into the global variable "models"
 */
async function loadModels () {
  console.log("\nBeginning to load all registered files.");

  // Loads all of the files listed in modelFileNames
  for (let selectedModel in modelFileNames) {
    let ioHandler = tf.io.fileSystem(__dirname + '/word-based-models/' + modelFileNames[selectedModel]);
    // loadModel returns {model:model,lstmLayersSizesInput:lstmLayersSizes}
    let model = await tf.loadLayersModel (ioHandler);
    models[selectedModel] = model;
    console.log("    Loaded Model: " + selectedModel);
  }

  // In order to save memory, all models with 0 epochs will point to the same model.
  for (let name of modelNamePrefixes) {
    models[name + "0"] = models["newModel"];
    console.log("    Loaded Model: " + name + "0");
  }
  console.log("Done loading all registered files. \n");
}



/**
 * sample - Given the model's prediction, which is a normalized distribution of
 * probabilities, this will return a number corresponding to one more than the index
 * of the word in the model's respective charset.
 *
 * @param  {tf.Tensor} preds       The output tensor predicted by the model
 * @param  {tf.Scalar} temperature The closer to 1, the more the text resemble the orig dataset
 * @return {Number}                One more than the index of the word in the model's respective charset
 */
function sample(preds, temperature) {
  return tf.tidy(() => {
    const logPreds = tf.div(tf.log(preds), temperature);
    const expPreds = tf.exp(logPreds);
    const sumExpPreds = tf.sum(expPreds);
    preds = tf.div(expPreds, sumExpPreds);
    // Could not do the mutinomial below because tfjs-node doesn't support it
    // "Treat preds a the probabilites of a multinomial distribution and
    // randomly draw a sample from the distribution."
    // return tf.multinomial(preds, 1, null, true).dataSync()[0];
    // Instead of multinomial, find logits (which are (-inf,inf) instead of [0,1])
    // by doing L = ln(preds/(1-preds))
    const subbed = tf.sub(1, preds).dataSync();
    const divved = tf.div(preds, subbed).dataSync();
    const unnormalized = tf.log(divved).dataSync();
    return tf.multinomial(unnormalized, 1, null, false).dataSync()[0];
  });
}



/**
 * generateText - Returns the text generated by the given model using
 *
 * @param  {tf.LayersModel} model        The model being used to generate text.
 * @param  {String[]}       charset      The charset for the model above
 * @param  {String}         corpus       The raw text that was given by the query
 * @param  {Number}         outputLength The number of words to generate from the model
 * @return {String}                      The result of the model's text generation
 */
async function generateText (model, charset, corpus, outputLength) {
  console.log("Beginning to ")
  // We first encode the corpus to get an array of numbers
  let e = encode(corpus, charset);
  let generatedText = "";

  for (let i = 0; i < outputLength; i++) {
    if (LOG_TEXT_GEN_PROGRESS && i % Math.floor(outputLength / 10) == 0) {
      process.stdout.write(`.`);
    }
    // We convert this encoding into tensor form that the model can read.
    let input = tf.tensor(e, [1,50]);
    // Next we get the prediction from the model
    let output = await model.predict(input);
    // we sample the output to get the next most likely word.
    let winnerIndex = sample(tf.squeeze(output), tf.scalar(1));
    let winnerWord = charset[winnerIndex - 1];
    if (winnerWord != ',' && winnerWord != '.')
      generatedText += ' '
    generatedText += winnerWord;
    // updates the feed
    e = e.slice(1);
    e.push(winnerIndex);
  }
  process.stdout.write(`. Done`);
  console.log('\n');

  return corpus + generatedText;
}



/**
 * Function to set up the model / start the server / start listening for requests
 */
async function setup () {

  loadModels();

  const app = http.createServer ( async function (request, response) {
    // TODO Respond to favicon.ico request properly probably
    if (request.url == '/favicon.ico')
      return response.end();
    console.log("\n==== New Request Being Handled");
    let respJSON;

    // Create a dictionary with the query data listed after the "?" in the url
    let q = url.parse (request.url, true).query;
    if (LOG_QUERY_INPUTS) {
      console.log("Data from URL: ");
      console.log(q);
    }

    // Reassigns the global variable which defaults to Default_OUTPUT_LEN
    const outputLen = q.outputLength || DEFAULT_OUTPUT_LEN;

    // Gets the name of the model to use
    let modelName = DEFAULT_MODEL;
    if (q.model) {
      if (models[q.model])
        modelName = q.model;
      else
        console.log(`  RequestedModel (${q.model}) does not exist.`);
    }
    let currentModel = models[modelName];

    console.log(`  Using the following model: '${modelName}'`);

    // if there is no input text, return an error
    // TODO, put an error code instead of a normal response with the error message
    if (!(q.inputText)) {
      console.log('There was no input text provided. Throwing error...');
      respJSON = { generated: "There was no input text provided." };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;
    }

    // Handles a simple test
    if (q.inputText == 'test') {
      console.log('This is a test!');
      respJSON = { generated: "The test returned correctly!" };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;
    }

    // split the seed text into a list of words
    let seed = q.inputText.split(' ');

    // TODO Check for inputlength
    try {

      // .. generate here
      let corpus = q.inputText;

      let charset = charsets[modelName.slice(0, modelName.indexOf('_'))]

      let result = await generateText (currentModel, charset, corpus, outputLen);

      respJSON = { generated: result };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;

    } catch (error) {
      console.log(error);
      // TODO Provide a propper errorCode in response
      respJSON = { generated: "An error has occured when generating text." };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;
    }
  });

  app.listen(port);
}


setup();
