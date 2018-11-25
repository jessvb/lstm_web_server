const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const spellcheck = require('nodehun-sentences');
const fs = require("fs");
const Nodehun = require('nodehun');
const hunspell = new Nodehun(
  fs.readFileSync('dictionaries/en_US.aff'),
  fs.readFileSync('dictionaries/en_US.dic')
);


// for simple_server:
const url = require('url');
const http = require('http');
const port = 1234;

/* ============================================================= */
/* ========================== Variables ======================== */
/* ============================================================= */

const LOG_QUERY_INPUTS  = true;     // Logs all of the query stuff
const LOG_SEED          = true;     // Logs the seed that is being used
const LOG_TEXT_GEN_PROGRESS = true; // Logs the progress of generating text every 20%
const LOG_TYPO_CORRECTION   = true; // Logs the generated text before and after beign corrected for typos
const LOG_TYPOS         = false;    // Logs the typos found in the generated text

// This is the default model given in case there is no model requested by the url
const DEFAULT_MODEL = 'narnia_1_20';
const DEFAULT_OUTPUT_LEN = 100;

// Stores the location of the tfjs model for the model with the given name
// the names are in the form <DATASET>_<Epochs trained>
let modelFileNames = {
  aliceInWonderland_1:  "AiW-1.json" ,
  aliceInWonderland_5:  "AiW-5.json" ,
  aliceInWonderland_10: "AiW-10.json",
  aliceInWonderland_20: "AiW-20.json",

  drSeuss_1:  "dr-seuss-1.json" ,
  drSeuss_5:  "dr-seuss-5.json" ,
  drSeuss_10: "dr-seuss-10.json",
  drSeuss_20: "dr-seuss-20.json",

  nancy_1: "nancy-1.json",
  nancy_5: "nancy-5.json",
  nancy_20: "nancy-20.json",
  nancy_40: "nancy-40.json",

  narnia_1_1: "narnia-1-1.json",
  narnia_1_5: "narnia-1-5.json",
  narnia_1_10:"narnia-1-10.json",
  narnia_1_20:"narnia-1-20.json",

  wizardOfOz_1:  "WoOz-1.json" ,
  wizardOfOz_5:  "WoOz-5.json" ,
  wizardOfOz_10: "WoOz-10.json",
  wizardOfOz_20: "WoOz-20.json",

  nietzsche: 'nietzsche.json',
  harryPotter: 'harryPotter.json',
};
// The variable where all the model objects will be stored and used
const models = {};
// Charsets shared between models
// (e.g. for models trained on same dataset saved at different epochs)
const alice = [
  "\n", " ", "!", "\"", "'", "(", ")", "*", ",", "-", ".", ":", ";", "?",
  "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
  "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
];
const narnia_1 = [
  "\n", " ", "!", "\"", "'", "(", ")", ",", "-", ".", ":", ";", "?", "_",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
  "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
];
const nancy = [
  "\n", " ", "!", "\"", "'", ",", "-", ".", "0", "1", "4", "5", "8", ":", ";",
  "?", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
  "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "£", "—", "’"
];
const drSeuss = [
  "\n", " ", "!", "\"", "$", "'", "(", ")", ",", "-", ".", "/", "0", "1",
  "3", "4", "6", "8", "9", ":", ";", "?", "a", "b", "c", "d", "e", "f",
  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
  "u", "v", "w", "x", "y", "z", "‘"
];
const woOz = [
  "\n", " ", "!", "\"", "'", "(", ")", ",", "-", ".", "0", "1", "2", "3",
  "4", "5", "6", "7", "8", "9", ":", ";", "?", "a", "b", "c", "d", "e",
  "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
  "t", "u", "v", "w", "x", "y", "z"
];
const nancyDrew = [
  "\n", " ", "!", "\"", "'", ",", "-", ".", "0", "1", "4", "5", "8", ":",
  ";", "?", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
  "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "£", "—", "’"
];
// Mapping of individual models with their charsets
const charSets = {
  aliceInWonderland_1: alice,
  aliceInWonderland_5: alice,
  aliceInWonderland_20: alice,

  drSeuss_1: drSeuss,
  drSeuss_5: drSeuss,
  drSeuss_10: drSeuss,
  drSeuss_20: drSeuss,

  nancy_1: nancy,
  nancy_5: nancy,
  nancy_20: nancy,
  nancy_40: nancy,

  narnia_1_1: narnia_1,
  narnia_1_5: narnia_1,
  narnia_1_10:narnia_1,
  narnia_1_20:narnia_1,

  nancyDrew_1: nancyDrew,
  nancyDrew_5: nancyDrew,
  nancyDrew_20: nancyDrew,
  nancyDrew_40: nancyDrew,

  wizardOfOz_1: woOz,
  wizardOfOz_5: woOz,
  wizardOfOz_20: woOz,

  harryPotter: [
    "\n", " ", "!", "\"", "'", "(", ")", "*", ",", "-", ".",
    "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "=",
    "?", "\\", "^", "_", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "}", "~", "�", "­", "é", "ü", "–", "‘", "’", "“"
  ],
  nietzsche: [
    'P', 'R', 'E', 'F', 'A', 'C', '↵', 'S', 'U', 'O', 'I', 'N', 'G', ' ',
    't', 'h', 'a', 'T', 'r', 'u', 'i', 's', 'w', 'o', 'm', 'n', '-', 'e',
    '?', 'g', 'd', 'f', 'p', 'c', 'l', ',', 'y', 'v', 'b', 'k', ';', '!',
    '.', 'B', 'z', 'W', 'H', ':', '(', 'j', ')', '"', 'V', 'L', '\'', 'D',
    'Y', 'K', 'q', 'M', 'x', 'J', '1', '8', '5', '2', '3', '_', '4', '6',
    '7', '9', '0', 'Q', 'X', '[', ']', 'Z', 'ä', '=', 'æ', 'ë', 'é', 'Æ'
  ],
};
// Smaller Temperature means less creative and more sensible
let temperatureInput = 0.75;
// let seedTextInput = 'This is a seed input. Hopefully it works.'; TODO THIS IS LENGTH 40 VS 60 FOR NIET VS HARRYPOTTER
let seedTextInput = "This is a seed input. Hopefully it works and we'll get results.";

const sampleLen = 40;



/**
 * Loads the text generator model at the given file path and returns it
 */
async function loadModel (selectedModel) {
  // acquires model
  const path = 'file://./models/' + selectedModel
  const model = await tf.loadModel(path);
  // returns the model and its accompanying data
  return model;
}



/**
 * Loads every model in the modelFileNames into the global variable "models"
 */
async function setupModels () {
  console.log("\nBeginning to load all registered files.");
  for (let selectedModel in modelFileNames) {
    // loadModel returns {model:model,lstmLayersSizesInput:lstmLayersSizes}
    let model = await loadModel (modelFileNames[selectedModel]);
    models[selectedModel] = model;
    console.log("    Loaded Model: " + selectedModel)
  }
  console.log("Done loading all registered files.\n")
}


/* ============================================================= */
/* ============= functions from original utils.js ============= */
/* ============================================================= */


/**
 * Draw a sample based onprobabilities.
 *
 * @param {tf.Tensor} preds Predicted probabilities, as a 1D `tf.Tensor` of
 *   shape `[this._charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
 *   `tf.Tensor`.
 * @returns {number} The 0-based index for the randomly-drawn sample, in the
 *   range of [0, this._charSetSize - 1].
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

/* ============================================================= */
/* ==== functions from orig LoadableLSTMTextGenerator class ==== */
/* ============================================================= */


/**
 * Generate text using the LSTM model.
 *
 * @param {number[]} sentenceIndices Seed sentence, represented as the
 *   indices of the constituent characters.
 * @param {number} length Length of the text to generate, in number of
 *   characters.
 * @param {number} temperature Temperature parameter. Must be a number > 0.
 * @returns {string} The generated text.
 */
async function genText(currentModel, sentenceIndices, length, temperature, model, charSet, charSetSize, sampleLen) {

  generatedTextInput = '';

  console.log('Generating text...');
  console.log("    Length: " + length);

  const temperatureScalar = tf.scalar(temperature);

  let generated = '';
  while (generated.length < length) {
    // Encode the current input sequence as a one-hot Tensor.
    const inputBuffer =
      new tf.TensorBuffer([1, sampleLen, charSetSize]);
    for (let i = 0; i < sampleLen; ++i) {
      inputBuffer.set(1, 0, i, sentenceIndices[i]);
    }
    const input = inputBuffer.toTensor();

    // Call model.predict() to get the probability values of the next
    // character.
    const output = model.predict(input);

    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), temperatureScalar);
    const winnerChar = charSet[winnerIndex];
    await onTextGenerationChar(winnerChar, length);

    generated += winnerChar;
    sentenceIndices = sentenceIndices.slice(1);
    sentenceIndices.push(winnerIndex);

    input.dispose();
    output.dispose();
  }
  temperatureScalar.dispose();
  return generated;
}



/**
 * Load generator (lstm model) function
 */
async function loadGen(modelName) {

  const path = 'file://./models/' + modelFileNames[modelName]
  const model = await tf.loadModel(path);

  lstmLayersSizesInput = lstmLayerSizes(model);
  model.summary();
  return model;
};



/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
async function onTextGenerationChar(charac, len) {
  generatedTextInput += charac;
  const charCount = generatedTextInput.length;
  const generateLength = len;
  const currStatus = `Generating text: ${charCount}/${generateLength} complete...`;

  if(LOG_TEXT_GEN_PROGRESS && charCount % Math.round(generateLength/5) == 0)
    console.log(currStatus);
  //console.log('character in onTextGenerationChar: '+charac);
  if (charCount / generateLength == 1) {
    console.log(generatedTextInput);
  }
  await tf.nextFrame();
}



/**
 * Use `textGenerator` to generate random text, show the characters in the
 * console as they are generated one by one.
 * @param currentModel, the string name of the current model
 *
*/
async function generateText(currentModel, sampleLen, outputLen, seedTextInput) {
  try {
    // loads the currentModel's preloaded model
    let model = models[currentModel]
    // Loads the charset for the currentModel
    let charSet = charSets[currentModel];
    // Determines the length of the charset
    let charSetSize = charSet.length;

    // ERROR 1 - No Model
    if (model == null) {
      console.log('ERROR: Please load text data set first.');
      return;
    }

    const generateLength = outputLen;
    const temperature = temperatureInput;

    // ERROR 2 - Improper value for generateLength
    if (!(generateLength > 0)) {
      console.log(
        `ERROR: Invalid generation length: ${generateLength}. ` +
        `Generation length must be a positive number.`);
      return;
    }

    // ERROR 3 - Temperature must be positive and between (0,1]
    if (!(temperature > 0 && temperature <= 1)) {
      console.log(
        `ERROR: Invalid temperature: ${temperature}. ` +
        `Temperature must be a positive number.`);
      return;
    }

    let seedSentence;
    let seedSentenceIndices;

    // ERROR 4 - Invalid Seed Text Length
    if (seedTextInput.length === 0) {
      console.log(
        `ERROR: seed sentence length is zero. Seed Sentence: ` +
        seedTextInput.value + '.');
      return;
    } else {
      seedSentence = seedTextInput;
      if (seedSentence.length < sampleLen) {
        console.log(
          `ERROR: Seed text must have a length of at least ` +
          `${sampleLen}, but has a length of ` +
          `${seedSentence.length}.`);
        return;
      }
      seedSentence = seedSentence.slice(
        seedSentence.length - sampleLen, seedSentence.length);

      seedSentenceIndices = [];
      for (let i = 0; i < seedSentence.length; ++i) {
        seedSentenceIndices.push(
          charSets[currentModel].indexOf(seedSentence[i]));
      }
    }

    console.log("generateLength before calling genText: " + generateLength)

    const sentence = await genText(
      currentModel, seedSentenceIndices, generateLength, temperature, model, charSet, charSetSize, sampleLen);
    generatedTextInput = sentence;
    const currStatus = 'Done generating text.';
    console.log(currStatus);
    return sentence;
  } catch (err) {
    console.log(`ERROR: Failed to generate text: ${err.message}, ${err.stack}`);
  }
}



/**
 * Function to set up the model / start the server / start listening for requests
 */
async function setUp() {
  // loads all the models
  setupModels();

  // set up the server
  const app = http.createServer(async function (request, response) {
    // TO-DO
    // Respond to the favicon.ico request properly if necessary
    if(request.url == "/favicon.ico")
      return response.end();

    var q, respJSON;

    // respond to query with generated text
    q = url.parse(request.url, true).query;

    // Logs to the console the Data given in the url
    if(LOG_QUERY_INPUTS){
      console.log("New Request Being Handled!");
      console.log("Data From URL:")
      console.log(q);
      console.log("\n");
    }

    // Reassigns the global variable which defaults to DEFAULT_OUTPUT_LEN
    const outputLen = q.outputLength || DEFAULT_OUTPUT_LEN;

    // Assigns a default model
    let currentModel = DEFAULT_MODEL;

    // Uses the request model if it exists
    if(q.model) {
      if(models[q.model])
        currentModel = q.model;
      else
        console.log(`  Requested model (${q.model}) does not exist.`);
    }
    console.log(`  Using the following model: ${currentModel}`);

    // if there is no input text, return an error
    if (!(q.inputText)) {
      console.log('There was no input text provided. Throwing error...');
      respJSON = { generated: "There was no input text provided." };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;
    }








    // This will store the model's generated text
    let generatedText = '';

    // Handles Tests
    if (q.inputText == 'test') {
      console.log('This is a test!');
      generatedText = "test returned correctly! Running Tom's Code";
      respJSON = { generated: generatedText };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
      return;
    }

    // Determines if the input is of sufficient length
    if (q.inputText.length < sampleLen) {
      console.log('inputText is too short. Using previous seed.');
    } else {
      seedTextInput = q.inputText;
    }

    if(LOG_SEED) {
      console.log('  Seed: ' + seedTextInput);
    }

    // Generate text
    try {
      generatedText = await generateText(
        currentModel,
        sampleLen,
        outputLen,
        seedTextInput);
    } catch (err) {
      console.log(err);
      generatedText = "An error has occurred.";
    }


    // Runs a Spell Checker
    let corpus = generatedText;
    console.log(`Sending generatedText to the spell checker`);
    try{
      spellcheck(hunspell, corpus, (error, typos) => {

        if(error) {
          console.log(`ERROR: Failed to spellcheck text.`)
          generatedText = "An error has occurred.";

          // The responseJSON is made and returned.
          respJSON = { generated: "An error has occured." };
          response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
          response.write(JSON.stringify(respJSON));
          response.end();
        }

        if(LOG_TYPOS) {
          console.log("\n  ==== Let's see some TYPOS ");
          console.log(typos);
        }

        if(LOG_TYPO_CORRECTION) {
          console.log("\n==== The text being spellchecked:");
          console.log(corpus);
          console.log("==== END TEXT");
        }

        // Corrects the typos with the first suggestion
        for (let i = typos.length - 1; i >= 0; i--) {
          const pos = typos[i].positions[0];
          const correction = typos[i].suggestions[0];
          corpus = corpus.slice(0,pos.from) + correction + corpus.slice(pos.to);
        }

        if(LOG_TYPO_CORRECTION) {
          console.log("\n==== The text after getting corrected:");
          console.log(corpus);
          console.log("==== END TEXT\n");
        }

        // The responseJSON is made and returned.
        respJSON = { generated: corpus };
        response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
        response.write(JSON.stringify(respJSON));
        response.end();
      });
    } catch (err) {
      console.log(`ERROR: Failed to spellcheck text: ${err.message}`);

      // The responseJSON is made and returned.
      respJSON = { generated: "An error has occured when spellchecking." };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
    }
  });

  // start listening for requests
  app.listen(port);
}

setUp();
