const tf = require('@tensorflow/tfjs-node');
const spellcheck = require('nodehun-sentences');
const fs = require("fs");
const Nodehun = require('nodehun');
const hunspell = new Nodehun(
  fs.readFileSync('dictionaries/en_US.aff'),
  fs.readFileSync('dictionaries/en_US.dic')
);


// for simple_server:
const url  = require('url');
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
const BLACKLISTED_WORDS = [ "mating", "fuck", "shit", "crap"];


// Stores the location of the tfjs model for the model with the given name
// the names are in the form <DATASET>_<Epochs trained>
let modelFileNames = {
  // because of harry potter's large character set, the model initialized
  // for this dataset will be considered the initial model for all other text as well
  // in order to take up less memory when all models are loaded
  newModel: "harryPotter-0.json",

  aliceInWonderland_1:  "AiW-1.json" ,
  aliceInWonderland_5:  "AiW-5.json" ,
  aliceInWonderland_20: "AiW-20.json",

  drSeuss_1:  "dr-seuss-1.json" ,
  drSeuss_5:  "dr-seuss-5.json" ,
  drSeuss_20: "dr-seuss-20.json",

  harryPotter_1: "harryPotter-1.json",
  harryPotter_5: "harryPotter-5.json",
  harryPotter_20: "harryPotter-20.json",

  nancy_1: "nancy-1.json",
  nancy_5: "nancy-5.json",
  nancy_20: "nancy-20.json",

  narnia_1_1: "narnia-1-1.json",
  narnia_1_5: "narnia-1-5.json",
  narnia_1_20:"narnia-1-20.json",

  tomSawyer_1: "tomSawyer-1.json",
  tomSawyer_5: "tomSawyer-5.json",
  tomSawyer_20: "tomSawyer-20.json",

  wizardOfOz_1:  "WoOz-1.json" ,
  wizardOfOz_5:  "WoOz-5.json" ,
  wizardOfOz_20: "WoOz-20.json",

  nietzsche: 'nietzsche.json',
  harryPotter: 'harryPotter.json',
};

// The variable where all the model objects will be stored and used
const models = {};

// Charsets shared between models
// (e.g. for models trained on same dataset saved at different epochs)
const alice       = [
  "\n", " ", "!", "\"", "'", "(", ")", "*", ",", "-", ".", ":", ";", "?",
  "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
  "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
];
const drSeuss     = [
  "\n", " ", "!", "\"", "$", "'", "(", ")", ",", "-", ".", "/", "0", "1",
  "3", "4", "6", "8", "9", ":", ";", "?", "a", "b", "c", "d", "e", "f",
  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
  "u", "v", "w", "x", "y", "z", "‘"
];
const harryPotter = [
  " ", "!", "\"", "'", "(", ")", "*", ",", "-", ".", "0", "1", "2", "3",
  "4", "5", "6", "7", "8", "9", ":", ";", "?", "\\", "a", "b", "c", "d",
  "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
  "s", "t", "u", "v", "w", "x", "y", "z", "~", "–", "“"
];
const nancy       = [
  "\n", " ", "!", "\"", "'", ",", "-", ".", "0", "1", "4", "5", "8", ":", ";",
  "?", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
  "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "£", "—", "’"
];
const narnia_1    = [
  "\n", " ", "!", "\"", "'", "(", ")", ",", "-", ".", ":", ";", "?", "_",
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
  "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
];
const tomSawyer   = [
  "\n", " ", "!", "\"", "'", "*", ",", "-", ".", "0", "1", "2", "3", "4",
  "6", "7", "8", ":", ";", "?", "[", "]", "a", "b", "c", "d", "e", "f",
  "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
  "v", "w", "x", "y", "z"
];
const woOz        = [
  "\n", " ", "!", "\"", "'", "(", ")", ",", "-", ".", "0", "1", "2", "3",
  "4", "5", "6", "7", "8", "9", ":", ";", "?", "a", "b", "c", "d", "e",
  "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
  "t", "u", "v", "w", "x", "y", "z"
];
// Mapping of individual models with their charsets
const charSets = {
  // a newly initiated model that has not seen any text, specifically the one prepared
  // to train on Harry Potter
  new_model: harryPotter,

  aliceInWonderland_0: harryPotter,
  aliceInWonderland_1: alice,
  aliceInWonderland_5: alice,
  aliceInWonderland_20: alice,

  drSeuss_0: harryPotter,
  drSeuss_1: drSeuss,
  drSeuss_5: drSeuss,
  drSeuss_20: drSeuss,

  harryPotter_0: harryPotter,
  harryPotter_1: harryPotter,
  harryPotter_5: harryPotter,
  harryPotter_20: harryPotter,

  nancy_0: harryPotter,
  nancy_1: nancy,
  nancy_5: nancy,
  nancy_20: nancy,

  narnia_1_0: harryPotter,
  narnia_1_1: narnia_1,
  narnia_1_5: narnia_1,
  narnia_1_20:narnia_1,

  tosawyer_0: harryPotter,
  tomSawyer_1: tomSawyer,
  tomSawyer_5: tomSawyer,
  tomSawyer_20: tomSawyer,

  wizardOfOz_0: harryPotter,
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
// the models hosted on the web server all have sample length of 40
const sampleLen = 40;



/**
 * Loads every model in the modelFileNames into the global variable "models"
 */
async function setupModels () {
  console.log("\nBeginning to load all registered files.");
  for (let selectedModel in modelFileNames) {
    let ioHandler = tf.io.fileSystem(__dirname + '/models/' + modelFileNames[selectedModel]);
    let model = await tf.loadLayersModel (ioHandler);
    models[selectedModel] = model;
    console.log("    Loaded Model: " + selectedModel)
  }

  let modelNames = ["aliceInWonderland_", "drSeuss_", "harryPotter_", "nancy_", "narnia_1_"];
  for (let i = 0; i < modelNames.length; i++) {
    models[modelNames[i] + "0"] = models["newModel"];
    console.log("    Loaded Model: " + modelNames[i] + "0");
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
 * Returns true if the provided word is in the list of blacklisted words
 * false otherwise
 */
function isBlacklistedWord(word){
  word = word.toLowerCase();
  return BLACKLISTED_WORDS.indexOf(word) != -1;
}



/**
 * Function to set up the model / start the server / start listening for requests
 */
async function setUp() {
  // loads all the models
  setupModels();

  // set up the server
  const app = http.createServer(async function (request, response) {
    // TO-DO Respond to the favicon.ico request properly if necessary
    if(request.url == "/favicon.ico")
      return response.end();

    function respond (text) {
      const respJSON = { generated: text };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
    }

    // respond to query with generated text
    let q = url.parse(request.url, true).query;

    // Logs to the console the Data given in the url
    if(LOG_QUERY_INPUTS){
      console.log("==== New Request Being Handled!");
      console.log("  Data From URL:")
      console.log(q);
    }

    // Reassigns the global variable which defaults to DEFAULT_OUTPUT_LEN
    const outputLen = q.outputLength || DEFAULT_OUTPUT_LEN;

    // Assigns a default model
    let currentModel = DEFAULT_MODEL;
    if(q.model){
      if(models[q.model])
        currentModel = q.model;
      else
        console.log(`  Requested model (${q.model}) does not exist.`);
    }
    console.log(`  Using the following model: ${currentModel}`);

    // if there is no input text, return an error
    if (!(q.inputText)) {
      console.log('There was no input text provided. Throwing error...');
      return respond("There was no input text provided.");
    }

    // Handles Tests
    if (q.inputText == 'test') {
      console.log('This is a test!');
      return respond ("The test returned correctly!");
    }


    // Determines if the input is of sufficient length
    if (q.inputText.length < sampleLen)
      console.log('inputText is too short. Using previous seed.');
    else
      seedTextInput = q.inputText;

    if(LOG_SEED)
      console.log('==== Seed:\n  ' + seedTextInput);

    // This will store the model's generated text
    let generatedText = '';
    // Try to Generate the Text using the LSTM
    try {
      generatedText = await generateText(
        currentModel,
        sampleLen,
        outputLen,
        seedTextInput);
      console.log(generatedText);
    } catch (err) {
      console.log(err);
      return respond ('An error has occured when generating text.');
    }

    // If the model is not trained, don't try to autocorrect it. 
    if(q.model.slice(-2) = "_0")
      return respond (generatedText);

    // Runs a Spell Checker
    let corpus = generatedText;

    console.log(`Sending generatedText to the spell checker`);

    try {
      spellcheck(hunspell, corpus, (error, typos) => {
        // what to do if there was an error within the spellcheck function
        if(error)
          throw error;

        // otherwise we continue and make edits given the typos and suggested corrections
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
          const corrections = typos[i].suggestions;
          let correctionIndex = 0;
          for(let j = 0; j < corrections.length; j++, correctionIndex++) {
            // if found a correction that is not blacklisted, then  use it
            if(!isBlacklistedWord(corrections[correctionIndex]))
              break;
            // if reached the end, then let it be known that there was no suitable correction
            if(correctionIndex >= corrections.length) {
              console.log("Found no suitable replacement for the blacklisted correction.");
              correctionIndex = 0;
            }
          }
          corpus = corpus.slice(0,pos.from) + corrections[correctionIndex] + corpus.slice(pos.to);
        }

        if(LOG_TYPO_CORRECTION) {
          console.log("\n==== The text after getting corrected:");
          console.log(corpus);
          console.log("==== END TEXT\n");
        }

        // The responseJSON is made and returned.
        return respond (corpus);
      });
    }
    catch (err) {
      console.log(`ERROR: Failed to spell check text: ${err.message}`);
      return respond ('An error has occured when spell-checking.');
    }
  });

  // start listening for requests
  app.listen(port);
}



setUp();
