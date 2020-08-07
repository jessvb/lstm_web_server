
/* ============================= Server Imports ============================= */

const tf = require('@tensorflow/tfjs-node');

// for simple_server:
const url  = require('url');
const http = require('http');
const PORT = 1234;

/* ============================ Server Variables ============================ */

// Logs the progress of generating text every 20%
const LOG_TEXT_GEN_PROGRESS = true;
// The default model given in case no model requested by the url
const DEFAULT_MODEL = 'narnia_1_20';
const DEFAULT_OUTPUT_LEN = 40;
// On range (0,1] temperature of 0 means each char is chosen by max likelihood
const TEMPERATURE = tf.scalar(0.6);
let seedTextInput = "This is a seed input. Hopefully it works and we'll get results.";
// the models hosted on the web server all have sample length of 40
const SAMPLE_LEN = 40;

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
  // a newly initiated model that has not seen any text. All <data>_0 models
  // can effectively share the same charset and model
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

  tomSawyer_0: harryPotter,
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



/**
 * Loads every model in the modelFileNames into the global variable "models"
 */
async function setupModels () {
  console.log("\nBeginning to load all registered files.");
  for (let selectedModel in modelFileNames) {
    let ioHandler = tf.io.fileSystem(__dirname + '/char-based-models/' + modelFileNames[selectedModel]);
    let model = await tf.loadLayersModel (ioHandler);
    models[selectedModel] = model;
    console.log("    Loaded Model: " + selectedModel);
  }

  let modelNames = ["aliceInWonderland_", "drSeuss_", "harryPotter_", "nancy_", "narnia_1_"];
  for (let i = 0; i < modelNames.length; i++) {
    models[modelNames[i] + "0"] = models["newModel"];
    console.log("    Loaded Model: " + modelNames[i] + "0");
  }
  console.log("Done loading all registered files.\n");
}



/**
 * Pick an index from the output of a model randomly.
 *
 * @param {tf.Tensor} preds Predicted probabilities, as a 1D `tf.Tensor` of
 *   shape `[charSetSize]`.
 * @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
 *   or diversity) to use during sampling. Number be a number > 0, as a Scalar
 *   `tf.Tensor`.
 * @returns {number} The 0-based index for the sampled char, in the range of
 *   [0, this._charSetSize - 1].
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
    const subbed = tf.sub(1, preds);
    const divved = tf.div(preds, subbed);
    const unnormalized = tf.log(divved);
    return tf.multinomial(unnormalized, 1, null, false).dataSync()[0];
  });
}



/**
 * Generate text using the LSTM model.
 *
 * @param {number[]} seed_seq - The input sequence provided as a list of integers representing the chars
 * @param {number} len_generate - The target number of characters to generate
 * @param {number} temperature Temperature parameter. Must be a number > 0.
 * @returns {string} The generated text.
 */
async function genText(seed_seq, len_generate, model, charSet, sampleLen) {
  let num_chars_generated = 0;
  let generated = '';

  while (generated.length < len_generate) {

    // prints to the console progress generating the text
    if(LOG_TEXT_GEN_PROGRESS && (++num_chars_generated) % Math.round(len_generate/5) == 0)
      console.log(`Generating text: ${num_chars_generated}/${len_generate} complete...`);

    // creates the one hot vector as input to the model
    const input = tf.oneHot(seed_seq, charSet.length)
      .reshape([1, sampleLen, charSet.length]);

    // Call model.predict() to get the probability values of the next char
    const output = model.predict(input);
    // Sample randomly based on the probability values.
    const winnerIndex = sample(tf.squeeze(output), TEMPERATURE);
    const winnerChar = charSet[winnerIndex];

    generated += winnerChar;
    seed_seq = seed_seq.slice(1);
    seed_seq.push(winnerIndex);

    input.dispose();
    output.dispose();
  }

  return generated;
}



/* ============================ Start the Server ============================ */


setupModels();


const app = http.createServer (async function (request, response) {

  // favicon.ico only gets requested when doing the GET req in a browser. ignore
  if(request.url == "/favicon.ico")
    return response.end();

  function respond (text) {
    console.log("Response:", text);
    response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
    response.write(JSON.stringify({ generated: text }));
    response.end();
  }


  // ---------------------------------------------------------------------------
  // STEP 1 - Parse the Query Inputs


  // Parse the query parameters
  let q = url.parse(request.url, true).query;
  console.log("==== New Request Being Handled!\n  Data From URL:")
  console.log(q);

  // Reassigns the global variable which defaults to DEFAULT_OUTPUT_LEN
  const outputLen = q.outputLength || DEFAULT_OUTPUT_LEN;

  // Assigns a default model
  let currentModelName = q.model ? q.model : DEFAULT_MODEL;

  // if there is no input text, return an error
  if (!(q.inputText) || q.inputText.length == 0)
    return respond("There was no input text provided.");

  // Handles Tests
  if (q.inputText == 'test')
    return respond("The test returned correctly!");

  // Determines if the input is of sufficient length, otherwise uses previous
  if (q.inputText.length >= SAMPLE_LEN)
    seedTextInput = q.inputText;


  // ---------------------------------------------------------------------------
  // STEP 2 - Check for Inappropriate Parameters


  // Error 1 - No Model
  if (!models[currentModelName])
    return respond(`ERROR: LSTM model "${currentModelName}" doesn't exist`);

  // Error 2 - Invalid number of chars requested to generate
  if (outputLen <= 0)
    return respond(`ERROR: Invalid generation length: ${outputLen}. ` +
      `Generation length must be a positive number.`);

  // Error 3 - The input sequence is too short to run the model
  if (seedTextInput.length < SAMPLE_LEN)
    return respond(`ERROR: Seed text must have a length of at least ` +
      `${SAMPLE_LEN}, but has a length of ${seedTextInput.length}.`);


  // ---------------------------------------------------------------------------
  // Step 3 - Some Pre-Processing of the inputs for the model


  // loads the preloaded model with the current name
  let model = models[currentModelName]
  // Loads the charset for the currentModel
  let charSet = charSets[currentModelName];

  // Select the last n-chars of the seed text, where n is the sample length
  let seedSentence = seedTextInput;
  seedSentence = seedSentence.slice(
    seedSentence.length - SAMPLE_LEN, seedSentence.length);

  // Converts the last n-chars into a list of n integers
  let seedSentenceIndices = new Array(seedSentence.length);
  for (let i = 0; i < seedSentence.length; i++)
    seedSentenceIndices[i] = charSet.indexOf(seedSentence[i]);


  // ---------------------------------------------------------------------------
  // STEP 4 - Generate the Text


  // Try to Generate the Text using the LSTM
  try {
    let generatedText = await genText(
      seedSentenceIndices, outputLen, model, charSet, SAMPLE_LEN);
    return respond(generatedText);
  }
  catch (err) {
    console.log(err);
    return respond ('An error occured when generating text.');
  }
});

// start listening for requests
app.listen(PORT);
