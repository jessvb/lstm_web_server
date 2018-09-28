const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
// for simple_server:
const url = require('url');
const http = require('http');
const port = 1234;

// variables
const charSets = {
  drSeuss: [
    'T', 'h', 'e', ' ', 'C', 'a', 't', 'i', 'n', 'H', '↵', 'B', 'y',
    'D', 'r', '.', 'S', 'u', 's', 'd', 'o', 'I', 'w', 'p', 'l', 'A',
    'c', ',', 'W', '"', 'm', 'g', '!', 'b', 'k', 'N', 'U', 'M', 'P',
    'j', '?', 'v', 'L', 'f', 'Y', 'O', 'F', '-', 'x', 'X', '\'', 'E',
    'G', 'K', 'q', 'J', 'R', 'V', 'z', 'Q', '<94>', ':', 'â', '<80>', '<99>',
    ';', 'Z', '(', '9', '8', '3', '/', '4', ')', '“', '’', '…', '”',
    '‘', '—', '1', '0', '$', ' ', '­', ';', '6'
  ],
  nietzsche: [
    'P', 'R', 'E', 'F', 'A', 'C', '↵', 'S', 'U', 'O', 'I', 'N', 'G', ' ',
    't', 'h', 'a', 'T', 'r', 'u', 'i', 's', 'w', 'o', 'm', 'n', '-', 'e',
    '?', 'g', 'd', 'f', 'p', 'c', 'l', ',', 'y', 'v', 'b', 'k', ';', '!',
    '.', 'B', 'z', 'W', 'H', ':', '(', 'j', ')', '"', 'V', 'L', '\'', 'D',
    'Y', 'K', 'q', 'M', 'x', 'J', '1', '8', '5', '2', '3', '_', '4', '6',
    '7', '9', '0', 'Q', 'X', '[', ']', 'Z', 'ä', '=', 'æ', 'ë', 'é', 'Æ'
  ]
};

//let selectedText = 'drSeuss';
let selectedText = 'nietzsche';

// VARIABLES IN LOADABLE CLASS CONSTRUCTOR
const charSet = charSets[selectedText];
const charSetSize = charSet.length;
const modelType = 'LSTM';

// continue regular variables
let modelFileNames = {
  drSeuss: 'drSeuss.json',
  nietzsche: 'nietzsche.json'
};

let lstmLayersSizesInput = 128;
let examplesPerEpochInput = 2048;
let batchSizeInput = 128;
let epochsInput = 5;
let validationSplitInput = 0.0625;
let learningRateInput = 1e-2;
let generateLengthInput = 100;
let temperatureInput = 0.75;
let seedTextInput = 'This is a seed input. Hopefully it works.';
let generatedTextInput = '';

const sampleLen = 40;
const sampleStep = 3;

//////////////////////////////////////
// functions from original utils.js //
//////////////////////////////////////
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
	const subbed =tf.sub(1,preds).dataSync();
	const divved = tf.div(preds,subbed).dataSync();
	const unnormalized = tf.log(divved).dataSync();
	return tf.multinomial(unnormalized, 1, null, false).dataSync()[0];
  });
}
//////////////////////////////////////
// functions from original utils.js //
//////////////////////////////////////


/////////////////////////////////////////////////////////////
// functions from original LoadableLSTMTextGenerator class //
/////////////////////////////////////////////////////////////

 /**
   * Get a representation of the sizes of the LSTM layers in the model.
   *
   * @returns {number | number[]} The sizes (i.e., number of units) of the
   *   LSTM layers that the model contains. If there is only one LSTM layer, a
   *   single number is returned; else, an Array of numbers is returned.
   */
function lstmLayerSizes(model) {
    if (model == null) {
      throw new Error('Load model first.');
    }
    const numLSTMLayers = model.layers.length - 1;
    const layerSizes = [];
    for (let i = 0; i < numLSTMLayers; ++i) {
      layerSizes.push(model.layers[i].units);
    }
    return layerSizes.length === 1 ? layerSizes[0] : layerSizes;
}

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
async function genText(sentenceIndices, length, temperature, model, charSet, charSetSize, sampleLen) {
    generatedTextInput = '';
    console.log('Generating text...');

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
      await onTextGenerationChar(winnerChar);

      generated += winnerChar;
      sentenceIndices = sentenceIndices.slice(1);
      sentenceIndices.push(winnerIndex);

      input.dispose();
      output.dispose();
    }
    temperatureScalar.dispose();
    return generated;
}


////////////////////////////////////////////////////////
// end functions from LoadableLSTMTextGenerator class //
////////////////////////////////////////////////////////


/**
 * Load generator (lstm model) function
 */
async function loadGen(){
  const model = await tf.loadModel('file://./models/nietzsche.json');
  
  lstmLayersSizesInput = lstmLayerSizes(model);
  model.summary();
  return model;
};

/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
async function onTextGenerationChar(charac) {
  generatedTextInput += charac;
  const charCount = generatedTextInput.length;
  const generateLength = generateLengthInput;
  const currStatus = `Generating text: ${charCount}/${generateLength} complete...`;
  //console.log(currStatus);
  //console.log('character in onTextGenerationChar: '+charac);
  if (charCount / generateLength == 1) {
    console.log(generatedTextInput);
  }
  await tf.nextFrame();
}

/**
 * Use `textGenerator` to generate random text, show the characters in the
 * console as they are generated one by one.
*/
async function generateText(model, charSet, charSetSize, sampleLen, seedTextInput) {
   try {
     if (model == null) {
       console.log('ERROR: Please load text data set first.');
       return;
     }
     const generateLength = generateLengthInput;
     const temperature = temperatureInput;
     if (!(generateLength > 0)) {
       console.log(
         `ERROR: Invalid generation length: ${generateLength}. ` +
         `Generation length must be a positive number.`);
       return;
     }
     if (!(temperature > 0 && temperature <= 1)) {
       console.log(
         `ERROR: Invalid temperature: ${temperature}. ` +
         `Temperature must be a positive number.`);
       return;
     }

     let seedSentence;
     let seedSentenceIndices;
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
           charSets[selectedText].indexOf(seedSentence[i]));
       }
     }

     const sentence = await genText(
       seedSentenceIndices, generateLength, temperature, model, charSet, charSetSize, sampleLen);
     generatedTextInput = sentence;
     const currStatus = 'Done generating text.';
     console.log(currStatus);
	
     console.log('sentence in generateText: '+ sentence);
     return sentence;
   } catch (err) {
     console.log(`ERROR: Failed to generate text: ${err.message}, ${err.stack}`);
   }
}


/**
 * Function to set up the model / start the server / start listening for requests
 */
async function setUp() {
	// load the lstm model
	let model = await loadGen();

	// set up the server
	const app = http.createServer(async function(request,response) {
		var q, respJSON;
		
		// respond to query with generated text
		q = url.parse(request.url,true).query;
		if(q.inputText){
			console.log('there is input text!');
			let generatedText = '';
			if(q.inputText == 'test'){
				console.log('this is a test');
				generatedText = 'test returned correctly';
			} else {
				if (q.inputText.length < sampleLen){
					console.log('inputText is too short. using previous seed.');
				} else {
				seedTextInput = q.inputText;
				}
			
				console.log('seed: '+seedTextInput);
				// Generate text and output in console.
				try {
					generatedText = await generateText(model, charSet, charSetSize, sampleLen, seedTextInput);
				} catch (err) {
					console.log(err);
				}
			}
			respJSON = {generated: generatedText};
		} else {
			console.log('no input text.');
			respJSON = {generated: null};
		}

		response.writeHead(200, {'Content-Type': 'application/json','json':'true'});
		response.write(JSON.stringify(respJSON));
		response.end();
	});
	
	// start listening for requests
	app.listen(port);
}

setUp();

