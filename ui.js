/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {
  LoadableLSTMTextGenerator
} from './index';

// UI controls.
const getText = document.getElementById('get-text');
// const testText = document.getElementById('test-text');
let testText = {
  value: ''
};

let textDataSelect = {
  value: "drSeuss"
};

// const lstmLayersSizesInput = document.getElementById('lstm-layer-sizes');
let lstmLayersSizesInput = {
  value: '128'
};

// const examplesPerEpochInput = document.getElementById('examples-per-epoch');
let examplesPerEpochInput = {
  value: '2048'
};
// const batchSizeInput = document.getElementById('batch-size');
let batchSizeInput = {
  value: '128'
};
// const epochsInput = document.getElementById('epochs');
let epochsInput = {
  value: '5'
};
// const validationSplitInput = document.getElementById('validation-split');
let validationSplitInput = {
  value: '0.0625'
};
// const learningRateInput = document.getElementById('learning-rate');
let learningRateInput = {
  value: '1e-2'
};

// const generateLengthInput = document.getElementById('generate-length');
let generateLengthInput = {
  value: '200'
};
// const temperatureInput = document.getElementById('temperature');
let temperatureInput = {
  value: '0.75'
};
// const seedTextInput = document.getElementById('seed-text');
let seedTextInput = {
  value: 'This is a seed input. Hopefully it works.'
};
// const generatedTextInput = document.getElementById('generated-text');
let generatedTextInput = {
  value: ''
};

let charSet = ["T", "h", "e", " ", "C", "a", "t", "i", "n", "H", "↵", "B", "y", 
"D", "r", ".", "S", "u", "s", "d", "o", "I", "w", "p", "l", "A", "c", ",", "W", 
"\"", "m", "g", "!", "b", "k", "N", "U", "M", "P", "j", "?", "v", "L", "f", "Y", 
"O", "F", "-", "x", "X", "'", "E", "G", "K", "q", "J", "R", "V", "z", "Q", "", 
":", "â", "", "", ";", "Z", "(", "9", "8", "3", "/", "4", ")", "“", "’", "…", 
"”", "‘", "—", "1", "0", "$", " ", "­", ";", "6"]; // dr seuss charset

// const modelAvailableInfo = document.getElementById('model-available');

const sampleLen = 40;
const sampleStep = 3;

// Module-global instance of SaveableLSTMTextGenerator.
let textGenerator;

function logStatus(message) {
  console.log(message);
  // appStatus.textContent = message;
}

/**
 * A function to call when text generation begins.
 *
 * @param {string} seedSentence: The seed sentence being used for text
 *   generation.
 */
export function onTextGenerationBegin() {
  generatedTextInput.value = '';
  logStatus('Generating text...');
}

/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
export async function onTextGenerationChar(char) {
  generatedTextInput.value += char;
  const charCount = generatedTextInput.value.length;
  const generateLength = Number.parseInt(generateLengthInput.value);
  const status = `Generating text: ${charCount}/${generateLength} complete...`;
  logStatus(status);
  if (charCount / generateLength == 1) {
    console.log(generatedTextInput.value);
  }
  await tf.nextFrame();
}

export function setUpUI() {
  // /**
  //  * Refresh the status of locally saved model (in IndexedDB).
  //  */
  // async function refreshLocalModelStatus() {
  //   const modelInfo = await textGenerator.checkStoredModelStatus();
  //   if (modelInfo == null) {
  //     console.log(
  //       `No locally saved model for "${textGenerator.modelIdentifier()}".`
  //     );
  //   } else {
  //     console.log(`Saved @ ${modelInfo.dateSaved.toISOString()}`);
  //   }
  // }

  /**
   * Use `textGenerator` to generate random text, show the characters on the
   * screen as they are generated one by one.
   */
  async function generateText() {
    try {
      if (textGenerator == null) {
        logStatus('ERROR: Please load text data set first.');
        return;
      }
      const generateLength = Number.parseInt(generateLengthInput.value);
      const temperature = Number.parseFloat(temperatureInput.value);
      if (!(generateLength > 0)) {
        logStatus(
          `ERROR: Invalid generation length: ${generateLength}. ` +
          `Generation length must be a positive number.`);
        return;
      }
      if (!(temperature > 0 && temperature <= 1)) {
        logStatus(
          `ERROR: Invalid temperature: ${temperature}. ` +
          `Temperature must be a positive number.`);
        return;
      }

      let seedSentence;
      let seedSentenceIndices;
      if (seedTextInput.value.length === 0) {
        // [seedSentence, seedSentenceIndices] = textData.getRandomSlice();
        // seedTextInput.value = seedSentence;
        logStatus(
          `ERROR: seed sentence length is zero. Seed Sentence: ` +
          seedTextInput.value + '.');
        return;
      } else {
        seedSentence = seedTextInput.value;
        if (seedSentence.length < sampleLen) {
          logStatus(
            `ERROR: Seed text must have a length of at least ` +
            `${sampleLen}, but has a length of ` +
            `${seedSentence.length}.`);
          return;
        }
        seedSentence = seedSentence.slice(
          seedSentence.length - sampleLen, seedSentence.length);

        function textToIndices(text) {
          const indices = [];
          for (let i = 0; i < text.length; ++i) {
            indices.push(this.charSet_.indexOf(text[i]));
          }
          return indices;
        };
        seedSentenceIndices = textToIndices(seedSentence);
      }

      const sentence = await textGenerator.generateText(
        seedSentenceIndices, generateLength, temperature);
      generatedTextInput.value = sentence;
      const status = 'Done generating text.';
      logStatus(status);

      return sentence;
    } catch (err) {
      logStatus(`ERROR: Failed to generate text: ${err.message}, ${err.stack}`);
    }
  }

  function updateModelParameterControls(lstmLayerSizes) {
    lstmLayersSizesInput.value = lstmLayerSizes;
  }

  /**
   * Wire up UI callbacks.
   */

  getText.addEventListener('click', async () => {
    // // from loadTextDataButton:
    // let dataIdentifier = textDataSelect.value; // todo: allow user to change this in Alexa skill (currently set to julesverne)
    textGenerator = new LoadableLSTMTextGenerator(sampleLen, charSet, textDataSelect.value); //new SaveableLSTMTextGenerator(textData);

    // from createOrLoadModelButton:
    if (textGenerator == null) {
      logStatus('ERROR: Please load text data set first.');
      return;
    }

    // Load locally-saved model.
    logStatus('Loading model... Please wait.');
    // await textGenerator.loadModel(); // todo uncomment
    // updateModelParameterControls(textGenerator.lstmLayerSizes()); // todo uncomment
    logStatus(
      'Done loading model. ' +
      'Now you can use it to generate text.');

    // await generateText(); // TODO(JESS): uncomment :)
  });
}