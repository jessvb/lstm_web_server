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

/**
 * TensorFlow.js Example: LSTM Text Generation.
 *
 * Inspiration comes from:
 *
 * -
 * https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
 * - Andrej Karpathy. "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 */

import 'babel-polyfill'; // todo--> added b/c I got ReferenceError: regeneratorRuntime is not defined...
import * as tf from '@tensorflow/tfjs';

import {
  onTextGenerationBegin,
  onTextGenerationChar,
  setUpUI
} from './ui';
import {
  sample
} from './utils';

/**
 * Class that loads LSTM models and manages LSTM-based text generation.
 *
 * The model is loaded from where the modelIdentifier is saved.
 */
export class LoadableLSTMTextGenerator {
  /**
   * Constructor of NeuralNetworkTextGenerator.
   */
  constructor(sampleLen, charSet, modelIdentifier) {
    this.charSet_ = charSet;
    this.charSetSize_ = charSet.length;
    this.sampleLen_ = sampleLen;
    this.modelIdentifier_ = modelIdentifier;
   this.modelType_ = 'LSTM';
    this.MODEL_LOAD_PATH_PREFIX_ = 'https://s3.amazonaws.com/lstm-model-bucket/pretrained_models';
    this.modelLoadPath_ =
      `${this.MODEL_LOAD_PATH_PREFIX_}/${this.modelIdentifier_}`;
    console.log('MODEL LOAD PATH: ' + this.modelLoadPath_);
  }

  /**
   * Get model identifier.
   *
   * @returns {string} The model identifier.
   */
  modelIdentifier() {
    return this.modelIdentifier_;
  }

  /**
   * Load LSTM model.
   */
  async loadModel() {
    console.log('Loading existing model...');
    this.model = await tf.loadModel(this.modelLoadPath_);
    console.log('Loaded model from ' + this.modelLoadPath_);
  }

  /**
   * Get a representation of the sizes of the LSTM layers in the model.
   *
   * @returns {number | number[]} The sizes (i.e., number of units) of the
   *   LSTM layers that the model contains. If there is only one LSTM layer, a
   *   single number is returned; else, an Array of numbers is returned.
   */
  lstmLayerSizes() {
    if (this.model == null) {
      throw new Error('Load model first.');
    }
    const numLSTMLayers = this.model.layers.length - 1;
    const layerSizes = [];
    for (let i = 0; i < numLSTMLayers; ++i) {
      layerSizes.push(this.model.layers[i].units);
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
  async generateText(sentenceIndices, length, temperature) {
    onTextGenerationBegin();
    const temperatureScalar = tf.scalar(temperature);

    let generated = '';
    while (generated.length < length) {
      // Encode the current input sequence as a one-hot Tensor.
      const inputBuffer =
        new tf.TensorBuffer([1, this.sampleLen_, this.charSetSize_]);
      for (let i = 0; i < this.sampleLen_; ++i) {
        inputBuffer.set(1, 0, i, sentenceIndices[i]);
      }
      const input = inputBuffer.toTensor();

      // Call model.predict() to get the probability values of the next
      // character.
      const output = this.model.predict(input);

      // Sample randomly based on the probability values.
      const winnerIndex = sample(tf.squeeze(output), temperatureScalar);

      const winnerChar = this.charSet_[winnerIndex];
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
}

setUpUI();