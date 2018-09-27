# Load an LSTM and Generate Text

This project will generate text using a pretrained LSTM network when provided with an input seed via a http request. This project is based off of this [tensorflowjs example](https://github.com/tensorflow/tfjs-examples/tree/master/lstm-text-generation). To create and download an LSTM model that can be loaded into this project, see the [lstm_model_creator](https://github.com/jessvb/lstm_model_creator). The server itself is based off of [simple_server](https://github.com/jessvb/simple_server.git).

Before running this project, install tfjs-node with `npm install @tensorflow/tfjs-node`. To run to project, use `node index.js`.

## Overview from [Original Source](https://github.com/tensorflow/tfjs-examples)

This example illustrates how to use TensorFlow.js to train a LSTM model to
generate random text based on the patterns in a text corpus such as
Nietzsche's writing or the source code of TensorFlow.js itself.

The LSTM model operates at the character level. It takes a tensor of
shape `[numExamples, sampleLen, charSetSize]` as the input. The input is a
one-hot encoding of sequences of `sampleLen` characters. The characters
belong to a set of `charSetSize` unique characters. With the input, the model
outputs a tensor of shape `[numExamples, charSetSize]`, which represents the
model's predicted probabilites of the character that follows the input sequence.
The application then draws a random sample based on the predicted
probabilities to get the next character. Once the next character is obtained,
its one-hot encoding is concatenated with the previous input sequence to form
the input for the next time step. This process is repeated in order to generate
a character sequence of a given length. The randomness (diversity) is controlled
by a temperature parameter.

The UI allows creation of models consisting of a single
[LSTM layer](https://js.tensorflow.org/api/latest/#layers.lstm) or multiple,
stacked LSTM layers.

This example also illustrates how to save a trained model in the browser's
IndexedDB using TensorFlow.js's
[model saving API](https://js.tensorflow.org/tutorials/model-save-load.html),
so that the result of the training
may persist across browser sessions. Once a previously-trained model is loaded
from the IndexedDB, it can be used in text generation and/or further training.

This example is inspired by the LSTM text generation example from Keras:
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

## Usage

```sh
yarn && node index.js
```
