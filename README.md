# Load an LSTM Model and Generate Text

This project shows how to host a handful of pre-trained text generating models on a node server. The Github repository with information on these models and how they were trained can be found [here](https://github.com/tomiyee/lstm-text-gen).

## About the Models Used

Three different approaches were used in creating these text-generating models. 

The first approach is based on the LSTM text generation model presented in this [tensorflowjs example](https://github.com/tensorflow/tfjs-examples/tree/master/lstm-text-generation), which generates one character at a time. I will call this the **char-based model**. Because these text-generating models were intended for a younger audience, we found it favorable to apply some spelling correction to the generated text and some word filtering. (Todo, improve the word filtering).

The second approach uses a very similar LSTM model which generates entire words at a time instead of character by character. I will call this model the **word-based model.** With the word-based model, we would not need to run a spell check on the output, since the model would only choose from words which it has seen before, ideally spelled correctly in the corpus. Additionally, we hoped that the speed of this model would be faster than the char-based model. Although this was the case, the speed increase was not as dramatic as we would have liked. 

The third and final approach was to stray away from LSTM models, and apply an [Unsmoothed Maximum Likelihood Character Level Language Model](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139?utm_content=bufferefcf2&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer). I will call this model the **max-likelihood model.** In short, this model looks at the entire corpus and tracks the probability a certain character follows the given phrase. For more detail, I recommend looking at the notebook linked above. This model is incredibly fast and will never give a spelling mistake. Syntactical and grammatical errors remain a problem though.


## Running the Node Server

By default, this project will run instances of the char-based models trained on a number of corpuses in the index.js file. If you would like to run a node server which will generate text using the word-based model or the max-likelihood model, run `node word-based-index.js` or `node max-likelihood-index.js` instead. The instructions for querying the models is given in the section below. 

Before you can run a node server with either the char-based or word-based models, install the tfjs-node package with `npm install @tensorflow/tfjs-node`. To run to project, run the command `node index.js`.

Before you can run a node server with the max-likelihood model, install the python shell package with `npm install python-shell`. 

## Querying the Node server

When querying the node server, you can provide various parameters such as the model you want to use, the seed text you want to provide, and the length you would like the generated text to be (in characters). The only parameter that is required is the seed text, which must meet or exceed a length of 40 characters.

### Model Name

You can then specify the model you would like to load by adding `model=` and adding the name of the model. Names of models currently included can be found in the variable `modelFileNames` in the index.js file.

For the char-based and word-based models, the structure for the models is `<name>_<num epochs>`

<table>
    <tr>
        <th>Corpus</th>
        <th>Char-Based</th>
        <th>Word-Based</th>
        <th>Max-Likelihood</th> 
    </tr>
    <tr>
        <td>alice-in-wonderland.txt</td>
        <td>aliceInWonderland_0<br/>aliceInWonderland_1<br/>aliceInWonderalnd_5<br/>aliceInWonderland_20</td>
        <td>aliceInWonderland_0<br/>aliceinWonderland_25<br/>aliceinWonderland_100<br/>aliceinWonderland_500</td>
        <td>aliceInWonderland</td>
    </tr>
    <tr>
        <td>drseuss.txt</td>
        <td>drSeuss_0<br/>drSeuss_1<br/>drSeuss_5<br/>drSeuss_20</td>
        <td>drSeuss_0<br/>drSeuss_25<br/>drSeuss_100<br/>drSeuss_500</td>
        <td>drSeuss</td>
    </tr>
    <tr>
        <td>harry-potter-1.txt</td>
        <td>harryPotter_0<br/>harryPotter_1<br/>harryPotter_5<br/>harryPotter_20</td>
        <td>harryPotter_0<br/>harryPotter_25<br/>harryPotter_100<br/>harryPotter_500</td>
        <td>harryPotter</td>
    </tr>
    <tr>
        <td>nancy-drew.txt</td>
        <td>nancy_0<br/>nancy_1<br/>nancy_5<br/>nancy_20</td>
        <td>nancy_0<br/>nancy_25<br/>nancy_100<br/>nancy_500</td>
        <td>nancy</td>
    </tr>
    <tr>
        <td>narnia-1.txt</td>
        <td>narnia_1_0<br/>narnia_1_1<br/>narnia_1_5<br/>narnia_1_20</td>
        <td>narnia_0<br/>narnia_25<br/>narnia_100<br/>narnia_500</td>
        <td>narnia</td>
    </tr>
    <tr>
        <td>tomsawyer.txt</td>
        <td>tomSawyer_0<br/>tomSawyer_1<br/>tomSawyer_5<br/>tomSawyer_20</td>
        <td>tomSawyer_0<br/>tomSawyer_25<br/>tomSawyer_100<br/>tomSawyer_500</td>
        <td>tomSawyer</td>
    </tr>
    <tr>
        <td>wizard-of-oz.txt</td>
        <td>wizardOfOz_0<br/>wizardOfOz_1<br/>wizardOfOz_5<br/>wizardOfOz_20</td>
        <td>wizardOfOz_0<br/>wizardOfOz_25<br/>wizardOfOz_100<br/>wizardOfOz_500</td>
        <td>wizardOfOz</td>
    </tr>
</table>

In addition, the max-likelihood model has the models `hamlet`, `hungerGames`, and `shakespeare` which you can use to generate text. 


### Seed Text

When querying the node server, the only required input is the seed text, `inputText=`. The only requirement for the seed text is that, when using the char-based model, the seed text needs to be at least 40 characters long. The other word-based and max-ikelihood models, the seed text just needs to be at least one character long. 

### Output Text Length

You can also specify the number of characters which you want the resulting string to be. Simply add `outputLength=` followed by the number of characters. The default values are:

<table>
    <tr><th>Model Architecture</th><th>Default Output Length</th></tr>
    <tr><td>Char-Based</td><td>40 characters</td></tr>
    <tr><td>Word-Based</td><td>40 characters</td></tr>
    <tr><td>Max-Likelihood</td><td>40 words</td></tr>
</table>

### Spell Check [Default=1]

This setting applies only to the char-based model. You can deactivate the spellchecker for the model by setting `spellcheck=0` in the query. By default, the model's output gets checked by a spell checker and takes the first suggestion for the correction.

### Example URL

`path-to-the-node-server:1234/?inputText=hello%20there%20how%20are%20you%20today%20because%20I%20am%20doing%20pretty%20well&model=drSeuss_20&outputLength=10`

To simply test if the server is online and responding properly, the server checks to see if `inputText='test'`. If it does, it will return "The test returned correctly" for all of the node servers. 

`path-to-the-node-server:1234/?inputText=test&model=narnia_100`

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
