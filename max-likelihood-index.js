

/**
 * This instance uses an unsmoothed maximum likelihood model for text generation, rather than
 * the original idea of using an LSTM text generator. The benefit of using this model over either
 * the word-based LSTM and the char-based LSTM text generators is the time.
 *
 * This model generates the models in what's practically constant time. During the initial tests,
 * I set it to generate 16 characters and it took about 1800 ms. When I set it to 60,000 characters,
 * again it took about 1800 ms. I want to believe that this is du to the fact that this model
 * takes advantage of the speed of Python dictionaries. Either way, this model is much more
 * effective if we want to use it as an educational tool. Some of the curriculum will have to
 * change in order to reflect the actual change in model architecture.
 *
 * TODO - Find some way to prepare this for upscaling.
 */

const PythonShell = require('python-shell').PythonShell;
const MAX_LIKELIHOOD_PY_SCRIPT = 'max-likelihood.py';

// for simple_server:
const url  = require('url');
const http = require('http');
const port = 1234;

const LOG_QUERY_INPUTS  = true;     // Logs all of the query stuff
const LOG_TEXT_GEN_PROGRESS = true; // Logs the progress of generating text every 20%
const LOG_WORD_TOKENIZING = false;  // Logs the number that each word is mapped to

const DEFAULT_MODEL = 'narnia';
const DEFAULT_OUTPUT_LEN = 40;

const modelNames = ["aliceInWonderland", "drSeuss", "hamlet", "harryPotter", "hungerGames", "nancy", "narnia", 'shakespeare', 'wizardOfOz'];

/**
 * generateText - Using the maximum likelihood models, this runs a python script
 * which will return the model's predictions
 *
 * @param  {String}  model        The name of the model's dataset, given in the array above
 * @param  {String}  corpus       The seed text to feed the model
 * @param  {Integer} outputLength The desired number of characters we want the model to generate
 * @return {Promise}              A promise that will resolve into the string the model generated
 */
async function generateText (model, corpus, outputLength) {
  console.log("Beginning to generate text.");
  var options = {
    mode: 'text',
    pythonOptions: ['-u'],
    args: [model, corpus, outputLength]
  };
  return new Promise ((resolve, reject) => {
    PythonShell.run(MAX_LIKELIHOOD_PY_SCRIPT, options, function (err, results) {
      if (err) reject(err);
      // Results is an array consisting of messages collected during execution. Preserve the number of lines
      resolve(results.join('\n'));
    });
  });
}



/**
 * Function to set up the model / start the server / start listening for requests
 */
async function setup () {

  const app = http.createServer ( async function (request, response) {
    // TODO Respond to favicon.ico request properly if necessary
    if (request.url == '/favicon.ico')
      return response.end();

    /**
     * respond - Generates an object and
     *
     * @param  {type} response description
     * @return {type}          description
     */
    function respond (text) {
      respJSON = { generated: text };
      response.writeHead(200, { 'Content-Type': 'application/json', 'json': 'true' });
      response.write(JSON.stringify(respJSON));
      response.end();
    }


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
    if (q.model && modelNames.includes(q.model))
      modelName = q.model;
    else if (q.model) {
      console.log(`The requested model '${q.model}' does not exist.`);
      return respond('The requested model does not exist');
    }
    console.log(`  Using the following model: '${modelName}'`);

    // if there is no input text, return an error
    // TODO, put an error code instead of a normal response with the error message
    if (!(q.inputText)) {
      console.log('There was no input text provided. Throwing error...');
      return respond("There was no input text provided. Please provide some text to start the text generation");
    }

    // Handles a simple test
    if (q.inputText == 'test') {
      console.log('This is a test!');
      return respond ("The test returned correctly!");
    }

    // Generate text, or throw an error otherwise.
    try {
      let result = await generateText (modelName, q.inputText, outputLen);
      console.log("Finished generating text.");
      return respond (result);

    } catch (error) {
      // TODO Provide a propper errorCode in response
      console.log(error);
      return respond ("An error has occured when generating text.");
    }
  });

  app.listen(port);
  console.log("Server listening on port: " + port);
}


setup();
