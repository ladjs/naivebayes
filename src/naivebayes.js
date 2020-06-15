const debug = require('debug')('naivebayes');
const stopword = require('stopword');

/**
 * keys we use to serialize a classifier's state
 */
const STATE_KEYS = [
  'categories',
  'docCount',
  'totalDocuments',
  'vocabulary',
  'wordCount',
  'wordFrequencyCount',
  'options'
];

/**
 * Given an input string, tokenize it into an array of word tokens.
 * This is the default tokenization function used if user does not provide one in `options`.
 *
 * @param  {String} text
 * @return {Array}
 */
const defaultTokenizer = (text) => {
  const rgxPunctuation = /[^(A-Яa-я\u4E00-\u9FA5\w)+\s]/g;

  return text
    .replace(rgxPunctuation, ' ')
    .replace(/[\u4E00-\u9FA5]/g, (word) => `${word} `)
    .split(/\s+/);
};

/**
 * Naive-Bayes Classifier
 *
 * This is a naive-bayes classifier that uses Laplace Smoothing.
 *
 */
class NaiveBayes {
  constructor(options = {}) {
    // set options object
    this.options = {};
    if (typeof options !== 'undefined') {
      if (!options || typeof options !== 'object' || Array.isArray(options)) {
        throw new TypeError(
          'NaiveBayes got invalid `options`: `' +
            options +
            '`. Pass in an object.'
        );
      }

      this.options = options;
    }

    this.tokenizer = this.options.tokenizer || defaultTokenizer;

    this.vocabulary = [];

    // max vocabulary size based on word frequency, default is no limit
    this.vocabularyLimit = this.options.vocabularyLimit || 0;

    // number of documents we have learned from
    this.totalDocuments = 0;

    // document frequency table for each of our categories
    this.docCount = {};

    // filter stopwords from vocabulary
    this.stopwords = options.stopwords || false;

    // for each category, how many words total were mapped to it
    this.wordCount = {};

    // word frequency table for each category
    this.wordFrequencyCount = {};

    // hashmap of our category names
    this.categories = [];

    debug('init %O', this);
  }

  /**
   * Initialize each of our data structure entries for this new category
   *
   * @param  {String} categoryName
   */
  initializeCategory(categoryName) {
    if (!this.categories.includes(categoryName)) {
      this.docCount[categoryName] = 0;
      this.wordCount[categoryName] = 0;
      this.wordFrequencyCount[categoryName] = {};
      this.categories.push(categoryName);
    }

    return this;
  }

  /**
   * train our naive-bayes classifier by telling it what `category`
   * the `text` corresponds to.
   *
   * @param  {String} text
   * @param  {String} class
   */
  learn(text, category) {
    debug({ text, category });
    // initialize category data structures if we've never seen this category
    this.initializeCategory(category);

    // update our count of how many documents mapped to this category
    this.docCount[category]++;

    // update the total number of documents we have learned from
    this.totalDocuments++;

    // normalize the text into a word array
    let tokens = this.tokenizer(text);

    if (this.stopwords) {
      tokens = stopword.removeStopwords(tokens);
    }

    // get a frequency count for each token in the text
    const frequencyTable = this.frequencyTable(tokens);

    /*
     * Update our vocabulary and our word frequency count for this category
     */
    Object.keys(frequencyTable).forEach((token) => {
      // add this word to our vocabulary if not already existing
      if (!this.vocabulary[token]) {
        this.vocabulary.push(token);
      }

      const frequencyInText = frequencyTable[token];

      if (!this.wordFrequencyCount[category])
        this.wordFrequencyCount[category] = {};

      // update the frequency information for this word in this category
      if (this.wordFrequencyCount[category][token]) {
        this.wordFrequencyCount[category][token] += frequencyInText;
      } else {
        this.wordFrequencyCount[category][token] = frequencyInText;
      }

      // update the count of all words we have seen mapped to this category
      this.wordCount[category] += frequencyInText;
    });

    if (!this.vocabularyLimit || this.vocabulary.length <= this.vocabularyLimit)
      return this;

    const newFrequencyCount = {};
    for (const category in this.wordFrequencyCount) {
      if (Object.hasOwnProperty.call(this.wordFrequencyCount, category)) {
        const frequencyTable = this.wordFrequencyCount[category];
        const words = Object.keys(frequencyTable);
        if (words.length <= this.vocabularyLimit) {
          newFrequencyCount[category] = this.wordFrequencyCount[category];
          continue;
        }

        // sort words by highest frequency
        const frequentWords = words.sort(
          (a, b) => frequencyTable[b] - frequencyTable[a]
        );

        debug({ frequentWords });

        // build up new structure until vocab limit reached
        for (let count = 0; count < this.vocabularyLimit; count++) {
          const word = frequentWords[count];
          if (!newFrequencyCount[category]) {
            newFrequencyCount[category] = {};
          }

          newFrequencyCount[category][word] = this.wordFrequencyCount[category][
            word
          ];
        }
      }
    }

    this.wordFrequencyCount = newFrequencyCount;

    this.vocabulary = [];
    this.wordCount = {};
    const categories = Object.keys(this.wordFrequencyCount);
    for (const category of categories) {
      const words = Object.keys(this.wordFrequencyCount[category]);
      this.wordCount[category] = words.length;
      this.vocabulary = [...this.vocabulary, ...words];
    }

    return this;
  }

  /**
   * Determine what category `text` belongs to.
   *
   * @param  {String} text
   * @param  {Boolean} probability
   * @return {String} category
   */
  categorize(text, probability) {
    const category = probability
      ? this.probabilities(text)[0]
      : this.probabilities(text)[0].category;
    debug('categorize: %O', { text, category, probability });
    return category;
  }

  /**
   * Determine category probabilities for `text`.
   *
   * @param  {String} text
   * @return {Array} probabilities
   */
  probabilities(text) {
    // [W1,W2,W3,W4,Wn...]
    const tokens = this.tokenizer(text);
    const frequencyTable = this.frequencyTable(tokens);

    // P(W1|C) * P(W2|C) ... P(Wn|C) * P(C)
    // iterate thru our categories to calculate the probability for this text
    return this.categories
      .map((category) => {
        // start by calculating the overall probability of this category
        // => out of all documents we've ever looked at, how many were
        //    mapped to this category
        const categoryProbability =
          this.docCount[category] / this.vocabularyLimit
            ? this.wordCount[category]
            : this.totalDocuments;

        // take the log to avoid underflow
        let logProbability = Math.log(categoryProbability);

        // now determine P( w | c ) for each word `w` in the text
        Object.keys(frequencyTable).forEach((token) => {
          const frequencyInText = frequencyTable[token];
          const tokenProbability = this.tokenProbability(token, category);

          // determine the log of the P( w | c ) for this word
          logProbability += frequencyInText * Math.log(tokenProbability);
        });

        debug('probabilities: %O', { category, logProbability });

        return {
          category,
          probability: logProbability
        };
      })
      .sort((previous, next) => next.probability - previous.probability);
  }

  /**
   * Calculate probability that a `token` belongs to a `category`
   *
   * @param  {String} token
   * @param  {String} category
   * @return {Number} probability
   */
  tokenProbability(token, category) {
    if (!this.wordFrequencyCount[category])
      this.wordFrequencyCount[category] = {};

    const wordFrequencyCount = this.wordFrequencyCount[category][token] || 0;

    const wordCount = this.wordCount[category];

    // P(W|C)
    return (wordFrequencyCount + 1) / (wordCount + this.vocabulary.length);
  }

  /**
   * Build a frequency hashmap where
   * - the keys are the entries in `tokens`
   * - the values are the frequency of each entry in `tokens`
   *
   * @param  {Array} tokens  Normalized word array
   * @return {Object}
   */
  frequencyTable(tokens) {
    const frequencyTable = Object.create(null);
    for (const token of tokens) {
      if (frequencyTable[token]) {
        frequencyTable[token]++;
      } else {
        frequencyTable[token] = 1;
      }
    }

    return frequencyTable;
  }

  /**
   * Dump the classifier's state as a JSON string.
   * @param {Boolean} Optionally format the serialized JSON output for easier human consumption
   * @return {String} Representation of the classifier.
   */
  toJson(prettyPrint) {
    const prettyPrintSpaces = prettyPrint ? 2 : 0;
    return JSON.stringify(this.toJsonObject(), null, prettyPrintSpaces);
  }

  toJsonObject() {
    const state = {};
    for (const key of STATE_KEYS) {
      state[key] = this[key];
    }

    return state;
  }

  /**
   * Initializes a NaiveBayes instance from a JSON state representation.
   * Use this with classifier.toJson().
   *
   * @param  {String} jsonStr   state representation obtained by classifier.toJson()
   * @return {NaiveBayes}       Classifier
   */
  static fromJson(json, limit) {
    if (typeof json === 'string') {
      try {
        json = JSON.parse(json);
      } catch {
        throw new Error('Naivebayes.fromJson expects a valid JSON string.');
      }
    }

    if (json.options && limit) {
      json.options.vocabularyLimit = limit || 0;
    }

    // init a new classifier
    const classifier = new NaiveBayes(json.options);

    // override the classifier's state
    STATE_KEYS.forEach((key) => {
      if (json[key] === undefined) {
        throw new Error(
          `NaiveBayes.fromJson: JSON string is missing an expected property: '${key}'.`
        );
      } else {
        classifier[key] = json[key];
      }
    });

    return classifier;
  }

  static getStateKeys() {
    return STATE_KEYS;
  }
}

module.exports = NaiveBayes;
