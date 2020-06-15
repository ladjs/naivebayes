# [**@ladjs/naivebayes**](https://github.com/ladjs/naivebayes)

[![build status](https://img.shields.io/travis/com/shaunwarman/naivebayes.svg)](https://travis-ci.com/shaunwarman/naivebayes)
[![code coverage](https://img.shields.io/codecov/c/github/shaunwarman/naivebayes.svg)](https://codecov.io/gh/shaunwarman/naivebayes)
[![code style](https://img.shields.io/badge/code_style-XO-5ed9c7.svg)](https://github.com/sindresorhus/xo)
[![styled with prettier](https://img.shields.io/badge/styled_with-prettier-ff69b4.svg)](https://github.com/prettier/prettier)
[![made with lass](https://img.shields.io/badge/made_with-lass-95CC28.svg)](https://lass.js.org)
[![npm downloads](https://img.shields.io/npm/dt/@ladjs/naivebayes.svg)](https://npm.im/@ladjs/naivebayes)

> A ladjs naivebayes package forked from surmon-china/naivebayes


## Table of Contents

* [What can I use this for](#what-can-i-use-this-for)
* [Install](#install)
  * [npm](#npm)
  * [yarn](#yarn)
* [Usage](#usage)
* [API](#api)
  * [Class](#class)
  * [Learn](#learn)
  * [Probabilities](#probabilities)
  * [Categorize](#categorize)
  * [ToJson](#tojson)
  * [ToJsonObject](#tojsonobject)
  * [FromJson](#fromjson)
  * [Debug](#debug)
* [Contributors](#contributors)


## What can I use this for

Naive-Bayes classifier for JavaScript.

`naivebayes` takes a document (piece of text), and tells you what category that document belongs to.

You can use this for categorizing any text content into any arbitrary set of **categories**. For example:

* Is an email **spam**, or **not spam** ?
* Is a news article about **technology**, **politics**, or **sports** ?
* Is a piece of text expressing **positive** emotions, or **negative** emotions?


## Install

### npm

```sh
npm install @ladjs/naivebayes
```

### yarn

```sh
yarn add @ladjs/naivebayes
```


## Usage

```javascript
const NaiveBayes = require('naivebayes')

const classifier = new NaiveBayes()

// teach it positive phrases
classifier.learn('amazing, awesome movie!! Yeah!! Oh boy.', 'positive')
classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')

// teach it a negative phrase
classifier.learn('terrible, cruddy thing. Damn. Sucks!!', 'negative')

// now ask it to categorize a document it has never seen before
classifier.categorize('awesome, cool, amazing!! Yay.')
// => 'positive'

// serialize the classifier's state as a JSON string.
const stateJson = classifier.toJson()

// load the classifier back from its JSON representation.
const revivedClassifier = NaiveBayes.fromJson(stateJson)

```

```javascript
const NaiveBayes = require('naivebayes')

const Segment = require('segment')
const segment = new Segment()

segment.useDefault()

const classifier = new NaiveBayes({

    tokenizer(sentence) {

        const sanitized = sentence.replace(/[^(a-zA-Z\u4e00-\u9fa50-9_)+\s]/g, ' ')

        return segment.doSegment(sanitized, { simple: true })
    }
})
```


## API

### Class

```javascript
const classifier = new NaiveBayes([options])
```

Returns an instance of a Naive-Bayes Classifier.

#### Options

* `tokenizer(text)` - (type: `function`) -  Configure your own tokenizer.
* `vocabularyLimit` - (type: `number` default: 0) - Reference a max word count where `0` is the default, meaning no limit.
* `stopwords` - (type: `boolean` default: false) - To remove [stopwords](https://en.wikipedia.org/wiki/Stop_words) from text

Eg.

```javascript
const classifier = new NaiveBayes({
    tokenizer(text) {
        return text.split(' ')
    }
})
```

### Learn

```javascript
classifier.learn(text, category)
```

Teach your classifier what `category` the `text` belongs to. The more you teach your classifier, the more reliable it becomes. It will use what it has learned to identify new documents that it hasn't seen before.

### Probabilities

```javascript
classifier.probabilities(text)
```

Returns an array of `{ category, probability }` objects with probability calculated for each category. Its judgement is based on what you have taught it with `.learn()`.

### Categorize

```javascript
classifier.categorize(text ,[probability])
```

Returns the `category` it thinks `text` belongs to. Its judgement is based on what you have taught it with `.learn()`.

### ToJson

```javascript
classifier.toJson()
```

Returns the JSON representation of a classifier. This is the same as `JSON.stringify(classifier.toJsonObject())`.

### ToJsonObject

```javascript
classifier.toJsonObject()
```

Returns a JSON-friendly representation of the classifier as an `object`.

### FromJson

```javascript
const classifier = NaiveBayes.fromJson(jsonObject)
```

Returns a classifier instance from the JSON representation. Use this with the JSON representation obtained from `classifier.toJson()`.

### Debug

To run `naivebayes` in debug mode simply set `DEBUG=naivebayes` when running your script.


## Contributors

| Name             | Website                    |
| ---------------- | -------------------------- |
| **Surmon**       | <http://surmon.me/>        |
| **Shaun Warman** | <https://shaunwarman.com/> |
