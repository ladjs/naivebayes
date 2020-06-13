const test = require('ava');
const NaiveBayes = require('../src/naivebayes.js');

function decode(text) {
  return Buffer.from(text, 'base64').toString();
}

test('naivebayes with limit', (t) => {
  const classifier = new NaiveBayes({ vocabularyLimit: 10 });

  classifier.learn('amazing, awesome movie!! Yeah!! Oh boy.', 'positive');
  classifier.learn(
    'Sweet, this is incredibly, amazing, perfect, great!!',
    'positive'
  );
  classifier.learn('Do one thing at a time, and do well.', 'positive');
  classifier.learn('Never forget to say “thanks”.', 'positive');
  classifier.learn('Believe in yourself.', 'positive');

  classifier.learn('terrible, crappy thing. Dang. Stinks!!', 'negative');
  classifier.learn('ugh, bad. This is annoying.', 'negative');
  classifier.learn('No, why. This is dumb', 'negative');
  classifier.learn('Are you serious? This sucks!', 'negative');
  classifier.learn("I don't want to be here", 'negative');

  classifier.learn(decode('R2V0IG91dCAhQmVhdCBpdCEgR2V0IGxvc3Qh'), 'foul');
  classifier.learn(decode('R28gdG8gaGVsbCEgR28gdG8gdGhlIGRldmlsIQ=='), 'foul');
  classifier.learn(decode('T2gsIGhlbGwncyBiZWxscyE='), 'foul');
  classifier.learn(decode('WW91IFNPQiAoc29uIG9mIGEpIQ=='), 'foul');
  classifier.learn(decode('U09HIChzb24gb2YgR3VuKSE='), 'foul');
  classifier.learn(decode('RGFtbiB5b3Uh'), 'foul');

  const pFoul = classifier.categorize(decode('R2V0IGxvc3QgeW91IFNPQg=='));
  t.is(pFoul, 'foul');

  const pNegative = classifier.categorize('Oh no that is crappy');
  t.is(pNegative, 'negative');

  const pPositive = classifier.categorize('Sweet that was awesome');
  t.is(pPositive, 'positive');

  const classifierJson = classifier.toJson();
  const classifierJsonObject = classifier.toJsonObject();
  t.is(typeof classifierJson, 'string');
  t.is(typeof classifierJsonObject, 'object');
  t.deepEqual(classifierJsonObject.categories, [
    'positive',
    'negative',
    'foul'
  ]);
});

test('naivebayes from json with implicit limit', (t) => {
  const json = require('./fixtures/classifier-with-limit');
  const classifier = NaiveBayes.fromJson(json);
  t.is(classifier.vocabularyLimit, 80);

  const pFoul = classifier.categorize(decode('WW91IGdldCBvdXQh'));
  t.is(pFoul, 'foul');

  const pNegative = classifier.categorize('Oh no that is crappy');
  t.is(pNegative, 'negative');

  const pPositive = classifier.categorize('Sweet that was awesome');
  t.is(pPositive, 'positive');

  const state = classifier.toJsonObject();

  t.true(state.vocabulary.length <= 80);
});

test('naivebayes from json with explicit limit', (t) => {
  const json = require('./fixtures/classifier-with-limit');
  const classifier = NaiveBayes.fromJson(json, 80);
  t.is(classifier.vocabularyLimit, 80);

  classifier.learn(decode('WW91IGdldCBvdXQh'), 'foul');

  const pFoul = classifier.categorize(decode('WW91IGdldCBvdXQh'));
  t.is(pFoul, 'foul');

  const pNegative = classifier.categorize('Oh no that is crappy');
  t.is(pNegative, 'negative');

  const pPositive = classifier.categorize('Sweet that was awesome');
  t.is(pPositive, 'positive');

  const state = classifier.toJsonObject();

  t.true(state.vocabulary.length <= 80);
});
