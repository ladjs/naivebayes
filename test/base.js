const test = require('ava');
const NaiveBayes = require('../src/naivebayes.js');
const classifier = new NaiveBayes();

function decode(text) {
  return Buffer.from(text, 'base64').toString();
}

test('naivebayes', (t) => {
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
  classifier.learn('crud, this sucks', 'negative');
  classifier.learn('awful, no way', 'negative');

  classifier.learn(decode('R2V0IG91dCAhQmVhdCBpdCEgR2V0IGxvc3Qh'), 'foul');
  classifier.learn(decode('R28gdG8gaGVsbCEgR28gdG8gdGhlIGRldmlsIQ=='), 'foul');
  classifier.learn(decode('T2gsIGhlbGwncyBiZWxscyE='), 'foul');
  classifier.learn(decode('WW91IFNPQiAoc29uIG9mIGEpIQ=='), 'foul');
  classifier.learn(decode('U09HIChzb24gb2YgR3VuKSE='), 'foul');
  classifier.learn(decode('RGFtbiB5b3Uh'), 'foul');

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
