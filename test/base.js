const test = require('ava');
const NaiveBayes = require('../src/naivebayes.js');
const classifier = new NaiveBayes();

function decode(text) {
  return Buffer.from(text, 'base64').toString();
}

test('naivebase with limit', (t) => {
  classifier.learn(
    decode('YW1hemluZywgYXdlc29tZSBtb3ZpZSEhIFllYWghISBPaCBib3ku'),
    'positive'
  );
  classifier.learn(
    decode(
      'U3dlZXQsIHRoaXMgaXMgaW5jcmVkaWJseSwgYW1hemluZywgcGVyZmVjdCwgZ3JlYXQhIQ=='
    ),
    'positive'
  );
  classifier.learn(
    decode('RG8gb25lIHRoaW5nIGF0IGEgdGltZSwgYW5kIGRvIHdlbGwu'),
    'positive'
  );
  classifier.learn(
    decode('TmV2ZXIgZm9yZ2V0IHRvIHNheSDigJx0aGFua3PigJ0u'),
    'positive'
  );
  classifier.learn(decode('QmVsaWV2ZSBpbiB5b3Vyc2VsZi4='), 'positive');
  classifier.learn(
    decode(
      'TmV2ZXIgcHV0IG9mZiB3aGF0IHlvdSBjYW4gZG8gdG9kYXkgdW50aWwgdG9tb3Jyb3cu'
    ),
    'positive'
  );
  classifier.learn(
    decode(
      'RG9uJ3QgYWltIGZvciBzdWNjZXNzIGlmIHlvdSB3YW50IGl0OyBqdXN0IGRvIHdoYXQgeW91IGxvdmUgYW5kIGJlbGlldmUgaW4sIGFuZCBpdCB3aWxsIGNvbWUgbmF0dXJhbGx5Lg=='
    ),
    'positive'
  );

  classifier.learn(
    decode('dGVycmlibGUsIHNoaXR0eSB0aGluZy4gRGFtbi4gU3Vja3MhIQ=='),
    'negative'
  );

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
