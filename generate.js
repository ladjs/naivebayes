const fs = require('fs');
const path = require('path');
const { promisify } = require('util');
const { readDirDeep } = require('read-dir-deep');
const { simpleParser } = require('mailparser');

const NaiveBayes = require('.');
const classifier = new NaiveBayes({ vocabularyLimit: 300, stopwords: true });

const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

const FIXTURES_PATH = path.join(process.cwd(), 'test', 'fixtures');
const HAM_PATH = path.join(FIXTURES_PATH, 'ham');
const SPAM_PATH = path.join(FIXTURES_PATH, 'spam');

async function getEmailFromSource(filepath) {
  const sourceFiles = await readDirDeep(filepath);
  const files = await Promise.all(sourceFiles.map((file) => readFile(file)));
  const emails = await Promise.all(files.map((file) => simpleParser(file)));
  const texts = emails.map((email) => email.text);
  return texts;
}

(async () => {
  const hamEmails = await getEmailFromSource(HAM_PATH);
  console.log('got ham emails');
  const spamEmails = await getEmailFromSource(SPAM_PATH);
  console.log('got spam emails');

  for (const text of hamEmails) {
    if (text) {
      classifier.learn(text.trim(), 'ham');
    }
  }

  for (const text of spamEmails) {
    if (text) {
      classifier.learn(text.trim(), 'spam');
    }
  }

  await writeFile(
    path.join(FIXTURES_PATH, 'classifier-limit.json'),
    classifier.toJson()
  );
})();
