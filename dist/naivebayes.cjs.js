/**
 * @file naivebayes v0.5.0
 * @copyright Copyright (c) Surmon. All rights reserved.
 * @license Released under the MIT License.
 * @author Surmon <https://github.com/surmon-china>
 */
"use strict";var t=["categories","docCount","totalDocuments","vocabulary","wordCount","wordFrequencyCount","options"],o=function(t){return t.replace(/[^(a-zA-ZA-Яa-я\u4e00-\u9fa50-9_)+\s]/g," ").replace(/[\u4e00-\u9fa5]/g,(function(t){return t+" "})).split(/\s+/)},r=function(t){if(this.options={},void 0!==t){if(!t||"object"!=typeof t||Array.isArray(t))throw TypeError("NaiveBayes got invalid `options`: `"+t+"`. Pass in an object.");this.options=t}this.tokenizer=this.options.tokenizer||o,this.vocabulary=[],this.totalDocuments=0,this.docCount={},this.wordCount={},this.wordFrequencyCount={},this.categories=[]};r.prototype.initializeCategory=function(t){return this.categories.includes(t)||(this.docCount[t]=0,this.wordCount[t]=0,this.wordFrequencyCount[t]={},this.categories.push(t)),this},r.prototype.learn=function(t,o){var r=this;this.initializeCategory(o),this.docCount[o]++,this.totalDocuments++;var e=this.tokenizer(t),n=this.frequencyTable(e);return Object.keys(n).forEach((function(t){r.vocabulary.includes(t)||r.vocabulary.push(t);var e=n[t];r.wordFrequencyCount[o][t]?r.wordFrequencyCount[o][t]+=e:r.wordFrequencyCount[o][t]=e,r.wordCount[o]+=e})),this},r.prototype.categorize=function(t,o){return o?this.probabilities(t)[0]:this.probabilities(t)[0].category},r.prototype.probabilities=function(t){var o=this,r=this.tokenizer(t),e=this.frequencyTable(r);return this.categories.map((function(t){var r=o.docCount[t]/o.totalDocuments,n=Math.log(r);return Object.keys(e).forEach((function(r){var i=e[r],s=o.tokenProbability(r,t);n+=i*Math.log(s)})),{category:t,probability:n}})).sort((function(t,o){return o.probability-t.probability}))},r.prototype.tokenProbability=function(t,o){return((this.wordFrequencyCount[o][t]||0)+1)/(this.wordCount[o]+this.vocabulary.length)},r.prototype.frequencyTable=function(t){var o=Object.create(null);return t.forEach((function(t){o[t]?o[t]++:o[t]=1})),o},r.prototype.toJson=function(t){var o=t?2:0;return JSON.stringify(this.toJsonObject(),null,o)},r.prototype.toJsonObject=function(){var o=this,r={};return t.forEach((function(t){return r[t]=o[t]})),r},r.fromJson=function(o){if("string"==typeof o)try{o=JSON.parse(o)}catch(t){throw new Error("Naivebayes.fromJson expects a valid JSON string.")}o.options=o.options||{};var e=new r(o.options);return t.forEach((function(t){if(null==o[t])throw new Error("NaiveBayes.fromJson: JSON string is missing an expected property: '"+t+"'.");e[t]=o[t]})),e},r.getStateKeys=function(){return t};var e=r;module.exports=e;