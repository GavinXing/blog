---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.Eng, Shanghai Jiaotong University, 2018 (expected)

Research experience
======
* 11/2016 - present:Big data Driven Individual Health Management of Cardiovascular 
  * National Natural Science Foundation Project of China ; SJTU Advanced Data and Programming Technology Lab
  *   Worked as a team leader being responsible for a subproject “Cardiovascular Disease Dictionary and Disease-Symptom Relation Extraction Based on Social Data”
  * Applied disease-symptom relation extraction on the question-answer pairs between citizen and 
  * Supervisor: Prof. Kenny Q. Zhu

* 07/2016 - 10/2016: Transform Free Text Airline Regulation to Structured Data
  * SJTU Advanced Data and Programming Technology Lab ; Ctrip.Com International, Ltd.
  * Involved in the structured data extraction from free text airline regulation, such as whether change is permitted for reissue/revalidation, whether cancelation is permitted for no-show, how much is charged for reissue, etc.
   Supervisor: Prof. Kenny Q. Zhu

* 09/2015 - 05/2016: Bank Card Digits Recognition Based on Support Vector Machine 
  * Participated in the picture preprocessing (cutting, drying, edge extraction) and feature engineering to realize the identification of bank card digits through images
   Supervisor: Prof. Fei Huan

Work experience
======
* 07/2017 - present: Machine Learning Engineer
  * Shanghai Synyi Medical Technology Co., Ltd.
  * Carried out medical knowledge extraction in medical literature (books, papers and medical records), a pipeline including Chinese word segmentation, clinical named entity recognition and relation extraction
  * Supervisor: Dr. Shaodian Zhang

* 01/2016 - 08/2016: Front-end Engineer
  * Shanghai Zhinan Information Technology Co., Ltd.
  * Assisted in front-end development for several websites
  
Skills
======
* Programming language
  * C/C++
  * Python
  * Matlab
* Deep learning framework
  * TensorFlow
  * PyTorch
* Interests
  * Guitar
  * Calligraphy
  * Photography

{% if site.publications.size > 0 %}
Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
{% endif %}
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
  
{% if site.teaching.size > 0 %}
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
{% endif %}
  
Service and leadership
======
* Currently signed in to 43 different slack teams
