---
permalink: /
title: "Gavin Junjie Xing - 邢俊劼"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I am a Senior Researcher in the [Data Systems Group](https://www.microsoft.com/en-us/research/group/datasystems/) at [Microsoft Research Redmond](https://www.microsoft.com/en-us/research/lab/microsoft-research-redmond/). I received my Ph.D. at University of Michigan Ann Arbor advised by [Prof. H. V. Jagadish](https://web.eecs.umich.edu/~jag/).

My research interests focus on data exploration, data integration, and data preparation. I’m particularly interested in leveraging AI and large language models (LLMs) to enhance these areas.

Before my graduate study, I received my Bachelor's degree at Shanghai Jiaotong University. I worked with [Prof. Kenny Q. Zhu](https://kenzhu2000.github.io/) since the summer vacation of my sophomore year.



{% if site.news.size > 0 %}
News
====
  <ul>
  {% assign sorted = site.news | sort: 'date' | reverse %}
    {% for news in sorted limit: 10 %}
    <li><i>{{ news.date | date: '%B %d, %Y' }}</i> - {{ news.text }}</li>
  {% endfor %}</ul>
{% endif %}
