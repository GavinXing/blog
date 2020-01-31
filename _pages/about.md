---
permalink: /
title: "Gavin Junjie Xing - 邢俊劼"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I'm a Master's student at University of Michigan, major in Computer Science and Engineering. I'm now working with [Prof. H.V. Jagadish](https://web.eecs.umich.edu/~jag/) at [UM DBGroup](http://dbgroup.eecs.umich.edu). My ultimate goal is to help both experts and novices gain access to data more easily, process data faster, and learn from data more efficiently.

Before my graduate study, I got my Bachelor's degree at Shanghai Jiaotong University. I worked with [Prof. Kenny Q. Zhu](http://www.cs.sjtu.edu.cn/~kzhu/) since the summer vacation of my sophomore year.



{% if site.news.size > 0 %}
News
====
  <ul>
	{% assign sorted = (site.news | sort: 'date') | reverse | slice: 0, 10 %}
  	{% for news in sorted %}
    <li><i>{{ news.date | date: '%B %d, %Y' }}</i> - {{ news.text }}</li>
  {% endfor %}</ul>
{% endif %}