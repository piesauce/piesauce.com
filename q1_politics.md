---
layout: page
title: The Political Embeddings Project
---

# Part 1 - Introduction

Citizens are constantly reminded of their civic duty to participate in democracy by being politically informed and casting their vote responsibly. However, even the most seasoned political aficionado would find it hard to keep track of all political developments, policy proposals, and electoral candidates. Politics is simply...complex.

Politics, as presented in the media to the general public, is extremely simplified. Complex multi-dimensional ideological positions are compressed into a two-dimensional left-right spectrum. Voters are forced to identify themselves within this narrow spectrum even while they reconcile with holding seemingly contradictory positions due to the lack of representational space in the traditional left-right spectrum. 

Can we do better? Can we use natural language processing and machine learning techniques to give citizens better insight into the political process? In this upcoming series of blog posts, we will aim to do exactly that.

Drawing on publicly available data on Canadian parliamentary proceedings, we aim to better understand the ideological positions of the parties involved as well as gain more insight into the bills introduced and voted on in the Parliament. We aim to provide a means to compare and contrast different political parties and representatives and help understand which party/politician comes closest to representing any individual’s political views.

We will start with a short primer on the Canadian parliamentary system for those unfamiliar with it. If you are already well-versed with the system, please feel free to skip the next section.

## The Canadian Parliament
The Parliament has two houses - The House of Commons and the Senate. The House of Commons is the more powerful house of the two.

The House of Commons has 338 members called MPs(Member of Parliament), voted in directly by the Canadian population across the ten provinces and territories. Each Member of Parliament represents an electoral district (‘riding’ in Canadian parlance), with each riding containing an average of 75,000 voters(as of 2015). Elections are held once every four years unless the ruling party or coalition loses ‘confidence’ of the parliament. To maintain confidence, 170 votes are required (50% + 1 votes).

The Senate is comprised of 105 members and its members are appointed on the advice of the Prime Minister. Senators usually keep their seats until the age of 75 unless they choose to resign or are removed.

## The Party System

Canada has a multi-party system and there are currently 5 parties represented in the House of Commons. The parties and their ideological positions according to Wikipedia are as follows:

Liberal Party of Canada (Centre to Centre-Left)
Conservative Party of Canada (Centre-Right to Right-wing)
New Democratic Party (Centre-Left to Left-wing)
Bloc Quebecois (Centre-Left)
Green Party of Canada (Centre-Left?)

## Bills
A bill is a proposed law that is introduced into the Canadian Parliament by a member of the HoC or the Senate. Bills can be ‘public’ or ‘private’. Government bills are usually introduced by a member of the ruling Cabinet from the HoC. Private Member’s bills can be introduced by either MPs or Senators, regardless of political affiliation.

## Voting for bills
Once introduced into Parliament, a bill goes through multiple readings and is voted on in both the HoC and the Senate. If the bill gets the requisite number of votes, it becomes law after receiving Royal Assent. Generally, voting for Government bills falls strictly along party lines, while Private Member’s bills usually see more independent voting.

Now that we have introduced the Canadian parliamentary system, we proceed to give a brief overview of the insights that we aim to uncover using NLP and machine learning techniques in the coming blog posts.

## Party Embeddings

We learn a vector for each party based on the voting history of its members and the text of the bills that they voted ‘Yes’ for. Using vector space distance measures, we try to understand the ideological distances between parties based on their voting records. 

Of course, every bill does not represent the same amount of ideological significance. A bill about making a particular day of the year National Autism Awareness day contains much less information ideologically than a comprehensive bill about environmental regulations across multiple sectors. We experiment with different weighting schemes of bills to take into account the relevant ideological importance of these. 

Once we have learned the vectors, we visualize them in 2D/3D space using dimensionality reduction techniques. More specifically, we aim to answer the following questions:

It is often alleged that the Liberals ‘campaign from the left, but govern from the right’. Does the data support it?
How similar are the Green Party and the NDP from each other?
Which party is the Bloc closest to, ideologically?

A yes/no signal sometimes doesn’t take into account the complex nuances behind the vote. For example, MPs of a party may vote against a bill not because they disagree with its contents, but because they feel it does too little. We experiment with different techniques to try to deal with these nuances wherever possible.
MP Embeddings

In a similar vein, we learn MP embeddings for each individual MP based on their previous voting record and the text of the bills they voted for. Once we obtain the embeddings, we try to answer questions such as the following:

Who are the habitual aisle crossers? 

Do we observe local clusters within a party cluster representing different factions within a party?

For a particular MP, which other MPs are its nearest neighbors in terms of voting records?

How deviant is a particular MP from the party voting average?

## Bill Embeddings

If Parties and MPs can get their own embeddings, why not Bills? We learn a document vector for each bill which can then be used for cluster analysis. We perform topic modeling using these representations to understand the different types of bills being voted on.

We also use the bill text to summarize the contents of a bill using both extractive and abstractive summarization techniques. Further, we also build a text simplification model to remove the ‘legalese’ portions of the bill text and produce a more layperson amenable text.

## Prediction

Now that we have all these representations, can we use them for prediction?
We can certainly try.
We perform the following predictions:

Predict if a bill introduced into parliament will become a law or not, based on the past voting behavior of the House’s members.

Predict the number of days/months it would take for a particular bill to become law based on past trends.

Predict the number of minutes for which a particular bill will be discussed in Parliament based on its initial text.

We will test our predictions using bills from the new Parliament. However, we are limited in some aspects - we do not have past voting behavior of new members. 91/338 members are newcomers to Parliament.

In the next post, we will introduce party embeddings and discuss the results and the techniques used in detail.

