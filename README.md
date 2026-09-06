# Football Match Prediction

A machine learning project for predicting football match outcomes using historical match data and pre-match team-strength features.

The project focuses on predicting the three possible match outcomes:

- 🏠 **Home Win**
- 🤝 **Draw**
- ✈️ **Away Win**

Rather than simply predicting a class, the model produces probabilities for each outcome and is evaluated primarily using **log loss**.

## Overview

Football prediction is a challenging machine learning problem because team strength changes over time, historical relationships between teams can matter, and the available information changes from match to match.

This project explores how historical information can be transformed into **pre-match features** while avoiding data leakage.

The current prediction pipeline uses:

- **Elo ratings** to represent team strength
- **Head-to-head (H2H)** statistics
- **Multiclass Logistic Regression** for outcome prediction

The project currently focuses on the English football league dataset spanning multiple seasons.

## Experiments

Other features currently being tested include:
- **Rolling form** experiments
- **Squad value** experiments

---

## How It Works

The general pipeline is:

```text
Historical Match Data
        │
        ▼
Data Preparation
        │
        ▼
Feature Engineering
   ┌───────────────────┐
   │                   │
   ▼                   ▼
 Elo                  H2H
   │                   │
   └───────────────────┘
        │
        ▼
   Feature Dataset
        │
        ▼
 Logistic Regression
        │
        ▼
 P(Home) / P(Draw) / P(Away)
        │
        ▼
       Evaluation