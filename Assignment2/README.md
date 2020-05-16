# Data Mining Techniques - VU Amsterdam
## Assignment 2: Expedia Kaggle competition
* Annemijn Dijkhuis
* Sanne Donker
* Sam Verhezen

## General
The link to the competition can be found [here](https://www.kaggle.com/c/2nd-assignment-dmt-2020/).

## Data exploration
See plots in results folder. (TODO nog beetje plotjes maken, zie [data description ppt](https://www.dropbox.com/sh/5kedakjizgrog0y/AABGuz_aKwCbpHvj7eTEbKc7a/ICDM_2013/2_data_description.pdf?dl=0))

## Preprocessing
### 1) Drop columns
Columns not taken into account for training set:
* "srch_adults_count"
* "srch_children_count"
* "srch_room_count"
* "date_time"
* "site_id"
* "gross_bookings_usd"

Columns not taken into account for test set:
* "srch_adults_count"
* "srch_children_count"
* "srch_room_count"
* "date_time"
* "site_id"

### 2) Add category variable to training set
Booked is represented by 5, clicked but not booked gets value 1, not clicked nor booked gets value 0. This variable is used to make the **prediction**.

### 3) Composite features
* Competitor columns have to be combined into 1 competitor variable. (TODO)
* Variable for price ranking per search id. (TODO)

### 4) Fill in missing values
This is done by taking the mean per column. (TODO)

### 5) Scale numerical data and transform categorical data
Numerical data is scaled, categorical data has yet to be selected and transformed. (TODO)

## Model
A neural network is used. First parameter tuning is performed (TODO).
