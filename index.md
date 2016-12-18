<nav>
  <h1>Overview</h1> |
  <a href="/exploration/exploration.md">Exploration</a> |
  <a href="/preprocessing/cleaning.md">Preprocessing</a> |
  <a href="/model/model.md">Model Building</a> |
  <a href="/feature_building/features.md">Feature Creation</a>
</nav>

# Overview

### Motivation
<p>Peer-to-peer borrowing or rental markets, otherwise known as the sharing economy, have surfaced as alternative suppliers of services traditionally delivered by long-established companies in industries that were previously thought to have a high barrier to entry. Since its founding in 2008, Airbnb has been successful in disturbing the well-established hotel industry – becoming the poster child of a thriving business modeled after the sharing economy. By finding a way to enable users to make passive income on property that would otherwise be vacant, Airbnb has found a creative approach to balancing the supply and demand for affordable short-term housing. One of the unique features that helps Airbnb balance this supply and demand is their suggested price feature. This feature offers hosts a suggested listing price based on the information provided in the listing in addition to existing reviews and host ratings. The price suggestion features helps hosts have a better idea of the fair market value of their listing, therefore better optimizing user experience by further streamlining the process of listing, thus enhancing overall user experience.</p>

### Objectives
<p>In this project, we set out to use the given information provided in an Airbnb listing to implement our own price suggestion model. Our two key objectives are:</p>

* Cleaning and streamlining our data set to focus on key variables that have predictive potency in predicting listing price.
* Analyzing the relationship that date has on listing price, to better help optimize our model to predict the optimal listing price on a specified date.

### Data Acquisition
<p>The primary data source for this project comes from publicly available January 2015 Airbnb listings in the New York City area (including review data and daily pricing data of selected listings in 2015).</p>

* [January 2015 Listing Data](http://data.beta.nyc/dataset/inside-airbnb-data/resource/9d64399b-36d6-40a9-b0bb-f26ae0d9c53f)
* [January 2015 Calendar Data](http://data.beta.nyc/dataset/inside-airbnb-data/resource/ce0cbf46-83f9-414a-8a1d-7fd5321d83ca)
* [January 2015 Review Data](http://data.beta.nyc/dataset/inside-airbnb-data/resource/8115833e-8a0e-4af6-8aed-4d96a0ae0b73)

<p>In addition, we used publicly available transit data on subway stations in New York City as provided by the New York State Government.</p>

* [NYC Subway Data](https://data.ny.gov/Transportation/NYC-Transit-Subway-Entrance-And-Exit-Data/i9wp-a4ja/data)

### Further Exploration
<p>While we tried to make our project as expansive as possible, there are some areas of further exploration outside the scope of the project that we believe would be interesting to explore as an extension of this project.</p>

* **N Grams:** For our sentiment analysis on listing title and reviews, we used a method focused on gauging the sentiment of individual words. However, a more meaningful exploration of sentiment analysis could come from analyzing word pairs and phrases. For example, the sentiment of phrases such as “not beautiful” is lost when analyzing the sentiment of “not” independent of “beautiful”.

* **Visual Feature Extraction:** One of the more interesting potential features that we would’ve liked to add to our model is that of visual feature extraction. This would’ve enabled us to take into account the choice and quality of photo that a host choses to upload affects suggested listing price. Creating such a feature would’ve mirrored the bag of words method that we employed when evaluating title and review sentiment, except this time we would have sampled random images to create our visual dictionary.

* **Geographic Model Dependency**: Another interesting phenomenon to observe is how geography affects our price prediction model. For example, comparing the important predictive features we found in New York with that of another urban city or even a suburban area. It would’ve been interesting to see whether the same features tend to manifest themselves across urban cities other whether these predictive features are akin to certain cities.

### References

Amato, Nick. 2016. "Predicting Airbnb Listing Prices With Scikit-Learn And Apache Spark". Blog. Converge Blog. [Predicting Airbnb Listing Prices Scikit-Learn and Apache Spark.](https://www.mapr.com/blog/predicting-airbnb-listing-prices-scikit-learn-and-apache-spark)

Bird, Steven, Ewan Klein, and Edward Loper. 2009. Natural Language Processing With Python. 1st ed. Beijing: O'Reilly.

Tang, Emily and Kunal Sangani. Neighborhood And Price Prediction For San Francisco Airbnb Listings. Stanford, CA: Stanford University, 2016. Web. 10 Dec. 2016. [Airnbnb San Francisco Price Prediction.](http://cs229.stanford.edu/proj2015/236_report.pdf)
