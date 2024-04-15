# MovieVsCriticsAnalysis
Repo to contain the data set and code used to analyze the differences in approaches of reviewing film based on critic and audience reviews.

The dataset consists of four primary folders that account for the categories by which films are viewed by audiences and critics. These are films loved by both critics and audiences, disliked by both, disliked by audiences but liked by critics, and disliked by critics but liked by audiences. The primary site used to scrape the data was rotten tomatoes and overall sentiment of the film was flattened to either a positive review, negative review, or neutral review. For critics, only the first 100 available were scraped, and if a link to their full review was available, that was utilized as their review text. If the link was not available (or it linked to a video review), their review on rottentomatoes was used instead.

Audiences had the first 10,000 available reviews scraped. As of the moment finalized data cleaning has not been done to remove reviews in languages other than English. All reviews are isolated into a folder of their respective film containing two separate csvs, one for critics, and one for audiences. Only date, score, and review text is recorded.

Three useable python files are currently included. One for scraping critics, and one for scraping audiences. If used on other machines, the only necessary downloads are selenium beyond the standard python packages. Code is currently set to work with brave browser. If looking to use other browswers, the brave_path variable needs to be set to the exe location of the other browser that is to be used. If using firefox then seleniums chromeOptions needs to be replaced with the firefox specific code options.

MovieList.csv is provided as a psuedo database to run the two automated scraping scripts. If there is a desire to scrape more movies than provided, simply add films as they show up in rotten tomatoes url's to the list.

A data cleaning script has been included that removes languages other than English, casts all letters to lower case, removes punctuation and extranuous carriage returns, removes all special characters, and removes urls and html tags.
