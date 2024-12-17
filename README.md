this model uses the data set from https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires

Model Selection and Interpretation
With my dataset selection of wildfire causes, I needed to pick a way to analyze a lot of information about past fires. After considering several options, I chose a method called Random Forest. I liked this approach because it's good at dealing with the messy, real-world data we often see in nature. It can handle situations where many factors interact to cause a fire. Another big plus is that this method tells us which information is the most helpful in predicting fire causes.
After running the analysis, I spent a lot of time trying to understand what it told us, I looked closely at which factors seemed to matter most in predicting whether a fire was started by people or by natural causes. I ended up producing a feature importance chart to help me make a better decision and back up my thoughts on what factors are most important. What I found was very interesting. Location plays a major role, as well as the time in which the fire was discovered. I don’t quite understand why discovery time could affect the predictions, but it is what it is. 
To make sure my method was working well, I checked it in several ways. I looked at how often it correctly identified the cause of a fire. I also checked if it was better at finding human-caused fires or natural fires. I also investigated how sure the method was about its decisions. This helped me understand not just when it was right or wrong but how confident it was in different situations.


![output](https://github.com/user-attachments/assets/a4755569-7cdf-45fe-aab0-cc241b53728e)
Natural Fires
Precision: 0.86 - When our method said a fire was natural, it was right 86% of the time.
Recall: 0.79 - Out of all the actual natural fires, our method correctly identified 79% of them.
F1-score: 0.82 - This is a balanced measure of precision and recall. A score of 0.82 is good, showing our method is fairly reliable for natural fires.
Support: 55,972 - This is the number of natural fires in our test data.
Man-made Fires
Precision: 0.96 - When our method said a fire was man-made, it was right 96% of the time.
Recall: 0.98 - Out of all the actual man-made fires, our method correctly identified 98% of them.
F1-score: 0.97 - This high score shows our method is very reliable for identifying man-made fires.
Support: 320,121 - This is the number of man-made fires in our test data.
Additional Information
Macro avg: This is the simple average of the scores for both types of fires. It doesn't account for the number of fires in each category.
Weighted avg: This average takes into account how many fires are in each category. Since there are many more man-made fires, this average is closer to the man-made scores.


![image](https://github.com/user-attachments/assets/6c9ef1be-5bed-4132-974f-80655d9d191d)
Each bar represents information we used to predict fire causes, like the time of year or the location of the fire. The longer the bar, the more important that information was in making predictions. This plot helps us see which factors matter most in determining if a fire was caused by humans or by nature. 

![image](https://github.com/user-attachments/assets/dfeda301-63ab-4ced-b0ec-c5e2032f9b43)
This plot shows how often our method correctly guessed the cause of a fire and how often it made mistakes. The boxes along one diagonal show correct guesses, while the others show mistakes. This helps us understand whether our method is better at identifying human-caused fires or natural fires. It also reveals whether it tends to make certain types of errors more often than others.


![image](https://github.com/user-attachments/assets/572be2b2-c2b6-4ed2-9c99-54835f7a5d93)
This is a curved line on a graph; the closer this line is to the top-left corner of the graph, the better our method will tell the difference between human-caused and natural fires. We also calculate the AUC (Area Under the Curve) from this plot. An AUC closer to 1 means our method is doing a great job, while a value closer to 0.5 suggests it's not much better than random guessing.


![image](https://github.com/user-attachments/assets/43f4db54-304b-4205-b400-a5f5f6691050)
This plot shows two overlapping hill-shaped curves. One represents human-caused fires, and the other represents natural fires. The amount of overlap between these "hills" tells us how well our method can distinguish between the two types of fires. Less overlap means our method is better at telling them apart. The shape and position of these curves also show us how confident our method is in its predictions for each type of fire.
