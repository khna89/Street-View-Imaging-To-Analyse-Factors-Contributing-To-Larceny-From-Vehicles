# Street View Imaging To Analyse Factors Contributing To Larceny From Vehicles
In order to explore the opportunities explainable Deep Learning offers for social research, code in this repository aims at predicting socially relevant target - crime occurrence - based on the Street View Images (SVI). The code uses publicly available datasets and is a preliminary step for the research on the relations between mental health and environment in the Dutch context that I conduct for my thesis. My thesis supervisor is dr. Giacomo Spigler, and his advice and insight influenced the work greatly, however, he shouldn't be held responsible for any inaccuracies in this repository as my thesis project is not finished yet, not graded by dr. Spigler and also goes beyond the scope of this repository.

![image](https://user-images.githubusercontent.com/78618639/232754881-058124e8-c86d-4101-9e9a-6bd5192e02ea.png)

First, the San Francisco Police Department (SFPD) Incident Report dataset was preprocessed and analysed. The dataset is available for download on the San Francisco OpenData website: https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783 . Exploratory data analysis was conducted. 

![image](https://user-images.githubusercontent.com/78618639/232755080-10c851bd-2b6a-4405-a3bd-a29ce891024a.png)

![image](https://user-images.githubusercontent.com/78618639/232755120-484ddcc5-7986-4222-9389-fe32bc78f6fc.png)

![image](https://user-images.githubusercontent.com/78618639/232754534-1ffcee10-8cef-4b03-948c-f9c49ca4fefe.png)

After the exploration, the subcategory "Larceny - From Vehicle" was chosen as "crime" for the further exploration (please see the details in the Jupyter notebook). 

Street view images (SVI) were collected from City Street View Dataset, publicly available here: https://www.kaggle.com/datasets/stelath/city-street-view-dataset . All the images belong to Google Street View. After pre-processing (transforming coordinates, selecting only the images from San Francisco), the data was juxtaposed with the "crime" data in order to find the images in the viscinity of which the crime has occurred. This allowed to produce labels to the images.

The Facebook DeiT family of models were chosen to learn the relationship between the visual cues and crime because of its benchmark performance and because a readily-available explainability visualization pipeline was found that's developed for this model. 

For hyperparameter selection a function was created to load a given type of DeiT (pre-trained), freeze the pre-trained layers, un-freeze a given number of layers and train the model for 15 epochs with a given optimizer and learning rate. 

![image](https://user-images.githubusercontent.com/78618639/235307920-ac6680ae-a183-405f-b0da-101e7837ccd9.png)

Because of the dataset being imbalanced with the minority class constituting around 25% of the data, the F1 measure with macro-averaging was selected as the most accurate measure of performance. 

The performance on given hyperparameters was first monitored by the means of print statements.

![image](https://user-images.githubusercontent.com/78618639/235293660-74595f7b-54c0-41c8-8ee1-12affbe52702.png)

However, at a certain point it became apparent that storing the main metrics with some estimate of their stability is more practical. The build-train function was modified to allow for that. Keep in mind: the "stability" measurements in this table are SDs, so actually these should be called "unstability". 

![image](https://user-images.githubusercontent.com/78618639/235293713-2c35fcc8-0d2b-4ad2-8eff-b97c0616920a.png)

After the fine-tuning was finished, a combined metric was created by adding a normalized F1 progress and a reverse of the normalized SD of F1 progress. 

![image](https://user-images.githubusercontent.com/78618639/235293730-e56880a1-bd2d-4807-8d2f-bda6326fd815.png)

SD has limitations as a stability measurement (the more model managed to learn - the bigger the SD, even though the learning could have been stable; however, in many cases it was a good measure, because the models with high SD mostly were the ones that oscillated a lot). Because of the limitations, the manual examination of the dynamics of learning of particular models was also taken into account. 

The Deit Small with 5 unfrozen layers, trained with the SGD as optimizer and with learning rate of 0.001, showed the best performance and was trained on all the data, first without, and then with regularization (weight_decay = 1e-4). A number of best-performing models (according to observations and according to the combined metric) were also trained, but did not reach the level of performance of this chosen model. The F1 of the final model on the test set is 0.57.

The model's accuracy and the 'crime'/'no crime' images could also be explored visually:

![image](https://user-images.githubusercontent.com/78618639/235293767-1babba4b-186c-4450-8566-ad57d5b45bf5.png)

After training, testing and manual exploration of accuracy, attention heat maps were visualized with the code adapted from Gildenblat, J. (2020). Exploring Explainability for Vision Transformers. https://jacobgil.github.io/deeplearning/vision-transformer-explainability . 

From 300 random images from the test set (so the ones unseen to the model) the ones for which the model's predictions are correct were chosen, and they were ranked according to the absolute value of the model's certainty about the label (absolute of logits). The heat maps of the weights of the attention heads were visualized for these images, about which the model was certain it outputs the right prediction (and it in fact did).

![image](https://user-images.githubusercontent.com/78618639/235305704-601295b7-d15f-44c8-af6d-4df67fbfd4a2.png)

![image](https://user-images.githubusercontent.com/78618639/235305838-cc8760fc-ff47-4be2-85ea-af838a195a2a.png)

There don't seem to be some obvious features in the images that would systematically point to 'crime' occurrence. 

The project has a number of limitations:
- the publicly available data contains indoor images, which have nothing to do with the urban landscape, and because there are quite a lot of them, they create a lot of noise
- the images were randomly cropped during transformation for the train set. This allowed the model to see the objects on the images in detail, at the same time it reduced a lot of scenes to particular objects (or walls, or sky areas) instead of landscapes, which might have limited the ability of the model to learn the general landscape-related features
- only a transformer-architecture was used. Using common benchmark CNNs would be the next step
- the targed variable was represented in a binary way based on the availability of a chosen crime subtype record in the radius of 30 meters from the street view image. However, a gradual change in the likelihood of crime would be a better measure. 

The project is a preliminary step for an exploration of another socially-relevant variable on the curated dataset in the Dutch context. However, it was a nice preparation step and it gave a lot of insights into the aspects that need extra attention. 

The project is not directly based on some particular prior research, but it is inspired by a vast tradition of the usage of Deep Learning in urban and health studies. For good inspiring examples refer to:

Biljecki, F., & Ito, K. (2021). Street view imagery in urban analytics and GIS: A review. Landscape
and Urban Planning, 215, 104217.

Helbich, M., Yao, Y., Liu, Y., Zhang, J., Liu, P., & Wang, R. (2019). Using deep learning to examine
street view green and blue spaces and their associations with geriatric depression in Beijing, China.
Environment international, 126, 107-117.

Ki, D., & Lee, S. (2021). Analyzing the effects of Green View Index of neighborhood streets on
walking time using Google Street View and deep learning. Landscape and Urban Planning, 205,
103920.

Li, X., Zhang, C., Li, W., Kuzovkina, Y. A., & Weiner, D. (2015). Who lives in greener
neighborhoods? The distribution of street greenery and its association with residents’socioeconomic
conditions in Hartford, Connecticut, USA. Urban Forestry & Urban Greening, 14(4), 751-759.

Ma, R., Wang, W., Zhang, F., Shim, K., & Ratti, C. (2019). Typeface reveals spatial economical
patterns. Scientific Reports, 9(1), 15946.

Rzotkiewicz, A., Pearson, A. L., Dougherty, B. V., Shortridge, A., & Wilson, N. (2018). Systematic
review of the use of Google Street View in health research: Major themes, strengths, weaknesses and
possibilities for future research. Health & place, 52, 240-246.

etc.
