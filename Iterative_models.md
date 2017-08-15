# Iteratives models

What I found is that the very small number of in the original training set lead to a surpopulation of false positives in the first classification.
This is a method to add more positive cases to the training data.

After the classification with gentiMap_allImages.py, extract the positives using the to_positives.py script: res2pos_multi() and then multipos2singlepos()

With these positives extracted, convert them to paths readable by the review interface using positives_to_path.py: positives2couples(saveimages=True, review2retrain=False)

Then, use the review interface, accessible using a web browser with the URL http://10.69.81.188/review2/ (see web_interfaces section) to asses if the detected positives are whether true or false positives

Then, convert the resulting review_results.csv file to a .txt file, and delete the first row.

Convert it to a readable format using positives_to_path.py: positives2couples(saveimages=False, review2retrain=True)

Append the resulting retrain_new.txt to retrain.txt. Remove the duplicates using http://textmechanic.com/text-tools/basic-text-tools/remove-duplicate-lines/

Feed the new retrain.txt file to a new model using gentriNetConvServer2.py
