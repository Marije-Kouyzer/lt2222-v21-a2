# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Marije Kouyzer

*Answer all questions in the notebook here.  You should also write whatever high-level documentation you feel you need to here.*
Part 1: I return the data as a list of tuples.

Part 2: I use just the plain words as features. If the named entity is near the end of the sentence one or multiple end symbols ('<e>') will be feautures, depending on how many words are after the named entity in the sentence still. If it is near the start of the sentence, the same is true with start symbols ('<s>'). This way there are always 10 features, 5 from before the named entity and 5 from after the named entity.

Part 5: The confusion matrix from the training data seems to have a bit higher numbers in the top left to bottom right diagonal than the confusion matrix from the test data. These numbers represent the true positives for each class. It makes sense that the training data has more true positives than the test data since the model was trained on the training data.