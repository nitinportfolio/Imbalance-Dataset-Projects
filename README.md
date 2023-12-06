<h1>About Imbalanced Data </h1>
# Quick Notes
An **imbalanced dataset** refers to a situation in machine learning where the distribution of classes or categories in the dataset is not uniform. This means that one class has significantly more instances (data points) than the others. For example, in a binary classification problem where you're predicting whether an email is spam or not, if you have 90% non-spam emails and only 10% spam emails, the dataset is imbalanced. Datasets can have more than one minority class. Typical ratio is 1:10 and can be 1:100

This can pose challenges during model training because the learning algorithm might be biased towards the majority class, leading to poor performance in predicting the minority class. Models trained on imbalanced datasets can have lower accuracy, precision, recall, or F1-score for the minority class due to their lower representation in the data.

* Samples from the minority class are more often misclassified. *

Domains where we are more interesed in minority classes - 
1. Fraud detection
2. Medical Diagnosis
3. Equipment manufacturing
4. Network Intrusion Detection



## Why its tough to find out minority class
1. imbalance ration = X-minority/X-majority
2. Lesser the dataset or particular class more the error in the model. Finding & learning pattern is tough.
3. A. Small sample size
   B. Class sepearability - minority & majority classes are overlapping or very close to each.other. Rules or boundaries are not easy to find.
   If minority and majority class are linearly seprable then model is not very sensitive to any amount of imbalance.
   C. Within-classs sub-clusters - If minority class has different labels which is again imbalance then it also increases the complexity.

   ## Solutions to imbalanced datasets
   1. Data Level -
    a. Under Sampling
    b. Over Sampling
    - Changing distribution of the data
    - Random OVer or Under sampling
    - Creating new synthetic data
    - Removing noise or removing easy osbservations to classify

   2. Cost Sensitivie
    a. Higher Miss-classification costs - Penalising more to missclassified minority class
    - SHould apply different costs to diffrent errors

   3. Ensemble Algorithms
    a. Boosting & Bagging
    b. With Sampling
    - Combine weak learners.

## Metrices
### Accuracy  - 
1. Accuracy is not the best metric for imbalanced dataset as it does not distinguish between the numbers of correctly classified examples of different classes. 
2. Minority class has very little impact on the accuracy as compared to that of majority class.
3. It looks only at true positives and true negatives. It focuses on what you have correctly predicted. N doesnot focus on what yo uahve not predicted correctly. Hence Precision and Recall.

Eg.Say in the dataset i have ratio - 1:100
1 - Minority - fraud
100 - Majority - Not fraud
if my model always say not fraud then the acuracy of my model will be
99/100 that is 99%. But this is not correct as it classify minority class that is fraud transaction also as not fraud. Though am more interested in the minority class. Also by looking only at the accuracy I can't tell how many of my correctly classify are from the minority classs.

Dealing with imbalanced datasets in machine learning involves several strategies to improve model performance. The best approach often depends on the specific problem and the dataset. Here are some common techniques:

1. **Resampling Methods:**
   - **Over-sampling:** Increase the number of instances in the minority class by duplicating or generating synthetic samples (e.g., SMOTE - Synthetic Minority Over-sampling Technique).
   - **Under-sampling:** Decrease the number of instances in the majority class by removing samples randomly or using more sophisticated techniques to select representative samples.

2. **Different Performance Metrics:**
   - Instead of accuracy, use evaluation metrics that are better suited for imbalanced datasets. Metrics like precision, recall, F1-score, or area under the ROC curve (AUC-ROC) provide a better understanding of model performance across classes.

3. **Class Weighting:**
   - Assign different weights to different classes in the model training process. Algorithms like decision trees or SVMs often allow for adjusting class weights to penalize misclassifications in the minority class more.

4. **Generate Synthetic Data:**
   - Techniques like generative adversarial networks (GANs) or other data generation methods can create synthetic data for the minority class, balancing the dataset.

5. **Ensemble Methods:**
   - Ensemble techniques like bagging, boosting, or stacking can be helpful. Algorithms like Random Forest or Gradient Boosting inherently handle imbalanced data better than some others.

6. **Use of Anomaly Detection Techniques:**
   - For extremely imbalanced datasets where the minority class represents anomalies or rare events, anomaly detection methods like Isolation Forest or One-Class SVM can be effective.

7. **Cost-Sensitive Learning:**
   - Modify the learning algorithm to take misclassification costs into account during training, penalizing errors in the minority class more.

8. **Collect More Data:**
   - If feasible, collecting more data, especially for the minority class, can significantly improve model performance.

The choice of method depends on the nature of the problem, the size of the dataset, and the available computational resources. Often, a combination of these techniques or careful experimentation is necessary to determine the most effective approach for a particular imbalanced dataset.

### Precision - The number of positives that you have predicted, that are actually positives
### Recall - The number of positives you have correctly predicted, out of the total number of positives.
### F1 Score

## Choice of Threshold - 
 - In classification problems our model predict probabilites of each class. FOr balanced dataset we generally have threshold of 0.5. But for imbalanced dataset we can have decide other threshold values for better recall & precision & F1.

 ## Undersampling -  Deleting samples from the majority calss
 ## Oversampling - Duplicating samples from the miority class
 Both are not the best way.

 ## OverSampling - 
1. SMOTE - The majority class is over sampled by creating synthetic examples instead of extracting data at random. It prevents duplication. Uses KNN technique for creating synthetic examples. 
For each observations it will create new samples based on how many to be created, using KNN. New samples = Original sample - factor*(original samples - neighbour)
##############################################
##############################################
## The "best" approach to handle imbalanced datasets varies based on several factors:

1. **Nature of Data:** Understanding the specific characteristics of your dataset is crucial. For instance, in some cases, over-sampling might work better, while in others, under-sampling or a combination of both might be more effective.

2. **Class Imbalance Severity:** The degree of class imbalance also matters. For extreme imbalances, more sophisticated techniques like generating synthetic data or anomaly detection might be more suitable.

3. **Algorithm Sensitivity:** Different algorithms handle imbalanced data differently. For instance, ensemble methods like Random Forest or Gradient Boosting tend to perform well with imbalanced data inherently.

4. **Computational Resources:** Some methods might be more computationally intensive than others. For instance, generating synthetic data using algorithms like SMOTE might require more resources, especially for large datasets.

5. **Evaluation Metrics:** The choice of evaluation metrics matters. If you're more concerned about correctly identifying rare cases (higher recall), certain techniques might be more favorable.

6. **Domain Knowledge:** Understanding the domain and the importance of each class can guide the selection of the most appropriate technique.

In practice, a combination of approaches might often yield the best results. It's usually recommended to try multiple techniques, possibly through cross-validation, and evaluate their performance using appropriate metrics before deciding on the best approach for a particular problem and dataset.

Additionally, the "best" approach can evolve through experimentation and iterative improvement, where you adjust techniques based on the observed model performance and domain-specific considerations.


#####################################
#####################################

There is no single "best" approach for dealing with imbalanced datasets in machine learning, as the most effective method depends on the specific characteristics of the dataset and the machine learning task. However, some general guidelines can help you choose the most appropriate technique for your situation:

If the minority class is very small, undersampling may not be a viable option, as it could lead to overfitting. In these cases, oversampling or synthetic minority oversampling techniques (SMOTE) may be more appropriate.

If the minority class is noisy or has a lot of outliers, oversampling techniques that generate new samples may introduce more noise and worsen the model's performance. In these cases, undersampling or cost-sensitive learning may be better choices.

If the minority class is not well-defined or there is a lot of overlap between the classes, metric selection may be important to ensure that the model is not evaluated solely on its accuracy on the majority class.

If the dataset is very large, resampling techniques may be computationally expensive. In these cases, algorithm choice or ensemble methods may be more practical.

Ultimately, the best way to choose the most appropriate approach is to experiment with different techniques and evaluate their performance on your dataset. You can use cross-validation to compare the performance of different models and metrics to select the best approach for your specific task.

Dealing with imbalanced datasets in machine learning is a crucial aspect of ensuring accurate and reliable classification models. When the distribution of classes in a dataset is uneven, traditional machine learning algorithms tend to favor the majority class, leading to poor performance on the minority class. This can be particularly problematic in domains where the minority class represents a critical outcome, such as fraud detection or medical diagnosis.

To address the challenge of imbalanced datasets, several techniques have been developed to balance the class distribution and improve classification accuracy for the minority class. Here are some of the most common and effective methods:

1. **Resampling:** Resampling techniques involve modifying the dataset to make the class distribution more balanced. This can be achieved through two approaches:

    **Undersampling:** Undersampling involves randomly removing samples from the majority class until the number of samples in each class is equal.

    **Oversampling:** Oversampling involves generating new samples for the minority class. This can be done through various methods, such as random replication, synthetic minority oversampling technique (SMOTE), or adaptive synthetic sampling (ADASYN).

2. **Cost-Sensitive Learning:** Cost-sensitive learning algorithms assign different misclassification costs to different classes. This approach penalizes the model more heavily for misclassifying minority class instances, forcing the model to focus on improving its performance on the minority class.

3. **Metric Selection:** Traditional evaluation metrics, such as accuracy, can be misleading when dealing with imbalanced datasets. Instead, metrics like precision, recall, and F1-score provide a more accurate assessment of the model's performance on both majority and minority classes.

4. **Algorithm Choice:** Certain machine learning algorithms are inherently better suited for handling imbalanced datasets. For instance, algorithms like random forests and support vector machines (SVMs) are less sensitive to class imbalance than algorithms like logistic regression.

5. **Ensemble Methods:** Ensemble methods combine multiple models to improve overall performance. By combining models trained on different subsets of the data or different algorithms, ensemble methods can mitigate the bias towards the majority class.

The choice of the most appropriate technique depends on the specific characteristics of the dataset and the machine learning task. It is often beneficial to combine multiple techniques to achieve the best results.
