# Predicting At-Risk Students to Improve High School Learning Delivery and Student-Teacher Dynamics

**GitHub repository: [link](https://github.com/MichaelAngelo-11/student-success-prediction.git)**

**Google Drive folder : [link](https://drive.google.com/drive/folders/1ojkiGGAU7tgQeX_CGRHMsx8FlqvSKIkI?usp=sharing)**

**Presentation video : [link](https://drive.google.com/drive/folders/1TcCDP8o3NUV86_50BHTZ_5IPUCJ5wPbN?usp=sharing)**

# **Abstract**

To improve the quality of the education received in our high schools, we need to become aware of how many students are having difficulties learning that content, and intervene in that learning before it becomes set in stone. I design and test an integrated machine-learning system for predicting at-risk students based on demographic, registration, assessment and learning platform engagement data. I have been using the Open University Learning Analytics Dataset (OULAD), which contains data on more than 32,000 learner-course records and over 10.6 million raw virtual learning environment (VLE) interaction logs, and I have formulated the problem as binary classification between "at risk" (pass-fail or withdraw) and "on track" (pass or distinction) classes. My models are all trained with the same split of the data into 70/15/15 train/validation/test, and I train four versions of a neural network using TensorFlow/Keras, using four different configurations, as well as three traditional machine learning models (Logistic Regression, Random Forest, Gradient Boosting). My best traditional model (Gradient Boosting) is able to achieve an F1 held out test score of 0.923 and an ROC-AUC of 0.977, while my best neural network is able to achieve an F1 held out test score of 0.918 and an ROC-AUC of 0.973. The findings from my feature analysis support my overall mission: signal to flag students for a change in learning delivery and more active learning student-teacher support signals before they fail or become completely disengaged, using similar signals as noted above: participation, assignment submission timeliness, platform/classroom engagement.

# **1\. Introduction / Problem Statement**

## **1.1 Mission context**

The mission that inspired this project is to make the educational experience in high school better by changing the experience of learning – how we deliver it and how we experience it – by changing the relationship between students and teachers for the better – by supporting students instead of allowing them to fail without hearing about it. I know that a prerequisite to any of these is information: if the teacher or school doesn't know early enough, who is struggling and why, they can't personalise instruction or approach a struggling student. The goal of educational data mining (EDM) and learning analytics is to capture the digital footprints students leave behind (such as logins, clicks, submissions, scores, etc.) to make inferences about their risk status and to initiate interventions before the outcome occurs (disengagement, failure, withdrawal) is realized \[1\] \[2\].

## **1.2 Why I use a proxy dataset**

I was unable to find a large enough (more than 20,000 rows) open, row-level, usable classification target on high-school student engagement. High-school student records are usually small, private, and limited to small-scale publishing. However, I employed the Open University Learning Analytics Dataset (OULAD) as an appropriate and well-established proxy \[3\]. I selected OULAD for this purpose, because it is one of the more widely used open datasets in LAMR, and because it has a very comprehensive daily record of the interaction with the VLE (over 10.6 million rows in my raw studentVle.csv file), alongside demographic and registration data, and the results of the assessments \[3\], \[4\]. There are multiple systematic reviews that confirm the routine use of OULAD for the three prediction problems that I aim to address in this project: Performance Prediction, At-risk/Dropout Identification and Engagement Prediction \[4\]. Although OULAD refers to distance learners at a UK university and not to students in physical schools, I believe the logic is the same as that which motivates my mission: students who do less with the learning materials and submit fewer assessments, are measurably more likely to fail or drop out \[5, 6\] and the same applies to school-level, attendance, activity and submission, dropout early-warning systems \[7, 8\].

## **1.3 Classification problem**

I frame this project as binary classification:

* Target \= 1 ("At risk"): the student's final\_result is Fail or Withdrawn.  
* Target \= 0 ("On track"): the student's final\_result is Pass or Distinction.

## **1.4 My objectives**

1. Build a complete, reproducible machine learning pipeline, from raw data to saved models, that predicts at-risk students from demographic, registration, assessment, and engagement features.  
2. Compare multiple traditional ML algorithms and multiple neural-network configurations under one consistent evaluation protocol (seven systematically varied experiments in total).  
3. Use my empirical results (which features matter, how big the errors are, where the model fails) to produce concrete, evidence-based recommendations for changing learning delivery and student-teacher dynamics in a high-school context.

# **2\. Literature Review**

## **2.1 Educational data mining and learning analytics on OULAD**

The open data set OULAD is introduced by Kuzilek et al., 2022 \[3\] as a dataset specifically created for research in machine learning on student behaviour and outcomes in virtual learning environments. Since it was released, I found that OULAD has become a benchmark dataset: in my review, I found that in the last 7 years (2017-2024), a large number of studies used OULAD as a data set, three of which are always the same, predicting student performance, identifying at risk students and predicting engagement, and they applied various machine learning, deep learning and statistical methods \[4\]. Additionally, I was able to find studies where only VLE-based performance prediction was performed and the results showed that behavioural engagement features extracted from the VLE logs have a significant impact on the quality of prediction when compared with demographic data \[5\]. As my own project is based on a similar scale of data (32,593 records and more than ten million VLE interaction logs) my study of how to use heterogeneous, class-imbalanced kinds of educational data for reliable prediction is motivated by the specific findings of a recent comparative study using logistic regression, SVM, random forest and XGBoost on the full OULAD dataset \[6\]. Other studies focused on OULAD applications showed that ensemble Tree architectures like Random Forest and optimised Decision Trees yielded a good balance of accuracy and recall compared to more complex deep networks (CNN, RNN, LSTM), and this result was similarly found in my own research (Gradient Boosting was able to outperform my neural network by a very small amount).

## **2.2 Engagement and performance prediction more broadly**

In addition to OULAD, many studies have been conducted on LMS data, consistently revealing that LMS metrics of behavioural engagement predict course outcomes in a variety of course settings, including a study of 17 blended Moodle courses that found LMS engagement variables to be generalizable and useful predictors of course performance across diverse course designs \[10\]. This helped me make the design decision to take raw VLE logs and sum them up as indicators of engagement (total clicks, number of interactions, number of distinct active days, number of distinct materials accessed) rather than on the basis of demographics.

## **2.3 Early-warning systems and dropout prediction**

Applied At-risk prediction research is called early-warning systems (EWS). I learned about the important impact of the early-warning system built and validated by Akcapinar et al., to identify at-risk students for intervention, which emphasized that the purpose of prediction is not the model itself but rather the intervention it allows \[11\]. I discovered that at the school level, not university-level, Lee and Chung have developed a random-forest based dropout early-warning system based on student attendance and activity data for more than 165,000 students from South Korea's high schools, writing specifically about the extreme class imbalance that is generally found with dropout/failure labels (dropouts form a small minority of all students) \[7, 8\]. Several other EWS studies, like they did, indicated that the students more likely to drop-out had poor attendance, were low in activity and were low in participation in school activities; I see all of these indicators of drop-out risk in VLE participation and assessment submission here as a single signal \[8\]. The findings that I have drawn from my OULAD research support my generalisation of the findings into an in-person high-school setting: the type of signal that predicts risk is similar across very different institutional settings.

## **2.4 Class imbalance in educational prediction**

I stated that nearly always, at-risk/ drop out are the minority class and multiple studies indicate that classifiers trained without any explicit focus on improving the accuracy of the minority class tend to be biased toward the majority class ("on track"/ "graduate") unless some effort is made to mitigate this effect \[7\] \[11\]. Common practices to reduce this effect are class-weighting (used in Logistic Regression and Random Forest in this project) and resampling methods like SMOTE, which are reported to be able to meaningfully increase minority-class recall and F1-score when applied carefully to the training split only, to prevent leaking data into the test set \[11\], \[12\]. The same leakage-avoidance principle applies to my own pipeline: everything I do on the train split, from preprocessing (imputation, encoding, scaling) to final evaluation, I don't touch the test set, unless it's a final evaluation\!

## **2.5 Teacher-student relationships and intervention**

Changing the student-teacher dynamic is a 2nd lever which is well supported in the educational-psychology literature. A meta-analysis by Roorda et al. indicated that poorly formed teacher-student relationships (e.g., conflict) are a significant and measurable predictor of lower academic outcomes in primary and secondary schools, whereas positive teacher-student relationships (e.g., closeness) are a positive predictor of improved achievement at both levels \[13\]. In other longitudinal work reviewed, the link between teacher-student relationship quality and student engagement and achievement persists over time, and is particularly influential for students already at risk and/or low-performing students \[14, 15\]. This literature is what makes me feel good about combining my data-driven risk-prediction system with a relational intervention (teacher outreach, mentoring) instead of just an automated intervention \- the literature on relationships complements the literature on prediction \- so there's both the “who” (from the model) and there's the “what to do” (from the literature, strengthen the relationship, not just flag the data point).

## **2.6 Personalised learning delivery**

Finally, the first lever of my mission the delivery of learning  is backed up by research in the field of personalised and adaptive learning. According to this scoping review of personalised adaptive learning systems in higher education, the majority of the reviewed studies reported improved academic outcomes and/or increased student engagement after implementing adaptive, personalised approaches to teaching and learning, albeit not in all cases and not across all types of interventions, and largely dependent on the quality of the implementation \[16\]. In another paper that I researched in the area of technology for personalised learning, I found that the same results: approaches that truly personalise content and feedback produce significantly more impact than approaches that only superficially personalise content and feedback \[17\]. With this risk-prediction evidence to go with that, I believe the mission strategy should be a two pronged approach: to identify through data who needs a different experience, and to provide a validated personalised-delivery and relationship-centred strategy to change that experience.

# **3\. Methodology**

## **3.1 Data**

I combined 6 OULAD tables into one per-student-per-course-presentation modelling table: studentInfo.csv ( \~ 32,600 rows, containing demographics and final result), courses.csv ( \~ 32,600 rows, module presentation length), studentRegistration.csv ( \~ 32,600 rows, registration date), assessments.csv ( \~ 173,000 rows, assessment metadata), studentAssessment.csv ( over 10.6 million raw rows, which I processed in 2-million-row chunks to avoid memory problems before summarising per student) and studentVle.csv ( over 10.6 million raw rows, daily VLE interaction logs).

## **3.2 Feature engineering**

* VLE engagement features (aggregated from studentVle.csv): total clicks, number of interaction records, number of distinct active days, number of distinct VLE materials/sites accessed.  
* Assessment performance features (aggregated from studentAssessment.csv merged with assessments.csv): average score, number of submissions, average days submitted before/after the due date.  
* Demographic and administrative features (from studentInfo.csv, studentRegistration.csv, courses.csv): gender, region, highest prior education, deprivation band (imd\_band), age band, number of previous attempts, studied credits, disability status, registration date, module presentation length.

## **3.3 Preprocessing**

Since those students who were inactive in VLE or did not submit any assessments were marked as zeroes instead of being imputed statistically, since being "never engaged" and "never submitted" is itself a signal of risk. The "imd\_band" feature, which contained missing values was treated as an extra "unknown" value. All the categorical features were one-hot encoded and all numerical features were median-imputed and scaled using a ColumnTransformer.

## **3.4 Train/validation/test split**

I employed a stratified 70/15/15 split so as to maintain the ratio of at-risk students in all three groups, and left out the test group completely until the final evaluation stage.

## **3.5 Models and experiments**

I ran seven systematically varied experiments with a consistent split and evaluation protocol:

| Run | Model | Experiment |
| :---- | :---- | :---- |
| **1** | Logistic Regression | Baseline (class\_weight="balanced") |
| **2** | Random Forest | Baseline (300 trees, class\_weight="balanced") |
| **3** | Gradient Boosting | Baseline |
| **4** | Neural Network | Baseline architecture (64 to 32 dense, Adam, lr=1e-3) |
| **5** | Neural Network | Lower learning rate (lr=1e-4) |
| **6** | Neural Network | With dropout (0.3) |
| **7** | Neural Network | Larger layers / bigger batch (128 to 64 to 32, batch=512) |

The Keras Functional API was used for training my model, using a tf.data.Dataset pipeline (shuffle/batch/prefetch). The model was trained using the Adam Optimizer and Binary Cross Entropy Loss. Early stopping was used to prevent overfitting by using patience 4 on the validation loss.

## **3.6 Evaluation metrics**

Due to an imbalance in my target variable, I found it necessary that accuracy was not going to be enough. I computed metrics such as accuracy, precision, recall, F1 Score, and AUC ROC for every experiment performed on the validation set. In addition, I tested the top two models based on F1 Score on the unseen test set.

# **4\. Experiments and Results**

## **4.1 Validation-set experiment table (all seven runs)**

| Run | Model | Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **1** | Logistic Regression | Baseline | 0.898 | 0.934 | 0.869 | 0.900 | 0.958 |
| **2** | Random Forest | Baseline | 0.920 | 0.961 | 0.886 | 0.922 | 0.975 |
| **3** | Gradient Boosting | Baseline | 0.922 | 0.959 | 0.890 | 0.924 | 0.978 |
| **4** | Neural Network | Baseline architecture | 0.919 | 0.950 | 0.893 | 0.921 | 0.974 |
| **5** | Neural Network | Lower learning rate | 0.904 | 0.942 | 0.872 | 0.906 | 0.966 |
| **6** | Neural Network | With dropout | 0.919 | 0.954 | 0.890 | 0.921 | 0.975 |
| **7** | Neural Network | Larger layers / bigger batch | 0.921 | 0.957 | 0.889 | 0.922 | 0.973 |

Gradient Boosting (Run 3\) gave me the highest validation F1/ROC-AUC score in all my classical algorithms, while a bigger layer size in the neural network (Run 7\) gave me the best performing deep learning model. Dropping the learning rate (Run 5\) definitely performed worse, and I would say it is under-training because of my fixed number of epochs, whereas dropping (Run 6\) had hardly any effect on the regularization of this particular dataset compared to the baseline model.

## **4.2 Held-out test-set results**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Gradient Boosting (best traditional)** | 0.921 | 0.957 | 0.892 | 0.923 | 0.977 |
| **Neural Network \- larger layers / bigger batch (best deep learning)** | 0.917 | 0.948 | 0.891 | 0.918 | 0.973 |

Since my validation and test results are quite similar for both models (e.g., the F1 score for Gradient Boosting reduces only from 0.924 to 0.923, ROC-AUC from 0.978 to 0.977), I am confident about the good generalisability of the models and reliability of my validation in estimating the test result. My recommendation is Gradient Boosting as my overall model because, although it does not have a higher F1 and ROC-AUC compared to my best neural network, it is much faster to train (few seconds versus seconds/minutes for the neural networks) and allows one to interpret the model through feature importances, which I find practically useful.

## **4.3 Visual diagnostics**

* figures/class\_balance.png shows the final\_result and binary at\_risk class distributions, confirming meaningful but not extreme class imbalance.  
* figures/eda\_boxplots.png shows that at-risk students have visibly lower total VLE clicks, lower average assessment scores, and fewer active days than on-track students, the core empirical pattern underlying my whole project.  
* figures/nn\_training\_curves.png shows training/validation loss and accuracy across my four neural-network configurations, which I used to check for overfitting and justify my early-stopping configuration.  
* figures/confusion\_matrices.png and figures/roc\_curves.png show the test-set confusion matrices and ROC curves for my two selected models, which I use for the error analysis in Section 5\.

# **5\. Error Analysis, Limitations and Discussion**

## **5.1 False positives vs. false negatives**

In this case, the two different error types will incur an unequal cost. The first error, namely a false positive, entails that a student who should not have been predicted as being at risk is given extra help or encouragement from his or her teacher. This error has low cost because no mentoring can be detrimental to a student who did not really require it. On the other hand, false negatives, where a student who should be predicted as being at risk ends up failing or withdrawing, are the costly errors since these are the students who will not have a different mode of learning delivered to them under my model and who may therefore become detached from classes or fail unnoticed. The confusion matrices (figures/confusion\_matrices.png) give the exact figures of all the four kinds of errors made in each model selected. Since the recall is the performance measure which best shows the influence of false negatives, and both my models yield around 0.89 recall for the at-risk class, I estimate that roughly 11% of students at risk are not caught by my model.

## **5.2 Overfitting**

Early stopping was used in each neural network experiment (tracking of validation loss, patience at 4 epochs, loading the best weights) precisely in order to avoid overfitting. As can be seen from the training and validation loss and accuracy charts (figure/nn\_training\_curves.png), my goal was to check if validation loss stops decreasing and starts increasing while training loss keeps decreasing; in my experiments, none of the four configurations demonstrate significant separation of training and validation charts, thus indicating to me that my choice of regularization and early stopping was appropriate enough for this set of features and dataset size.

## **5.3 Class imbalance**

The imbalanced nature of my binary dataset (more on track students than at-risk students) is expected, since previous literature about predicting dropouts points out imbalance as an almost universal characteristic of such a label \[7\], \[11\]. The issue was dealt with through class\_weight="balanced" parameter in the cases of Logistic Regression and Random Forest models and by using F1 and ROC-AUC metrics instead of accuracy, which gives a falsely inflated value of model performance due to choosing one class only (the majority one). The natural direction of extending the current experiment, which comes directly from literature overviewed in Section 2.4, is testing of SMOTE oversampling on the train subset alone.

## **5.4 Dataset limitations**

I acknowledge that OULAD dataset collects the distance/online-learning behaviors of one UK university, not the in-person high school. Some of the features I employed are peculiar to higher education learning process which would not apply: imd\_band and highest\_education are features pertaining to a UK adult/higher-education population; num\_of\_prev\_attempts and studied\_credits are characteristics of university modules enrolments instead of high-school grade behaviors; and "VLE clicks" are features indicating engagement with an online course portal instead of classroom attendance, participation and homework submission which are the more likely indicators a high school could be able to collect. I am sure that any real application of my method in a high school context would require revalidation using high-school attendance, participation and assessment data, using the same general method (engagement \+ submission behavior features, stratified split, class imbalance handling using class weighting or resampling, and F1 score/ROC-AUC metrics) but using features relevant to high schools, as done by previous school-based EWS studies (e.g., \[7\], \[8\]).

## **5.5 Connecting my results back to the mission**

The observation that is consistently made throughout each family of models that I have tested – linear (Logistic Regression), tree-based ensemble (Random Forest, Gradient Boosting) and neural network – is that measures of engagement intensity and assessment submission are the most effective predictors of being an at-risk student, as compared to demographic variables. I consider this to be precisely the kind of evidence my mission requires – proof in the form of data-driven results that show the effectiveness of shifting the student's engagement with educational material and identification of low submission behaviour as means to improve outcomes. Together with literature on personalization of learning delivery \[16\], \[17\] and the quality of the teacher-student relationship \[13\]-\[15\], I believe it can lead to a definite two-step approach that could be implemented in high school setting: (1) development of a simple dashboard based on the same pipeline as the one presented in the current project that would help to identify at-risk students early; and (2) implementation of relational intervention (teacher check-up, mentoring etc.) together with each flag of at-risk students found.

# **6\. Conclusion and Recommendations**

For this project, I created an end-to-end, replicable machine learning pipeline that included data integration of over 10.6 million raw interactions, feature creation, three traditional machine learning models, four neural network architectures, and strict held-out testing to predict at-risk students based on their interactions and performance. My best model, Gradient Boosting, gave me a test F1-score of 0.923 and an ROC-AUC score of 0.977, slightly better than my best neural network (F1-score of 0.918, ROC-AUC score of 0.973), so I suggest using it for this problem.

**My recommendations for my high-school mission:**

4. Instrument learning delivery for engagement signals. Whatever platform or classroom routine a school uses, I recommend capturing participation/engagement and assignment-submission data systematically, since my project shows these are the strongest predictors of risk, consistent with prior school-level dropout research \[7\], \[8\].  
5. Build an early-warning dashboard, not just a one-off model. Following the early-warning-system literature \[2\], \[7\], I believe the value of prediction lies in triggering timely intervention, not in the model's accuracy alone.  
6. Pair every flag with a relational intervention. Since teacher-student relationship quality independently predicts achievement and engagement \[13\]-\[15\], I recommend that a flagged student be met with personal teacher outreach, not an automated message.  
7. Favor interpretable models when stakes involve human judgement. I chose Gradient Boosting partly because its feature importances make it easier for teachers and counsellors to understand why a student was flagged, supporting trust and actionable follow-up, consistent with prior findings that tree ensembles offer a strong accuracy/interpretability balance relative to deep models on this type of data \[9\].  
8. Validate on real high-school data before deployment, given the dataset limitations I discuss in Section 5.4.

# **References**

\[1\] T. Chaka, "Educational data mining, student academic performance prediction, prediction methods, algorithms and tools: An overview of reviews," 2021\.

\[2\] G. Akcapinar, A. Altun, and P. Askar, "Using learning analytics to develop early-warning system for at-risk students," International Journal of Educational Technology in Higher Education, vol. 16, no. 1, p. 40, 2019\.

\[3\] J. Kuzilek, M. Hlosta, and Z. Zdrahal, "Open University Learning Analytics dataset," Scientific Data, vol. 4, no. 170171, 2017\.

\[4\] L. Jin, Y. Wang, H. Song, and H.-J. So, "Predictive modelling with the Open University Learning Analytics dataset (OULAD): A systematic literature review," in Artificial Intelligence in Education: Posters and Late Breaking Results, Workshops and Tutorials, Industry and Innovation Tracks, Practitioners, Doctoral Consortium and Blue Sky, Cham, Switzerland: Springer Nature Switzerland, 2024, pp. 477-484.

\[5\] S. Hussain, N. A. Dahan, F. M. Ba-Alwi, and N. Ribata, "Educational data mining and analysis of students' academic performance using WEKA," Indonesian Journal of Electrical Engineering and Computer Science, vol. 9, no. 2, pp. 447-459, 2018\.

\[6\] A. Omarbekova, A. Nazyrova, G. Bekmanova, Z. Lamasheva, R. Turebayeva, and Z. Abzal, "Comparative evaluation of machine learning models for early prediction of student academic performance using the Open University Learning Analytics Dataset," International Journal of Interactive Mobile Technologies, vol. 20, no. 17, 2026\.

\[7\] S. Lee and J. Y. Chung, "The machine learning-based dropout early warning system for improving the performance of dropout prediction," Applied Sciences, vol. 9, no. 15, p. 3093, 2019\.

\[8\] J. Y. Chung and S. Lee, "Dropout early warning systems for high school students using machine learning," Children and Youth Services Review, vol. 96, pp. 346-353, 2019\.

\[9\] "OULAD MOOC student performance prediction using machine and deep learning," in Proc. Int. Conf. on Advances in Computing Research, Atlantis Press, 2024\.

\[10\] R. Conijn, C. Snijders, A. Kleingeld, and U. Matzat, "Predicting student performance from LMS data: A comparison of 17 blended courses using Moodle LMS," IEEE Transactions on Learning Technologies, vol. 10, no. 1, pp. 17-29, 2017\.

\[11\] S. Mduma, "Data balancing techniques for predicting student dropout using machine learning," Data, vol. 8, no. 3, p. 49, 2023\.

\[12\] "A PSO weighted ensemble framework with SMOTE balancing for student dropout prediction in smart education systems," Scientific Reports, vol. 15, 2025\.

\[13\] D. L. Roorda, H. M. Y. Koomen, J. L. Spilt, and F. J. Oort, "The influence of affective teacher-student relationships on students' school engagement and achievement: A meta-analytic approach," Review of Educational Research, vol. 81, no. 4, pp. 493-529, 2011\.

\[14\] C. Longobardi, M. Settanni, S. Lin, and M. A. Fabris, "Student-teacher relationship quality and prosocial behaviour: The mediating role of academic achievement and a positive attitude towards school," British Journal of Educational Psychology, vol. 91, no. 2, pp. 547-562, 2021\.

\[15\] "The longitudinal associations between teacher-student relationships and school outcomes in typical and vulnerable student populations: A systematic review," Social Psychology of Education, 2025\.

\[16\] "Personalized adaptive learning in higher education: A scoping review of key characteristics and impact on academic performance and engagement," 2024\.

\[17\] L. E. Major, S. Francis, and M. Tsapali, "The effectiveness of technology-supported personalised learning in low- and middle-income countries: A meta-analysis," British Journal of Educational Technology, vol. 52, no. 4, pp. 1935-1964, 2021\.