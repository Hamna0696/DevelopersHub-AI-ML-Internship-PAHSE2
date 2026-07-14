HAMNA SARFRAZ
DHC-2511
INTERNSHIP AI/ML
PHASE -2
Task 1: News Topic Classifier Using BERT
Objective
The objective of this project is to fine-tune the pre-trained BERT (bert-base-uncased) model for classifying news headlines into different categories. The AG News dataset is used for training and testing the model. After training, the model is evaluated using Accuracy and F1-score and deployed with Gradio for live predictions.
Tools and Technologies
•	Python 
•	Google Colab 
•	Hugging Face Transformers 
•	Hugging Face Datasets 
•	PyTorch 
•	Scikit-learn 
•	Evaluate 
•	Accelerate 
•	Gradio 
Dataset
The project uses the AG News Dataset available in the Hugging Face Datasets library. The dataset contains news headlines categorized into four classes.
Categories
•	World 
•	Sports 
•	Business 
•	Science/Technology 
The dataset is divided into training and testing sets.
Step 1: Install Required Libraries
The required Python libraries were installed using pip. These libraries provide support for loading datasets, using BERT, training the model, evaluating performance, and deploying the application.
Libraries installed:
•	transformers 
•	datasets 
•	evaluate 
•	accelerate 
•	scikit-learn 
•	gradio 
Step 2: Import Libraries
All required libraries were imported successfully. These libraries were used for preprocessing, model training, evaluation, and deployment.
Main imports include:
•	AutoTokenizer 
•	AutoModelForSequenceClassification 
•	Trainer 
•	TrainingArguments 
•	load_dataset 
•	pipeline 
Step 3: Load Dataset
The AG News dataset was loaded from the Hugging Face Datasets library. The dataset contains training and testing samples with news headlines and corresponding labels.
Step 4: Load Tokenizer
The bert-base-uncased tokenizer was loaded. The tokenizer converts text into tokens that can be processed by the BERT model.
Step 5: Tokenize Dataset
The dataset was tokenized using the BERT tokenizer.
During tokenization:
•	Headlines were converted into input IDs. 
•	Attention masks were generated. 
•	Maximum sequence length was set to 128. 
•	Long sequences were truncated. 
•	Short sequences were padded. 
Step 6: Rename Label Column
The original label column was renamed to labels because the Hugging Face Trainer requires the target column to be named "labels".
Step 7: Convert Dataset to PyTorch Format
The tokenized dataset was converted into PyTorch tensors using set_format().
The following columns were used:
•	input_ids 
•	attention_mask 
•	labels 
Step 8: Load Pre-trained BERT Model
The pre-trained bert-base-uncased model was loaded with four output classes.
The model architecture was automatically downloaded from Hugging Face.
Step 9: Define Evaluation Metrics
Two evaluation metrics were used.
Accuracy
Accuracy measures the percentage of correctly classified news headlines.
F1-score
The weighted F1-score evaluates the balance between Precision and Recall.
Step 10: Configure Training Arguments
Training arguments were configured before training.
Parameters used:
•	Learning Rate = 2e-5 
•	Epochs = 3 
•	Batch Size = 16 
•	Weight Decay = 0.01 
•	Evaluation Strategy = Epoch 
•	Save Strategy = Epoch 
These settings help improve the model's learning process while preventing overfitting.
Step 11: Create Trainer
The Hugging Face Trainer API was used to simplify model training.
The Trainer combines:
•	Model 
•	Training arguments 
•	Training dataset 
•	Testing dataset 
•	Evaluation metrics 
Step 12: Train the Model
The BERT model was fine-tuned on the AG News dataset.
During training:
•	Training loss was calculated. 
•	Validation loss was monitored. 
•	Model weights were updated after each batch. 
Google Colab GPU (T4) is recommended because CPU training is very slow.
Step 13: Evaluate the Model
After training, the model was evaluated on the testing dataset.
The evaluation produced:
•	Validation Loss 
•	Accuracy 
•	Weighted F1-score 
These metrics measure the overall classification performance.
Step 14: Save Model
The trained model and tokenizer were saved locally.
Saved files:
•	news_classifier/ 
•	tokenizer/ 
These files can be reused without retraining.
Step 15: Test the Model
The saved model was loaded using the Hugging Face pipeline.
Several custom news headlines were provided as input.
Example:
•	Pakistan wins Asia Cup 
•	Apple launches new AI chip 
•	Stock market falls sharply today 
The model predicted the correct category for each headline.
Step 16: Deploy with Gradio
A Gradio web interface was created.
The interface allows users to:
•	Enter a news headline. 
•	Click Predict. 
•	View the predicted news category instantly. 
This demonstrates a simple real-time deployment of the trained model.
Challenges Faced
During implementation, several issues were encountered:
•	Hugging Face dataset loading error (HFUriError). 
•	Package version conflicts between transformers and accelerate. 
•	WandB login prompt during training. 
•	CPU-only training caused slow execution. 
•	Runtime restarts cleared notebook variables. 
These issues were resolved by installing compatible package versions, restarting the runtime, and configuring the environment correctly.
Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API
Objective
The objective of this project is to build a reusable and production-ready machine learning pipeline using Scikit-learn's Pipeline API. The pipeline automates data preprocessing, model training, hyperparameter tuning, evaluation, and model export for predicting customer churn using the Telco Customer Churn dataset.
Dataset
Dataset Name: Telco Customer Churn Dataset
The dataset contains customer information such as gender, tenure, internet service, monthly charges, total charges, payment method, and contract type. The target variable is Churn, which indicates whether a customer left the company or not.
Target Variable:
•	Yes = Customer Churned 
•	No = Customer Stayed 
Tools and Libraries Used
The following Python libraries were used during this project:
•	Pandas 
•	NumPy 
•	Scikit-learn 
•	Joblib 
The Scikit-learn modules used include:
•	Pipeline 
•	ColumnTransformer 
•	StandardScaler 
•	OneHotEncoder 
•	SimpleImputer 
•	LogisticRegression 
•	RandomForestClassifier 
•	GridSearchCV 
•	Train-Test Split 
•	Classification Report 
•	Accuracy Score 
Project Workflow
The project follows an end-to-end machine learning workflow.
Step 1: Import Libraries
All required Python libraries were imported for data manipulation, preprocessing, model training, evaluation, and model saving.
Step 2: Load Dataset
The Telco Customer Churn dataset was loaded using Pandas.
pd.read_csv()
The dataset was successfully imported into a DataFrame.
Step 3: Data Exploration
The dataset was explored using:
•	head() 
•	info() 
•	isnull().sum() 
These functions helped understand:
•	Number of rows and columns 
•	Data types 
•	Missing values 
Step 4: Data Cleaning
Several preprocessing steps were performed before training.
The customerID column was removed because it does not contribute to prediction.
The TotalCharges column was converted from object type to numeric values.
Missing values were handled later using SimpleImputer inside the pipeline.
Step 5: Feature Selection
The target variable Churn was separated from the independent variables.
The target values were encoded as:
•	No → 0 
•	Yes → 1 
This converts the target into numerical format suitable for machine learning models.
Step 6: Train-Test Split
The dataset was divided into:
•	80% Training Data 
•	20% Testing Data 
This ensures the model is evaluated on unseen data.
Step 7: Identify Numerical and Categorical Features
The dataset contains both numerical and categorical columns.
Numerical columns included:
•	SeniorCitizen 
•	tenure 
•	MonthlyCharges 
•	TotalCharges 
Categorical columns included:
•	gender 
•	Partner 
•	Dependents 
•	InternetService 
•	Contract 
•	PaymentMethod 
•	and other customer-related attributes. 
These were automatically identified using Pandas.
Step 8: Numerical Pipeline
A numerical preprocessing pipeline was created.
The pipeline performs:
•	Missing value imputation using the median 
•	Feature scaling using StandardScaler 
This ensures numerical features are standardized before model training.
Step 9: Categorical Pipeline
A separate pipeline was created for categorical features.
The pipeline performs:
•	Missing value imputation using the most frequent value 
•	One-Hot Encoding of categorical variables 
One-Hot Encoding converts text categories into numerical format.
Step 10: Column Transformer
A ColumnTransformer combined both preprocessing pipelines.
It automatically applies:
•	Numerical Pipeline to numerical columns 
•	Categorical Pipeline to categorical columns 
This creates a single preprocessing stage for the complete dataset.
Step 11: Logistic Regression Pipeline
A complete machine learning pipeline was created using:
•	Preprocessing 
•	Logistic Regression 
The pipeline automatically preprocesses the data before training the model.
Step 12: Model Training
The Logistic Regression model was trained using the training dataset.
The pipeline automatically:
•	Handles missing values 
•	Scales numerical data 
•	Encodes categorical data 
•	Trains the classifier 
Step 13: Model Evaluation
The trained model was evaluated using:
•	Accuracy Score 
•	Classification Report 
Performance metrics such as precision, recall, F1-score, and overall accuracy were calculated.
Step 14: Random Forest Pipeline
A second machine learning pipeline was created using Random Forest Classifier.
The preprocessing steps remained the same while only the classifier was changed.
This allows easy comparison between different machine learning models.
Step 15: Hyperparameter Tuning
GridSearchCV was used to optimize Random Forest parameters.
The following parameters were tuned:
•	Number of Trees 
•	Maximum Tree Depth 
•	Minimum Samples Split 
Cross-validation was used to select the best parameter combination.
Step 16: Final Model Evaluation
The best model returned by GridSearchCV was evaluated on the testing dataset.
The final model achieved improved performance after hyperparameter tuning.
Performance metrics included:
•	Accuracy 
•	Precision 
•	Recall 
•	F1-score 
Step 17: Model Export
The trained pipeline was saved using Joblib.
joblib.dump(best_model, "customer_churn_pipeline.pkl")
Saving the model allows it to be reused without retraining.
Step 18: Load Saved Pipeline
The saved model was loaded using Joblib.
joblib.load()
Predictions were successfully generated using the loaded pipeline.
Technologies Used
•	Python 
•	Google Colab 
•	Pandas 
•	NumPy 
•	Scikit-learn 
•	Joblib 
Machine Learning Algorithms
The following algorithms were implemented:
•	Logistic Regression 
•	Random Forest Classifier 
Hyperparameter optimization was performed using GridSearchCV.
Skills Gained
This project helped develop the following skills:
•	Data preprocessing 
•	Missing value handling 
•	Feature encoding 
•	Feature scaling 
•	Machine Learning Pipelines 
•	ColumnTransformer 
•	Logistic Regression 
•	Random Forest Classification 
•	Hyperparameter Tuning 
•	Model Evaluation 
•	Model Export 
•	Production-ready Pipeline Design 
Challenges Faced
During the implementation of this project, several issues were encountered:
•	Dataset loading issues 
•	Missing values in the TotalCharges column 
•	Incorrect parameter names in GridSearchCV 
•	Inconsistent sample size errors during model training 
•	Pipeline configuration errors 
These issues were resolved by correcting the preprocessing pipeline, ensuring consistent train-test splits, and using valid parameter names with double underscores (classifier__parameter) in GridSearchCV.
Conclusion
This project successfully demonstrates the development of an end-to-end machine learning pipeline using Scikit-learn's Pipeline API. The pipeline automates preprocessing, training, evaluation, hyperparameter tuning, and model saving in a single workflow. The final pipeline is reusable, production-ready, and can be deployed for predicting customer churn on new customer data.
AI/ML Internship Task 3
Title
Housing Price Prediction using Image and Tabular Data
Objective
The main objective of this project is to predict house prices by using both house images and tabular property information. Instead of relying only on numerical features, this project combines visual features from house images with property details to improve prediction accuracy.
Dataset
The project uses the SoCal House Price Dataset, which contains house images along with a CSV file. The CSV includes information such as the number of bedrooms, bathrooms, square footage, city name, and house price. Each house image is linked with its corresponding record through an image ID.
Project Description
In this project, the dataset was loaded and preprocessed before training the model. Missing values were removed, categorical city names were converted into numerical values, and numerical features were normalized. House images were resized and processed so they could be used by the deep learning model.
A pre-trained ResNet50 Convolutional Neural Network was used to extract important visual features from each house image. These image features were then combined with the tabular features such as bedrooms, bathrooms, square footage, and city information. The combined data was used to train a Deep Neural Network regression model that predicts the price of a house.
After training, the model generated predictions on the test data. The performance of the model was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score. Different graphs were also generated to compare actual and predicted house prices and to analyze the model's performance.
Technologies Used
•	Python 
•	Google Colab 
•	TensorFlow 
•	Keras 
•	OpenCV 
•	Pandas 
•	NumPy 
•	Matplotlib 
•	Scikit-learn 
Model Used
•	ResNet50 for extracting image features. 
•	Deep Neural Network (DNN) for house price prediction. 
Evaluation Metrics
The model performance was evaluated using:
•	Mean Absolute Error (MAE) 
•	Root Mean Squared Error (RMSE) 
•	R² Score 
These metrics were used to measure the prediction accuracy and overall performance of the model.
Results
The model successfully extracted image features using ResNet50 and combined them with tabular property information. The trained regression model was able to predict house prices, and the evaluation metrics showed the overall performance of the model. Visualizations such as training graphs and prediction graphs were also generated to better understand the model's accuracy.
Conclusion
This project demonstrates the use of a multimodal machine learning approach for house price prediction by combining image data with numerical property information. Using ResNet50 for feature extraction and a Deep Neural Network for regression improves the model's ability to learn from both visual and tabular data. The project highlights how combining different types of data can produce more informative predictions for real estate applications.
AI/ML Internship Task 5 Documentation
Title
Auto Tagging Support Tickets Using Prompt Engineering with Gemini LLM
Objective
The objective of this project was to develop an intelligent support ticket classification system using a Large Language Model (LLM). The system automatically analyzes customer support tickets and predicts the three most relevant tags, reducing manual effort and improving the efficiency of customer support operations.
Dataset
The project used the Customer Support Ticket Dataset downloaded from Kaggle. The dataset contains customer support ticket descriptions along with their corresponding ticket categories. The ticket description was used as the input text, while the ticket type was used as the reference label for evaluation.
Methodology
The dataset was loaded into Google Colab using Pandas and the required columns were selected for processing. The ticket description column was renamed as text, and the ticket type column was renamed as label to simplify further processing.
Instead of fine-tuning a local transformer model, the project used Google Gemini 1.5 Flash through the Gemini API. This approach avoided downloading large pretrained models and provided a faster and more reliable solution.
Prompt engineering techniques were applied to classify each support ticket. Two prompting strategies were used:
•	Zero-Shot Prompting, where the model classified tickets without any examples. 
•	Few-Shot Prompting, where example ticket-label pairs were included in the prompt to improve classification quality. 
For every support ticket, the model predicted the top three most suitable tags from the predefined list of ticket categories.
Results
The Gemini model successfully analyzed customer support tickets and generated the three most relevant categories for each ticket. The predicted tags were consistent with the ticket content and demonstrated the effectiveness of prompt engineering for automated ticket classification.
Due to runtime limitations and model download issues in Google Colab, local model fine-tuning was not completed. Instead, prompt engineering with Gemini API was used to accomplish the classification task successfully.
Tools and Libraries Used
•	Google Colab 
•	Python 
•	Pandas 
•	Google Generative AI (Gemini API) 
•	Google Gemini 1.5 Flash 
Conclusion
This project demonstrated how Large Language Models can automate customer support ticket classification using prompt engineering. By applying Zero-Shot and Few-Shot prompting, the system accurately predicted relevant ticket categories without requiring local model training. The developed solution provides a practical and efficient approach for automatically tagging customer support tickets, helping organizations improve response time and customer service efficiency.
AI/ML Internship – Task 4 Report
Task Title
Context-Aware Chatbot Using LangChain (RAG)
Objective
The objective of this project is to build a context-aware conversational chatbot that can understand user queries, remember conversation history, and retrieve relevant information from external documents using Retrieval-Augmented Generation (RAG). The chatbot combines document retrieval with a Large Language Model to provide accurate and context-based responses.
Dataset
A custom PDF document was used as the knowledge base for the chatbot. The document was loaded into the system and divided into smaller text chunks for efficient searching. These chunks were converted into vector embeddings and stored in a FAISS vector database.
Project Overview
The chatbot was developed using the LangChain framework with a Retrieval-Augmented Generation (RAG) architecture. The PDF document was first loaded and split into smaller chunks. HuggingFace sentence embeddings were generated for each chunk and stored in a FAISS vector database. Whenever a user asks a question, the chatbot searches the vector database to retrieve the most relevant document sections and uses them as context to generate an accurate response. Conversation memory is also maintained so that the chatbot can remember previous interactions and answer follow-up questions naturally.
Technologies Used
•	Python 
•	LangChain 
•	HuggingFace Embeddings 
•	FAISS Vector Database 
•	Sentence Transformers 
•	PyPDF 
•	Streamlit 
Methodology
The project begins by loading a PDF document containing the knowledge base. The document is divided into smaller text chunks using a text splitter. These chunks are converted into vector embeddings using the HuggingFace all-MiniLM-L6-v2 embedding model. The embeddings are stored in a FAISS vector database for fast similarity search. During a conversation, the chatbot retrieves the most relevant document chunks based on the user's query and provides responses using the retrieved information. Conversation memory is maintained to preserve the context throughout the chat session. Finally, the chatbot is deployed using Streamlit to provide an interactive web interface.
Features
•	Context-aware conversation 
•	Retrieval-Augmented Generation (RAG) 
•	Document-based question answering 
•	PDF knowledge base support 
•	Vector similarity search using FAISS 
•	Conversation memory 
•	Interactive Streamlit interface 
Results
The chatbot successfully retrieves relevant information from the uploaded PDF document and provides context-aware responses. It remembers previous conversations, allowing users to ask follow-up questions naturally. The use of FAISS enables fast document retrieval, while LangChain integrates retrieval and language model capabilities effectively.
Skills Gained
•	Retrieval-Augmented Generation (RAG) 
•	LangChain Framework 
•	Conversational AI Development 
•	Vector Embeddings 
•	FAISS Vector Database 
•	Document Retrieval 
•	HuggingFace Embeddings 
•	Streamlit Deployment 
Conclusion
This project demonstrates how Retrieval-Augmented Generation (RAG) can improve chatbot performance by combining document retrieval with conversational AI. By integrating LangChain, HuggingFace embeddings, FAISS, and Streamlit, the chatbot can answer questions based on external knowledge while maintaining conversation context. The project provides practical experience in building intelligent AI applications that retrieve, process, and present information efficiently.

