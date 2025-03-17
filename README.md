Semantic retrieval of Anti-Microbial Resistance information from research articles using Natural Language Processing (project with PhD scholar under college faculty in collaboration with coursemate [@swat08mx](https://github.com/swat08mx))

# Contents
[Introduction](#introduction)<br>
[NLP in clinical studies](#nlp-in-clinical-studies)<br>
[Project Statement](#project-statement)<br>
[Methodology](#methodology)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[The Pipeline](#the-pipeline)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Data Preprocessing Steps](#detailed-steps-for-data-processing) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[Model Development](#model-development) <br>
&nbsp;&nbsp;&nbsp;&nbsp;[Training and Testing](#training-and-testing)<br>
[Conclusion and Future Perspectives](#conclusion-and-future-perspectives)<br>


# Introduction
- Antimicrobial resistance (AMR) has become a pressing concern in medical science. It is a growing public health concern that threatens the effectiveness of antibiotics and other antimicrobial medicines. 
- Traditional methods of retrieving information from research articles are often time-consuming and rely on manual curation. 
- To overcome these limitations, researchers are exploring ways to use natural language processing (NLP) and machine learning techniques to improve efficiency of collecting information.

# NLP in clinical studies
- A review of artificial intelligence applications for antimicrobial resistance found that AI technology can be employed against AMR by using the predictive AMR model and the rational use of antibiotics.
- Clinical NLP has been successfully used to identify diseases, detect pathological causes of obesity, and ascertain patient asthma status from clinical notes.

# Project Statement
The goal of this project is :
To develop a Deep Learning NLP model that can extract relevant AMR information from research articles in a "semantic" and relevant manner.

Overview of a deep learning model, NLP based 

![Overview of a deep learning model, NLP based](https://github.com/user-attachments/assets/5bc57449-eeb7-4910-aa3e-4e4af45bd118)

# Methodology
## The Pipeline
![image](https://github.com/user-attachments/assets/c3eaca70-d84f-44f3-85a5-06234a685e82)

## Detailed steps for Data Processing
Code snippet for extracting data from JSON files, checking for annotations, appending start offset and end offset, fixing bugs, swapping labels, and zipping them together. (ALL in one)

![image](https://github.com/user-attachments/assets/4c4b83f8-b193-422e-9d54-25d9cb713d48)

![image](https://github.com/user-attachments/assets/0af7fd06-b884-48db-ad8e-42d3efaac7fd)

![image](https://github.com/user-attachments/assets/381d4e9c-c7ba-48ec-a5a7-52cd55f93bea)

Example of the output from the previous code 

![image](https://github.com/user-attachments/assets/67792ac1-c2cc-4ff9-8586-0a3cd5bc63e7)

Code snippet for taking in the list of lines and tags as inputs and their start and end offset for BILUO tagging of these sentences

![image](https://github.com/user-attachments/assets/dad1013f-c49b-492c-baf1-fae362880ba2)

Example of BILUO tagging by using spaCy library

![image](https://github.com/user-attachments/assets/0e741bf7-0f15-43ab-aea1-35793923acc7)

![image](https://github.com/user-attachments/assets/feb7ac73-9f7a-4965-9b1a-4a8942a30daa)

Finally, the half pre-processed data was inserted into a csv file and downloaded

![image](https://github.com/user-attachments/assets/866e4a14-d6ad-4c4e-9759-f1cbd03f4ecf)

The CSV file head() and tail() ‘ed view

![image](https://github.com/user-attachments/assets/91b9fc79-4f24-4e89-90df-3187cfec2140)

## Model Development
- Importing Transformers library and AutoTokenizer, AutoModel with it.
- Obtained from Huggingface repository.
- Model used: dmis-lab/biobert-v1
  
![image](https://github.com/user-attachments/assets/7e6653ed-893f-4388-8467-76eaa322655e)
![image](https://github.com/user-attachments/assets/782f7be9-4f35-4bde-b116-b49f03826cba)

Tokenizing

![image](https://github.com/user-attachments/assets/1129008d-c107-45b1-88f6-6c0503ea874a)

Code for aligning the labels list with the newly split tokens, marking the subwords as ‘X’ 

![image](https://github.com/user-attachments/assets/254a6ff3-9108-4342-91af-8fb4fc06a436)


Creating the inputs- 
- Input tokens
- Input tags
- Attention masks

Converting the tokens into input ids using a built-in AutoTokenizer function called convert_tokens_to_ids

![image](https://github.com/user-attachments/assets/23c59650-2cb5-47bf-b14e-e09ffa10945e)

Code snippet for padding the tokenids

![image](https://github.com/user-attachments/assets/3df7cf9c-567c-43af-a1ee-cc4557cc636c)

Sample output for padded tokenids

![image](https://github.com/user-attachments/assets/0be3deca-fb92-4e65-904e-e62e9f5d9a5f)

Representation of masking

![image](https://github.com/user-attachments/assets/8d7a40fb-3472-4898-b06d-91cbdfd8762e)

Code snippet for creating attention masks

![image](https://github.com/user-attachments/assets/e855e57b-bce0-4cd7-bd1a-db6bd59c938c)

To encode the tags into vectors-
- A dictionary was created to use as a reference for tags encoding.
- Then the tags were padded to maximum length
  
![image](https://github.com/user-attachments/assets/e4168735-e8d2-478f-bba5-ff51b5e26c7c)

Sample inputs for feeding into BERT

![image](https://github.com/user-attachments/assets/2f9aa261-e834-4356-b5c3-30e4c63f1a2c)

Split dataset into train and testing datasets

![image](https://github.com/user-attachments/assets/b05e9ed2-b895-438b-bda6-82bbf891ff09)

Convert the inputs into Tensors

![image](https://github.com/user-attachments/assets/a44beef0-6c28-4526-b632-5701bac43308)

Zipping the tokens, tags and attentions masks into TensorDataset

![image](https://github.com/user-attachments/assets/48efd202-2715-449a-82fa-681edf5a6289)

Creating a BERT layer from the imported Pretrained model with dropout and a linear layer

![image](https://github.com/user-attachments/assets/1596cdd3-3890-4495-84f0-8ea645da2ec2)

## Training and Testing
Model Training

![image](https://github.com/user-attachments/assets/e0f4cb52-03de-4ba4-9f81-b6ed69132fdb)

Model testing

![image](https://github.com/user-attachments/assets/1502c1f6-ad5f-4373-83ba-a951f5ebb712)

Classification report generation

![image](https://github.com/user-attachments/assets/58796859-94b7-434e-acf0-dd84d41a7a31)

# Result

![image](https://github.com/user-attachments/assets/0363c050-22d7-42c8-88a3-3ff737fb53d2)

# Conclusion and Future Perspectives
- The overall precision, recall, and f1-score for the model are 0.07, 0.17, and 0.10, respectively
- Weighted average precision, recall, and f1-score for the model are 0.86, 0.17, and 0.19, respectively
- Since these metrics are relatively low, next step is to check the quality of model and dataset
- Can be done by using another dataset
- Future – can be applied to other types of datasets

# References and Image Courtesy
https://cdn.ourcrowd.com/wp-content/uploads/2023/10/bigstock-Big-Data-Machine-Learning-And-475995541.jpg
https://st4.depositphotos.com/9233766/25455/i/450/depositphotos_254551828-stock-photo-nlp-natural-language-processing-cognitive.jpg
https://www.researchgate.net/publication/341262600/figure/fig1/AS:897717807087618@1591044012388/The-BIO-and-BILOU-encoding-schemas.png
