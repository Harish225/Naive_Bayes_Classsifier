__author__ = 'NKHarish'
## Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math

#####################Calculating Priors#########################################

## Function to calculate the Prior values P(Y)= (no.of documents belonging to a label/total no. of documents)
def calc_prior():

    global label_doc_count
    global label_prior_value
    m=np.loadtxt("labels.txt",delimiter="\n")   ## Loading the train.label file
    m=np.array(m)
    m=m.astype(int)
    y = np.bincount(m)
    ii = np.nonzero(y)[0]
    label_doc_count=np.matrix((ii,y[ii])).T     ## Matrix that stores the count of documents per each label


    label_prior_value=label_doc_count
    label_prior_value=label_prior_value.astype(float)
    label_prior_value[:,1]=label_prior_value[:,1]/11269  ## Matrix that stores the Prior values for every label


##############################Calculation of Document Index Matrix#################################

## Function to calculate the Document ID range for every label
def doc_indexes(label_doc_count):

    global doc_index
    doc_index=np.zeros([20,2])
    doc_index=doc_index.astype(int)
    start_index=label_doc_count[(0,0)]
    end_index=label_doc_count[(0,1)]
    i=0
    while i<20:
        doc_index[(i,0)]=start_index
        doc_index[(i,1)]=end_index
        if (i+1)!=20:
            start_index=end_index+1
            end_index=label_doc_count[(i+1,1)]+end_index
        i=i+1


################################Calculation of Likelihood###########################################

def calc_likelihood():

    global likelihood_matrix
    global input_data
    likelihood_matrix=np.zeros((vocabulary_count, no_labels))       ## Initiating the Likelihood matrix of size(vocabulary_count, no_labels)
    input_data=np.loadtxt("train_data.txt",delimiter=" ")           ## Loading the train_data file
    input_data=np.matrix(input_data).astype(int)
    row_index=0
    beta=1/vocabulary_count                                         ## Initiating the variables alpha and beta used for likelihood calculation
    alpha=1+beta
    print("Executing Likelihood Matrix Calculation")
    i=0
    while i < no_labels:
        print("Calculating Likelihood values for label ",i+1)
        condition=(np.logical_and(input_data[:,0]>= doc_index[(i,0)],input_data[:,0]<=doc_index[(i,1)]))
        current_index= np.extract(condition, input_data).size
        current_label_matrix=input_data[row_index:row_index+current_index]      ## Extracting the data into matrix that belongs to 'i' label
        current_label_wordcount=np.sum(current_label_matrix,axis=0)[(0,2)]      ## Finding the total number of words present in the 'i' label
        current_label_unique_words=np.unique(np.array(current_label_matrix[:,1].T))         ## Finding the distinct Word ID's present in the label 'i'
        unique_word_size=np.array(current_label_unique_words).size
        likelihood_matrix [:,i] = (alpha-1)/(current_label_wordcount+(alpha-1)*vocabulary_count)        ## Initiating default likelihood value to all the label 'i' Word ID's
        j=0
        while j < unique_word_size:
                current_label_word_matrix=np.array(current_label_matrix)
                 ## Calculating the total count for 'j' Word ID in the 'i' label data
                current_label_word_count=np.sum(np.matrix(current_label_word_matrix[current_label_word_matrix[:,1] == current_label_unique_words[j]]),axis=0)[(0,2)]
                ## Calculating the likelihood values for 'j' Word ID in 'i' label and assigning it to Likelihood Matrix
                likelihood_matrix[(current_label_unique_words[j]-1 , i)] = (current_label_word_count+(alpha-1))/(current_label_wordcount+(alpha-1)*vocabulary_count)
                j=j+1
        row_index=row_index+current_index
        i=i+1


########################################Calculation of Posterior Probability################################

## Function to calculate the Posterior value P(Y/X)=(Count of Word ID in Label+(alpha-1))/(Total Word Count in Label+(beta*vocabulary_size)) for every Word ID
def calc_posterior():

    global posterior_matrix
    global test_output_matrix
    global test_data
    global no_testdocs
    no_testdocs=7505
    test_output_matrix=np.zeros((no_testdocs,1))
    posterior_matrix=np.zeros((no_testdocs,no_labels))          ## Initiating the Posterior Matrix with zeros
    test_data=np.loadtxt("test_data.txt",delimiter=" ")         ## Loading the test_data file.
    test_data=np.matrix(test_data).astype(int)
    print("################# Test Data Matrix ##########################")
    print(test_data)
    row_index=0
    i=0
    print("Executing Posterior Matrix Calculations")
    while i<no_testdocs:                                    ## Looping for the total number of documents present in the test_data file.
        condition=(test_data[:,0]==i+1)
        current_index=(np.extract(condition, test_data)).size
        current_doc_matrix=test_data[row_index:row_index+current_index]     ## Extracting the 'i' Document ID data from the entire set.
        current_doc_wordcount=np.sum(current_doc_matrix,axis=0)[(0,2)]
        j=0
        while j < no_labels:                                            ## Looping to calculate the Posterior values for the all the labels.
                current_label_prior=label_prior_value[(j,1)]
                k=0
                sum_posterior=0
                while k < current_index:
                      ## Calculating the Posterior value for the document 'i' and for the label 'j'
                     sum_posterior=sum_posterior+current_doc_matrix[(k,2)]*math.log2(likelihood_matrix[((current_doc_matrix[(k,1)]-1),j)])
                     k=k+1
                sum_posterior=sum_posterior+math.log2(current_label_prior)
                posterior_matrix[(i,j)]=sum_posterior   ## Assigning the calculated Posterior value to Posterior matrix
                j=j+1
        row_index=row_index+current_index
        i=i+1
    test_output_matrix=(np.matrix(posterior_matrix.argmax(axis=1)+1)).astype(int)   ## Predicting labels for the total document set.



############################################Calculation of Accuracy & Confusion Matrix################################

## Function to calculate the Accuracy
def calc_accuracy():

    global confusionMatrix
    label_accuracy=np.zeros([no_labels,1])
    test_label_matrix=np.loadtxt("test_labels.txt",delimiter="\n")      ## Loading the given test_label file for calculating the Accuracy
    test_label_matrix=np.matrix(test_label_matrix)
    print("#################### Test Label Matrix #####################")
    print(test_label_matrix)
    i=0
    accuracy_counter=0
    while i < no_testdocs:
        if test_label_matrix[(0,i)]==test_output_matrix[(0,i)]:
            accuracy_counter=accuracy_counter+1
        i=i+1
    accuracy=(accuracy_counter/no_testdocs)*100             ## Calculating the Accuracy value
    confusionMatrix=confusion_matrix(np.array(test_label_matrix.T),np.array(test_output_matrix.T))          ## Calculating the Confusion Matrix
    plt.matshow(confusionMatrix)
    plt.colorbar()
    plt.savefig("confusion_matrix.pdf")
    j=0
    while j < 20:
        label_accuracy[j,0]=(confusionMatrix[(j,j)]/np.sum(confusionMatrix[j,:]))*100           ## Calculating the Accuracy per label
        j=j+1
    print("####################### Accuracy per Label ###########################")
    print(label_accuracy)
    return accuracy

###########################################Calculation of 100 words########################################

## Function to predict the 100 Words that are more significant.

def calc_100words():

    global top100words_list
    max_wordid_across_labels=np.matrix(np.amax(likelihood_matrix,axis=1)).T        ## Finding the maximum P(X/Y) value for Word ID across all the Labels
    max_difference=np.subtract(max_wordid_across_labels,likelihood_matrix)         ## Subtracting the Word ID's P(X/Y) value from the Maximum value across all the labels
    sum_max_difference=np.sum(max_difference,axis=1)                               ## Summing the P(X/Y) values across all the labels for a Word ID
    max_index_matrix=np.argsort(sum_max_difference.T,axis=1)                       ## Sorting the array to find the Word ID's with the largest summed up value.
    top100words_index=np.array(max_index_matrix[:,max_index_matrix.size-101:max_index_matrix.size]).astype(int)
    vocabulary_file=open("vocabulary.txt")
    vocabulary = [line.strip() for line in vocabulary_file]                        ## Loading the given Vocabulary file into a list.
    i=0
    top100words_list=[]
    while i<top100words_index.size:
        top100words_list.append(vocabulary[top100words_index[0,i]])                ## Finding and Appending the top 100 Words to list
        i=i+1
    vocabulary_file.close()



#############################################Main Function##################################################

## Main Function
if __name__=="__main__":

    global vocabulary_count                 ## Initiating the Global Variables
    global no_labels
    vocabulary_count=61188
    no_labels=20
    calc_prior()                            ## Calling the function that calculates Prior values
    print("###################### Prior Matrix #######################")
    print(label_prior_value)
    doc_indexes(label_doc_count)            ## Calling the function to calculate the document Id range for every label
    calc_likelihood()
    print("##################### Likelihood Matrix ###################")
    print(likelihood_matrix)                ## Calling the function to calculate the Likelihood Matrix
    calc_posterior()
    print("#################### Posterior Matrix #####################")
    print(posterior_matrix[0,:])            ## Calling the function to calculate the Posterior Matrix
    print("#################### Document Classification Matrix ########")
    print(test_output_matrix)
    accuracy=calc_accuracy()                ## Calling the function to calculate the Accuracy value
    print("#################### Accuracy ##############################")
    print(accuracy)
    print("#################### Confusion Matrix ######################")
    print(confusionMatrix)
    calc_100words()                         ## ## Calling the function to find the Top 100 Words.
    print("#################### Top 100 Words #########################")
    print(top100words_list)