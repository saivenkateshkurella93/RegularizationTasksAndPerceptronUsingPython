'''
Name: Sai Venkatesh Kurella
Campus ID: VR62250
CMSC 678 -Introduction to Machine Learning
Homework-2

'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plot
from random import random,shuffle,sample
from operator import itemgetter


def accuracy(testdata,actual_label,weights,bias): #accuracy function
    correct = 0
    for i in range(len(testdata)):
        pred = prediction(testdata[i],weights,bias)
        if pred == actual_label[i][0]: correct += 1
    return correct/float(len(testdata))*100

def prediction(inputs,weights,bias): #prediction fuction that handles inputs and weights
    activation = 0
    for input,weight in zip(inputs,weights):
        activation += input*weight + bias
    if activation > 0:
        return 1
    else:
        return -1

def perceptron(train_data,train_label,test_data,test_label,iteration): #perceptron function
    session = tf.compat.v1.Session()
    data_train = tf.compat.v1.placeholder(dtype=tf.float32,shape = [28*28])
    label_train = tf.compat.v1.placeholder(dtype= tf.float32,shape = [1])
    weight = tf.compat.v1.placeholder(dtype=tf.float32,shape=[28*28])

    weight = tf.multiply(data_train,label_train)

    weights = np.zeros(28*28)
    bias = 0
    for i in range(0,iteration):
        for j in range(0,len(train_data)):
            predict = prediction(train_data[j],weights,bias)
            if predict != train_label[j][0]:
                weights += session.run(weight,feed_dict={data_train:train_data[j],label_train:train_label[j]})
                bias += train_label[j][0]

    return accuracy(test_data,test_label,weights,bias),weights

def extract_data(train_images,train_labels,test_images,test_labels,num1,num2):
    image_train,label_train,image_test,label_test = [],[],[],[]
    #1 denoted by num1 and -1 denoted by num2
    #training data
    num_count = 0
    for i in range(len(train_labels)):
        #extract only num1's
        if num_count < 500 and train_labels[i][num1] == 1:
            image_train.append(train_images[i])
            label_train.append([1])
            num_count += 1
    num_count = 0
    for i in range(len(train_labels)):
        #extract only num2's
        if num_count < 500 and train_labels[i][num2] == 1:
            image_train.append(train_images[i])
            label_train.append([-1])
            num_count += 1
    num_count = 0
    #Testing data
    for i in range(len(test_labels)):
        if num_count < 500 and test_labels[i][num1] == 1:
            image_test.append(test_images[i])
            label_test.append([1])
            num_count += 1
    num_count = 0
    for i in range(len(test_labels)):
        #extract only num2's
        if num_count < 500 and test_labels[i][num2] == 1:
            image_test.append(test_images[i])
            label_test.append([-1])
            num_count += 1
    return image_train,label_train,image_test,label_test

def extract_data_shuffled(train_images,train_labels,test_images,test_labels,count,num1,num2):
    image_train,label_train,image_test,label_test = [],[],[],[]
    #1 denoted by num1 and -1 denoted by num2
    #training data
    num1_count,num2_count = 0,0
    for i in range(len(train_labels)):
        #extract only num1's
        if num1_count < count and train_labels[i][num1] == 1:
            image_train.append(train_images[i])
            label_train.append([1])
            num1_count += 1
        #extract only num2's
        if num2_count < count and train_labels[i][num2] == 1:
            image_train.append(train_images[i])
            label_train.append([-1])
            num2_count += 1

    #shuffle training data
    train = list(zip(image_train,label_train))
    shuffle(train)
    image_train, label_train = zip(*train)

    #Testing data
    num1_count,num2_count = 0,0
    for i in range(len(test_labels)):
        if num1_count < count and test_labels[i][num1] == 1:
            image_test.append(test_images[i])
            label_test.append([1])
            num1_count += 1
        #extract only num2's
        if num2_count < count and test_labels[i][num2] == 1:
            image_test.append(test_images[i])
            label_test.append([-1])
            num2_count += 1
    #shuffle testing data
    test = list(zip(image_test,label_test))
    shuffle(test)
    image_test, label_test = zip(*test)

    return image_train,label_train,image_test,label_test

def accuracy_iteration(train_data,train_label,test_data,test_label,iteration,num1,num2):

    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    plot_name = "accuracy_iteration%s_%s%s.png" %(iteration,num1,num2)
    plot.savefig(plot_name)

def get_score(train_images,train_labels,test_images,test_labels,iteration,num1,num2):
    weights =  perceptron(train_images,train_labels,test_images,test_labels,iteration)[1]
    weight_pos, weight_neg = [], []

    for i in range(len(weights)):
        if weights[i] >= 0:
            weight_pos.append(weights[i])
        else:
            weight_pos.append(0)

    for i in range(len(weights)):
        if weights[i] <= 0:
            weight_neg.append(weights[i])
        else:
            weight_neg.append(0)

    pos_test, neg_test = [], []

    for i in range(len(test_labels)):
        if test_labels[i][0] == num1:
            pos_test.append(test_images[i])
        else:
            neg_test.append(test_images[i])
    #calculate score for num1 images
    score_pos = []
    for i in range(len(pos_test)):
        score = 0
        for j in range(len(pos_test[i])):
            score += abs(weight_pos[j] - pos_test[i][j])
        score_pos.append(score)
    #calculate score for num2 images
    score_neg = []
    for i in range(len(neg_test)):
        score = 0
        for j in range(len(neg_test[i])):
            score += abs(weight_neg[j] - neg_test[i][j])
        score_neg.append(score)

    pos_test_score = list(zip(pos_test,score_pos))
    pos_test_score = sorted(pos_test_score,key=itemgetter(1),reverse = True)
    #20 best num1
    best_pos = []
    best_pos_arr = pos_test_score[0:20]
    for i in range(len(best_pos_arr)):
        best_pos.append(best_pos_arr[i][0])

    worst_pos_arr = pos_test_score[-21:-1]
    worst_pos = []
    for i in range(len(worst_pos_arr)):
        worst_pos.append(worst_pos_arr[i][0])

    neg_test_score = list(zip(neg_test,score_neg))
    neg_test_score = sorted(neg_test_score,key = itemgetter(1),reverse = True)
    #20 best num2
    best_neg = []
    best_neg_arr = neg_test_score[0:20]
    for i in range(len(best_neg_arr)):
        best_neg.append(best_neg_arr[i][0])
    #20 worst num2
    worst_neg = []
    worst_neg_arr = neg_test_score[-21:-1]
    for i in range(len(worst_neg_arr)):
        worst_neg.append(worst_neg_arr[i][0])

    #image plot for num1 best 20
    for i in range(len(best_pos)):
        temp = []
        for j in range(0,len(best_pos[i]),28):
            temp.append(best_pos[i][j:j+28])
        plot.subplot(4, 5, i + 1)
        plot.imshow(temp,'gray_r')
    plt_name = "best_20_%s.png"%(num1)
    plot.savefig(plt_name)

    #image plot for num1 worst 20
    for i in range(len(worst_pos)):
        temp = []
        for j in range(0,len(worst_pos[i]),28):
            temp.append(worst_pos[i][j:j+28])
        plot.subplot(4,5,i+1)
        plot.imshow(temp,'gray_r')
    plt_name = "worst_20_%s.png"%(num1)
    plot.savefig(plt_name)

    #image plot for num2 best 20
    for i in range(len(best_neg)):
        temp = []
        for j in range(0,len(best_neg[i]),28):
            temp.append(best_neg[i][j:j+28])
        plot.subplot(4, 5, i + 1)
        plot.imshow(temp,'gray_r')
    plt_name = "best_20_%s.png"%(num2)
    plot.savefig(plt_name)

    #image plot for num2 worst 20
    for i in range(len(worst_neg)):
        temp = []
        for j in range(0,len(worst_neg[i]),28):
            temp.append(worst_neg[i][j:j+28])
        plot.subplot(4,5,i+1)
        plot.imshow(temp,'gray_r')
    plt_name = "worst_20_%s.png"%(num2)
    plot.savefig(plt_name)

def visualize_weight_vector(train_data,train_label,test_data,test_label,iteration,num1,num2):
    weights = perceptron(train_data,train_label,test_data,test_label,iteration)[1]

    weight_pos, weight_neg = [], []

    for i in range(len(weights)):
        if weights[i] >= 0:
            weight_pos.append(weights[i])
        else:
            weight_pos.append(0)

    for i in range(len(weights)):
        if weights[i] <= 0:
            weight_neg.append(abs(weights[i]))
        else:
            weight_neg.append(0)
    pos_weight,neg_weight = [],[]

    for i in range(0,len(weight_pos),28):
        pos_weight.append(weight_pos[i:i+28])
    for i in range(0,len(weight_neg),28):
        neg_weight.append(weight_neg[i:i+28])

    plot.imshow(pos_weight,'gray_r')
    plt_name = "weight_%s.png"%(num1)
    plot.savefig(plt_name)
    plot.imshow(neg_weight,'gray_r')
    plt_name = "weight_%s.png"%(num2)
    plot.savefig(plt_name)
    
    
def sorted_data_visualization(train_data,train_label,test_data,test_label,iteration,num1,num2):
    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    plot_name = "sorted_accuracy_iteration%s%s.png" %(num1,num2)
    plot.savefig(plot_name)

def random_flip(train_data,train_label,test_data,test_label,iteration): #functon for random flip
    index = sample(range(1000), 100)

    for i in index:
        if train_label[i] == [1]:
            train_label[i][0] = -1
        else:
            train_label[i][0] = 1
    x = [i*len(train_data) for i in range(iteration)]
    y = []
    for i in range(0,iteration):
        y.append(perceptron(train_data,train_label,test_data,test_label,i)[0])
    plot.ylim(0,100)
    plot.plot(x,y)
    plot.savefig('accuracy_random_flip.png')
    return perceptron(train_data,train_label,test_data,test_label,iteration)[0]



def main():
    print("Importing MNIST Dataset")
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,1,6)

    print("(a). Accuracy for classifying digits 1 and 6\n(b). Accuracy plot with number of iterations for classifying digits 1 and 6\n(c). Visualization of learned model for digits 1 and 6\n(d). Visualization of 20 best and worst scoring images\n(e). Random flip for 10% of training data\n(f). Visualization of sorted training data\n(g). Accuracy plot for digits 2 and 8\n(h). Weight vector Visualization for digits 2 and 8\n(i). Accuracy plot with 10 training examples\n")

    choice = input("***Please enter your choice from a to i***")
    if choice == 'a':
        accuracy = perceptron(train_images,train_labels,test_images,test_labels,6)[0]
        print("Accuracy for classifying digits 1 and 6 is:",accuracy)
    elif choice == 'b':
        accuracy_iteration(train_images,train_labels,test_images,test_labels,10,1,6)
        print("Accuracy-iteration plot for digits 1 and 6 ploted!")
    elif choice == 'c':
        visualize_weight_vector(train_images,train_labels,test_images,test_labels,10,1,6)
        print("Learned model for digits 1 and 6 plotted!")
    elif choice == 'd':
        print("Image plot for best and worst 20 images")
        get_score(train_images,train_labels,test_images,test_labels,10,1,6)
    elif choice == 'e':
        accuracy_random = random_flip(train_images,train_labels,test_images,test_labels,10)
        print("Accuracy for classifying digits 1 and 6 with 10%\ random flip",accuracy_random)
        print("Accuracy plot with 10% \error plotted!")
    elif choice == 'f':
        train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted = extract_data(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,1,6)
        sorted_data_visualization(train_images_sorted,train_labels_sorted,test_images_sorted,test_labels_sorted,10,1,6)
        print("Accuracy plot with sorted data plotted!")
    elif choice == 'g':
        train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,2,8)
        accuracy_iteration(train_images,train_labels,test_images,test_labels,10,2,8)
        print("Accuracy-iteration plot for digits 2 and 8 plotted!")
    elif choice == 'h':
        train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,500,2,8)
        accuracy = perceptron(train_images,train_labels,test_images,test_labels,10)[0]
        print("Accuracy for digits 2 and 8",accuracy)
        visualize_weight_vector(train_images,train_labels,test_images,test_labels,10,2,8)
        print("Learned model for digits 2 and 8 plotted!")
    elif choice == 'i':
        train_images,train_labels,test_images,test_labels = extract_data_shuffled(mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels,5,1,6)
        accuracy = perceptron(train_images,train_labels,test_images,test_labels,10)[0]
        print("Accuracy with 10 training examples",accuracy)
        accuracy_iteration(train_images,train_labels,test_images,test_labels,10,1,6)
        print("Plot for 10 training examples plotted!!")





if __name__ =="__main__":
    main()
