import torch
import os
from torch import nn
import random
import sklearn
from sklearn.metrics import r2_score
import time
from torchsummary import summary
import pandas as pd

"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""


'''  
import google.generativeai as genai

genai.configure(api_key="AIzaSyANKhxvkHOFO7KnQ5H7PuOW9Y74UcFrXoc")

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

chat_session = model.start_chat(
  history=[
  ]
)
i=0
while True:
    i=i+1
    response = chat_session.send_message("INSERT_INPUT_HERE" + str(i))
    print(response.text)
x=1
 '''
 
 
X=None
y=None
 


# write a function to read a file
def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

class MyMachine(nn.Module): #inherit from nn.Module
    def __init__(self,width): #constructor
        super().__init__() #initialize the parent class
        self.fc = nn.Sequential( 
            # nn.Linear is a fully connected layer, it takes the input and multiplies it by a weight matrix and adds a bias term
            # the first parameter is the number of input features, the second parameter is the number of output features
            nn.Linear(width,5), 
            nn.ReLU(), 
            nn.Linear(5,1) 
        )

    def forward(self, x): #forward pass
        x = self.fc(x) #pass the input through the fully connected layer
        return x

def get_dataset2():
    X = pd.read_csv("C:/projects/train-nabila/train.csv",header=0, nrows=1000)
    for column in X.columns[0:]:
        X[column]=X[column].astype(dtype='float32')
    #df['StringColumn'] = df['StringColumn'].astype(float)
    y = torch.tensor(X['label'].values)
    X.pop('label')
    X = torch.tensor(X.values)
    return X, y

    
submit_test_df = pd.read_csv("C:/projects/train-nabila/test.csv",header=0, nrows=1000)
def get_dataset():
        X = torch.rand((1000,2))
        noise = random.uniform(0, 0.1)
        x1 = X[:,0] #first column 
        x2 = X[:,1] #second column
        y = x1 * x2 * noise #output is the product of the two columns with some noise
        return X, y

def train(X,y):
    # ge the tensor width which is the number of columns in the input
    # each column is a feature
    width = X.shape[1]
    # get the number of rows in the input
    # eqch row is a sample
    r = X.shape[0]
    model = MyMachine(width)
    model.train() #tell pytorch that we are training the model
    
    NUM_EPOCHS = 100
    # Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data
    # weight_decay value is a regularization term that penalizes large weights, and regularization is a technique used to prevent overfitting by adding a penalty term to the loss function
    # for example L1 regularization adds a penalty term that is proportional to the absolute value of the weights, while L2 regularization adds a penalty term that is proportional to the square of the weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5) #weight_decay is L2 regularization, L2 means that the error is proportional to the square of the weight
    # criterion differs from optimizer in that criterion is the loss function that we want to minimize, while optimizer is the algorithm that we use to minimize the loss function
    criterion = torch.nn.MSELoss(reduction='mean') #mean squared error loss

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad() #zero the gradients
        y_pred = model(X) #forward pass
        y_pred = y_pred.reshape(1000) #flatten the output
        loss = criterion(y_pred, y) #calculate the loss
        loss.backward() #backpropagate the loss
        optimizer.step() #update the weights
        print(f'Epoch:{epoch}, Loss:{loss.item()}')
        #sleep for a bit
        #time.sleep(0.1)
    torch.save(model.state_dict(), 'model1.h5')

def test(X,y):
    width = X.shape[1]
    model = MyMachine(width)
    model.load_state_dict(torch.load("model1.h5"))
    model.eval()
    X, y = get_dataset2()

    with torch.no_grad():
        y_pred = model(X)
        #print(r2_score(y.cpu(), y_pred.cpu())) #R^2 score is the proportion of the variance in the dependent variable that is predictable from the independent variable
        print(sklearn.metrics.mean_squared_error(y, y_pred)) #mean squared error is the average of the squares of the errors
  
X,y=get_dataset2()
train(X,y)
test(X,y)
#X1,y1=get_dataset()
#train(X1,y1)


x=4