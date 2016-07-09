--Author: Chun-Hao Liu
-- Date: 06/26/2016
-- Goal: End-to-end flow for Deep learning to predict online housing price
-- Comment: read data -> data normalization -> data separation -> train model -> prediction -> write results
-- Command: th main.lua
-- Result: Generate deep learning model with trained parameters and prediction error
require 'torch'
require 'math'

--Data parameters:
number = 131 --Total number of training + testing data
feature = 9  -- First with object value and rest with feature number
train_data_ratio = 0.8 --Percentage of data that is used for training
validation_type = 'k-fold' --Support two modes validation: 'normal' and 'k-fold' validation
k_fold = 10 --Set k-fold number

--Model parameters:
local opt = {
  nonlinearity_type = 'sigmoid',
  training_iterations = 150, -- note: the code uses *batches*, not *minibatches*, now.
  print_every = 25,          -- how many iterations to skip between printing the loss
  model_type = 'convnet',        -- Currently only support 'mlp', 'convnet', and 'rnn'
  L1_Reg = 0,             -- L1 regularization parameter; set to 0 if no use
  L2_Reg = 0                 -- L2 regularization parameter; set to 0 if no use  
}

--Load txt file data:
local file = 'Data_Preprocess.txt'
local f = io.open(file,'rb')
X = torch.Tensor(number,feature):zero()
local i = 1
for line in io.lines(file) do
  local j = 1
  for word in line:gmatch("%w+.%w+") do
      X[i][j] = word
      j = j + 1
  end
  i = i + 1
end

--Data normalization with mean and standard variation:
m = torch.mean(X,1)
s = torch.std(X,1)
Xs = torch.Tensor(number,feature):zero()
Xs = torch.cdiv(X - m:repeatTensor(number,1),s:repeatTensor(number,1))

if validation_type == 'normal' then

--Data randomization to separate it into training and testing data set
torch.manualSeed(1)    -- fix random seed so program runs the same every time
num_train = torch.floor(number * train_data_ratio)
num_test = number - num_train
index = torch.randperm(number)
X_train = torch.Tensor(num_train,feature):zero()
X_test = torch.Tensor(num_test,feature):zero()
for i = 1,number do
  if i <= num_train then
    X_train[i] = Xs[index[i]]
  else
    X_test[i-num_train] = Xs[index[i]]
  end
end

--Create deep learning model
local train = require 'train'

-- Train deep learning model
print('Start training ' .. opt.model_type .. ' model......')
model, losses, losses_test = train(opt, X_train, X_test)

--Test performance
local Test_Prediction = torch.Tensor(num_test):zero()
local Test_Prediction_Loss = torch.Tensor(num_test):zero()
local X_test_sized = torch.Tensor(8,1):zero()
for i=1, num_test do
  if opt.model_type == 'mlp' or opt.model_type == 'rnn' then
    Test_Prediction[i] = model:forward(X_test[i][{{2,9}}])
  elseif opt.model_type == 'convnet' then
    X_test_sized:copy(X_test[i][{{2,9}}])
    Test_Prediction[i] = model:forward(X_test_sized)
  end
    
    Test_Prediction_Loss[i] = torch.pow(Test_Prediction[i] - X_test[i][1],2)
end

print('Prediction MSE is: ')
print(torch.cumsum(Test_Prediction_Loss)[num_test]/num_test)

--Predict actual value for the all data set and write to file
--Write to txt file data:
print('Writing all predicted data to file......')
local file = 'Data_Prediction.txt'
local fo = io.open(file,'wb')
local Xs_sized = torch.Tensor(8,1):zero()
local Prediction_All = {}
for i=1, number do
  if opt.model_type == 'mlp' or opt.model_type == 'rnn' then 
    Prediction_All = model:forward(Xs[i][{{2,9}}])
  elseif opt.model_type == 'convnet' then
    Xs_sized:copy(Xs[i][{{2,9}}])
    Prediction_All = model:forward(Xs_sized)
  end
    Prediction_All = Prediction_All * s[1][1] + m[1][1]
    fo:write(tostring(i),". ",tostring(Prediction_All[1]),"\n")
end

-- Plot loss functions:
gnuplot.figure()
gnuplot.xlabel('Iterations')
gnuplot.ylabel('MSE')
gnuplot.plot({'Training',
  torch.range(1, #losses), -- x-coordinates
  torch.Tensor(losses),    -- y-coordinates
  '-'},
  {'Testing',
  torch.range(1, #losses_test), -- x-coordinates
  torch.Tensor(losses_test),    -- y-coordinates
  'o'}
  )

elseif validation_type == 'k-fold' then
  --Separate all data into k-folds
  num_per_fold = torch.ceil(number/k_fold)
  num_per_fold_last = number - num_per_fold * (k_fold-1)
  --Create deep learning model
  local train = require 'train'
  local loss_k_fold = 0
  for i=1, k_fold do
    print('Train the ' .. tostring(i) .. '-th fold cross-validation')
    --subsampling the training set and test:
    if i~= 1 and i~= k_fold then
      local X_train_sub = torch.Tensor(number - num_per_fold,feature):zero()
      local X_test_sub = torch.Tensor(num_per_fold,feature):zero()
      X_test_sub:copy(Xs:sub(1+num_per_fold*(i-1),num_per_fold*i,1,feature))
      X_train_sub:sub(1,num_per_fold*(i-1),1,feature):copy(Xs:sub(1,num_per_fold*(i-1),1,feature))
      X_train_sub:sub(1+num_per_fold*(i-1),number-num_per_fold,1,feature):copy(Xs:sub(1+num_per_fold*i,number,1,feature))
      --Train deep learning model
      print('Start training ' .. opt.model_type .. ' model......')
      model, losses, losses_test = train(opt, X_train_sub,X_test_sub)
      loss_k_fold = loss_k_fold + losses_test[#losses_test]
    elseif i==1 then
      local X_train_sub = torch.Tensor(number - num_per_fold,feature):zero()
      local X_test_sub = torch.Tensor(num_per_fold,feature):zero()
      X_test_sub:copy(Xs:sub(1+num_per_fold*(i-1),num_per_fold*i,1,feature))
      X_train_sub:copy(Xs:sub(1+num_per_fold*i,number,1,feature))
      --Train deep learning model
      print('Start training ' .. opt.model_type .. ' model......')
      model, losses, losses_test = train(opt, X_train_sub,X_test_sub)
      loss_k_fold = loss_k_fold + losses_test[#losses_test]
    elseif i==k_fold then
      local X_train_sub = torch.Tensor(number-num_per_fold_last,feature):zero()
      local X_test_sub = torch.Tensor(num_per_fold_last,feature):zero()
      X_test_sub:copy(Xs:sub(number-num_per_fold_last+1,number,1,feature))
      X_train_sub:copy(Xs:sub(1,number-num_per_fold_last,1,feature))
      --Train deep learning model
      print('Start training ' .. opt.model_type .. ' model......')
      model, losses, losses_test = train(opt, X_train_sub,X_test_sub)
      loss_k_fold = loss_k_fold + losses_test[#losses_test]
    end
  end
  print('K-fold cross validation loss is:')
  print(loss_k_fold/k_fold)
else
  print('No such validation type supported!')
end