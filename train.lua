--Author: Chun-Hao Liu
-- Date: 06/21/2016
-- Goal: Deep learning training process
-- Result: Generate deep learning model and return training and testing loss per iteration
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'

create_model = require 'create_model'

local function train(opt, data, data_test)
    ------------------------------------------------------------------------
    -- create model and loss/grad evaluation function
    --
    local model, criterion = create_model(opt)
    local params, grads = model:getParameters()

    -- (re-)initialize weights
    --params:uniform(-0.01, 0.01)

    -- return loss, grad
    local feval = function(x)
      if params ~= x then
        params:copy(x)
      end
      grads:zero()
      -- select a new training sample
     _nidx_ = (_nidx_ or 0) + 1
     if _nidx_ > (#data)[1] then _nidx_ = 1 end
     
     local sample = data[_nidx_]
     local target = sample[{ {1} }]      -- this funny looking syntax allows
     local inputs = sample[{ {2,9} }]    -- slicing of arrays.
     local inputs_sized = torch.Tensor(8,1)
     inputs_sized:copy(inputs)
     if opt.model_type == 'mlp' then
       inputs_select = inputs
     elseif opt.model_type == 'convnet' then
       inputs_select = inputs_sized
     elseif opt.model_type == 'rnn' then
       inputs_select = inputs
     end
      -- forward
      local outputs = model:forward(inputs_select)
      local loss = criterion:forward(outputs, target)
      -- backward
      local dloss_doutput = criterion:backward(outputs, target)
      model:backward(inputs_select, dloss_doutput)
      -- Regularization (L1 and L2):
      if opt.L1_Reg ~= 0 or opt.L2_Reg ~= 0 then
        -- locals:
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        loss = loss + opt.L1_Reg * norm(params,1)
        loss = loss + opt.L2_Reg * norm(params,2)^2/2
        -- Gradients:
        grads:add( sign(params):mul(opt.L1_Reg) + params:clone():mul(opt.L2_Reg) )
      end
      return loss, grads
    end

    ------------------------------------------------------------------------
    -- optimization loop
    --
    local losses = {}
    local losses_test = {}
    local optim_state = {learningRate = 1e-1}

    for i = 1, opt.training_iterations do
      --Calculate loss value for training data per iteration
      current_loss = 0
      for j = 1, (#data)[1] do
        local _, loss = optim.adagrad(feval, params, optim_state)
        current_loss = current_loss + loss[1] -- append the new loss
      end
      losses[#losses+1] = current_loss/(#data)[1]
      --Calculate loss value for testing data per iteration
      current_loss_test = 0
      data_test_sized = torch.Tensor(8,1):zero()
      for j=1, (#data_test)[1] do
        if opt.model_type == 'mlp' or opt.model_type == 'rnn' then
          Test_Prediction = model:forward(data_test[j][{{2,9}}])[1]
        elseif opt.model_type == 'convnet' then
          data_test_sized:copy(data_test[j][{{2,9}}])
          Test_Prediction = model:forward(data_test_sized)[1]
        end 
          current_loss_test = current_loss_test + (Test_Prediction - data_test[j][1])^2
      end
      losses_test[#losses_test+1] = current_loss_test/(#data_test)[1]
      if i % opt.print_every == 0 then 
            print(string.format("iteration %4d, loss = %6.6f", i, losses[i]))
      end
    end
    
    return model, losses, losses_test
end

return train

