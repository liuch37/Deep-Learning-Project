--Author: Chun-Hao Liu
-- Date: 06/23/2016
-- Goal: Deep learning model creation
-- Result: Generate deep learning model including multiple linear perceptron, convolutional neuralnetwork, and recurrent neural nestwork 
require 'nn'
require 'math'
require 'rnn'

function create_model(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  if opt.model_type == 'mlp' then
    local n_inputs = 8
    local embedding_dim = 4
    local n_classes = 1

    -- OUR MODEL:
    --     linear -> sigmoid -> linear
    model = nn.Sequential()
    model:add(nn.Linear(n_inputs, embedding_dim))

    if opt.nonlinearity_type == 'sigmoid' then
      model:add(nn.Sigmoid())
    else
      error('undefined nonlinearity_type ' .. tostring(opt.nonlinearity_type))
    end 

    model:add(nn.Linear(embedding_dim, n_classes))
  elseif opt.model_type == 'convnet' then
    local n_inputs = 8
    local kernal = 4
    local kw = 2
    local embedding_dim = 4
    local n_classes = 1
    -- Convolutional Model:
    model = nn.Sequential()
    model:add(nn.TemporalConvolution(1,kernal,kw)) --8x1 goes in, 7x4 goes out
    model:add(nn.TemporalMaxPooling(kw))   --7x4 goes in, 3x4 goes out 
    model:add(nn.View(3*4))
    model:add(nn.Linear(3*4,n_classes))
  elseif opt.model_type == 'rnn' then
    local n_inputs = 8
    local hidden = 10
    local n_classes = 1
    local rho = 50
    --Recurrent Neural Network Model:
    r = nn.Recurrent(
        hidden, nn.Linear(n_inputs,hidden),
        nn.Linear(hidden,hidden), nn.Sigmoid(),
        rho
    )
    model = nn.Sequential()
    model:add(r)
    model:add(nn.Linear(hidden,1))
  else
    error('undefined training model type' .. tostring(opt.model_type))
  end

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  --local criterion = nn.ClassNLLCriterion()
  criterion = nn.MSECriterion()
  return model, criterion
end

return create_model

