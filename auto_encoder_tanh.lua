require 'xlua'
require 'optim'
require 'torch'
require 'nn'
require 'cunn'
require 'augmentation'

require 'provider'
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:option('-save', 'logs/autoencoder', 'save_path')
cmd:option('-lr', 1.0, 'learning rate')
cmd:option('-weightDecay', 0.0005, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-learningRateDecay', 1e-7, 'learning rate decay')
cmd:option('-epoch_step', 25, 'epoch step')
cmd:option('-batchSize', 64, 'batch size')
cmd:option('-do_augment', 1, 'augmentation flag')
cmd:option('-max_epoch', 300, 'max epoch')
opt = cmd:parse(arg or {})


print 'loading and preprocessing data'

function preprocess(batch_data)
    for i = 1, batch_data:size(1) do
        maxx = batch_data[i]:max()
        minn = batch_data[i]:min()
        batch_data[i] = (batch_data[i] - minn) / (maxx - minn)
    end
end

data_size = 50000
val_data_size = 1000
local raw_extra_data = torch.load('./stl-10/extra_1.t7b')
local raw_val_data = torch.load('./stl-10/val.t7b')
train_data = {data = torch.Tensor(), labels = torch.Tensor(), 
                     size = function() return data_size end}
val_data = {data = torch.Tensor(), labels = torch.Tensor(),
            size = function() return val_data_size end}

train_data.data, train_data.labels = parseDataLabel(raw_extra_data.data, data_size, 3, 96, 96)
val_data.data, val_data.labels = parseDataLabel(raw_val_data.data, 1000, 3, 96, 96)


train_data.data  = train_data.data:float()
val_data.data = val_data.data:float()

preprocess(train_data.data)
preprocess(val_data.data)


--building model
--encoder
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
encoder:add(nn.SpatialBatchNormalization(64, 1e-3))
encoder:add(nn.Tanh())
encoder:add(nn.SpatialMaxPooling(4, 4, 4, 4))
encoder = encoder:cuda()

--decoder
decoder = nn.Sequential()
decoder:add(nn.SpatialMaxUnpooling(encoder:get(4)))
decoder:add(nn.SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))
decoder:add(nn.Sigmoid())
decoder = decoder:cuda()

--final model
model = nn.Sequential()
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(encoder)
model:add(decoder)
model:get(1).updateGradInput = function(input) return end

print(model)
paths.mkdir(opt.save)
--val_logger = optim.Logger(paths.concat(opt.save, 'val.log'))
--val_logger:setNames{'mean l2 loss(train), mean l2 loss(val)'}
--val_logger.showPlot = false

parameters, gradParameters = model:getParameters()

criterion = nn.MSECriterion():cuda()

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}

input_size = 96

function train()
    model:training()
    epoch = epoch or 1
    
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate /2  end

    local targets = torch.CudaTensor(opt.batchSize, 3, input_size, input_size)
    local indices = torch.randperm(train_data.size()):long():split(opt.batchSize)
    indices[#indices] = nil

    local tic = torch.tic()
    total_loss = 0.0
    for t,v in ipairs(indices) do
        local raw_inputs = train_data.data:index(1, v)
        local inputs
        if opt.do_augment == 1 then
            inputs = augment(raw_inputs)
        else
            inputs = raw_inputs
        end
        targets:copy(train_data.data:index(1, v)):cuda()

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            total_loss = total_loss + f
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)

        if (t - 1) % 10 == 0 then
            print(string.format('trainding %d.%d, avg_loss: %f', epoch, t, total_loss/t))
        end
    end
    epoch = epoch + 1
end


function val()
    model:evaluate()
    local bs = 25
    local val_loss = 0.0
    index = 0
    for i = 1, val_data.size(), bs do
        local outputs = model:forward(val_data.data:narrow(1, i, bs))
        local targets = val_data.data:narrow(1, i, bs):cuda()
        val_loss = val_loss + criterion:forward(outputs, targets)
        index = index + 1
        if i == 1 then
 	    input_img = val_data.data[1]
            output_img = outputs[1] 
            new_img = torch.Tensor(3, input_img:size(2), input_img:size(3) * 2)
            new_img[{{}, {}, {1, input_img:size(3)}}] = input_img
            new_img[{{}, {}, { input_img:size(3) + 1,  input_img:size(3)*2}}] =output_img 
            save_name = string.format('auto_encoder_vis/%d.png', epoch)
            image.save(save_name, new_img)
        end
    end

    print('val accuracy: ', val_loss / index)
    if val_logger then
        paths.mkdir(opt.save)
        val_logger:add{total_loss / t, val_loss / index}
    end

    if epoch % 25 ==0 then
        local filename = paths.concat(opt.save, ('model_%d.net'):format(epoch))
        torch.save(filename, model)
    end
end

for i = 1, opt.max_epoch do
    train()
    val()
end
