require 'xlua'
require 'optim'
require 'cunn'
require 'augmentation'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 64)          batch size
   -r,--learningRate          (default 1)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --initial_model	      (default '')           use which pretrained model to initialize
   --do_augment               (default 1)            whether do augmentation
]]

print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
if opt.initial_model == '' then
    model:add(dofile('models/'..opt.model..'.lua'):cuda())
else
    initial_model = torch.load(opt.initial_model)
    model:add(initial_model)
end
model:add(nn.LogSoftMax():cuda())
model:get(2).updateGradInput = function(input) return end

if opt.backend == 'cudnn' then
   require 'cudnn'
   cudnn.convert(model:get(3), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'

--load extra data
extra_data_1 = torch.load('./new_extra_1.t7')
provider.trainData.data = extra_data_1.data
provider.trainData.labels = extra_data_1.labels

--provider.trainData.data = provider.trainData.data:float()
provider.valData.data = provider.valData.data:float()
--modify label data of validation set, 3 -> (0,0,1,0,0,0,0,0,0,0)
val_label = torch.zeros(provider.valData.data:size(1), 10)
for i = 1, provider.valData.data:size(1) do
    val_label[i][provider.valData.labels[i]] = 1
end
val_label_original = provider.valData.labels
provider.valData.labels = val_label

-- add very tiny value to label data, in case of computing log(0)
tiny_val = 0.00000000001
provider.trainData.labels = provider.trainData.labels + tiny_val
provider.valData.labels = provider.valData.labels + tiny_val

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))
valLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (val set)'}
valLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
-- change the cross entropy loss to KL divergence loss, which measures the difference of two distribution
--criterion = nn.CrossEntropyCriterion():cuda()
criterion = nn.DistKLDivCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  --local targets = torch.CudaTensor(opt.batchSize)
  local targets = torch.CudaTensor(opt.batchSize, 10)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  total_loss = 0.0
  for t,v in ipairs(indices) do
    --xlua.progress(t, #indices)

    local raw_inputs = provider.trainData.data:index(1,v)
    local inputs
    if opt.do_augment == 1 then
        inputs = augment(raw_inputs)
    else
        inputs = raw_inputs
    end
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      total_loss = total_loss + f
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)
        
      --change one-hot label to index label
      --confusion:batchAdd(outputs, targets)
      m,new_target = targets:max(2)
      confusion:batchAdd(outputs, new_target)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)

    if (t - 1)%10 == 0 then
        print(string.format('training %d.%d, avg_loss:%f', epoch, t, total_loss/t))
    end
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function val()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." valing")
  local bs = 25
  index = 0
  val_loss = 0.0
  for i=1,provider.valData.data:size(1),bs do
    local outputs = model:forward(provider.valData.data:narrow(1,i,bs))
    local targets = provider.valData.labels:narrow(1, i, bs)
    val_loss = val_loss + criterion:forward(outputs, targets) 
    --change one-hot label to inde label
    --confusion:batchAdd(outputs, provider.valData.labels:narrow(1,i,bs))
    mini_batch_label = provider.valData.labels:narrow(1, i, bs)
    m,mini_batch_label = mini_batch_label:max(2)
    confusion:batchAdd(outputs, mini_batch_label)
    index = index + 1
  end
  local avg_loss = val_loss / index
  confusion:updateValids()
  print('val accuracy:', confusion.totalValid * 100)
  print('val loss: ', avg_loss)
  
  if valLogger then
    paths.mkdir(opt.save)
    valLogger:add{train_acc, confusion.totalValid * 100}
    valLogger:style{'-','-'}
    valLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/val.log.eps %s/val.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/val.png -out %s/val.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/val.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 25 == 0 then
    local filename = paths.concat(opt.save, ('model_%d.net'):format(epoch))
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  val()
end


