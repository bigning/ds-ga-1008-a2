require 'nn'
require 'image'
require 'xlua'
require 'provider'
require 'cunn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:option('-model','logs/sample/model.net', 'trained model path')
opt = cmd:parse(arg or {})
model = torch.load(opt.model)
model:add(nn.SoftMax():cuda())
print(model)


--load and prepocess test data
----firstly load mean and std of training data
provider = torch.load('./provider.t7')

----then load the raw test data
raw_test = torch.load('./stl-10/extra_1.t7b')

test_size = 50000
debug_size = 50000
channel = 3
height = 96
width = 96
test_data = {
    data = torch.Tensor(),
    labels = torch.Tensor(),
    size = function() return test_data.data:size(1) end
}

test_data.data, test_data.labels = parseDataLabel(raw_test.data, test_size, channel, height, width)
raw_test = nil
collectgarbage()

test_data.data = test_data.data[{{1,debug_size}}]
test_data.labels = test_data.labels[{{1, debug_size}}]

test_data.data = test_data.data:float()
test_data.labels = test_data.labels:float()

----normalize the test data using the same method as what is used to process training data
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1, test_data.size() do
--for i = 1, 100 do
    --xlua.progress(i, test_data.size())
    if i % 100 == 0 then
       print(string.format('preprocessing %d', i))
    end

    local rgb = test_data.data[i]
    local yuv = image.rgb2yuv(rgb)

    yuv[{1}] = normalization(yuv[{{1}}])
    test_data.data[i] = yuv;
end

test_data.data:select(2,2):add(-provider.trainData.mean_u)
test_data.data:select(2,2):div(provider.trainData.std_u)

test_data.data:select(2,3):add(-provider.trainData.mean_v)
test_data.data:select(2,3):div(provider.trainData.std_v)

test_data.data = test_data.data
test_data.labels = test_data.labels:cuda()

new_lables = torch.Tensor(test_data.size(), 10)

--load model
--file = io.open('res.txt', 'w')
--file:write('Id,Prediction\n')
model:evaluate()
confusion = optim.ConfusionMatrix(10)
local bs = 500
for i = 1,test_data.size(),bs do
    if (i-1) % 100 == 0 then
       print(string.format('predicting %d', i))
    end
    --xlua.progress(i, test_data.size())
    batch_input = test_data.data:narrow(1,i,bs)
    batch_input = batch_input:cuda()
    local outputs = model:forward(batch_input)
    confusion:batchAdd(outputs, test_data.labels:narrow(1,i,bs))
    --print(outputs)       
    for j = 1,bs do
        m,p = outputs[j]:max(1)
        str = string.format('%d,%d\n', i+j-1, p[1]) 
        --file:write(i +j -1, .. ',' .. p[1] .. '\n')
        --file:write(str)
        new_lables[i + j -1] = outputs[j]:float()
    end
end
test_data.labels = new_lables
test_data.data = test_data.data:float()
torch.save('new_extra_1.t7', test_data)
--file:close(file)
confusion:updateValids()
print('\n')
print('accuracy: ', confusion.totalValid)
