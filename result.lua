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
print(model)


--load and prepocess test data
----firstly load mean and std of training data
provider = torch.load('./provider.t7')

----then load the raw test data
raw_test = torch.load('./stl-10/test.t7b')

test_size = 8000
channel = 3
height = 96
width = 96
test_data = {
    data = torch.Tensor(),
    labels = torch.Tensor(),
    size = function() return test_size end
}

test_data.data, test_data.labels = parseDataLabel(raw_test.data, 8000, channel, height, width)
test_data.data = test_data.data:float()
test_data.labels = test_data.labels:float()

----normalize the test data using the same method as what is used to process training data
local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
for i = 1, test_data.size() do
--for i = 1, 100 do
    xlua.progress(i, test_data.size())

    local rgb = test_data.data[i]
    local yuv = image.rgb2yuv(rgb)

    yuv[{1}] = normalization(yuv[{{1}}])
    test_data.data[i] = yuv;
end

test_data.data:select(2,2):add(-provider.trainData.mean_u)
test_data.data:select(2,2):div(provider.trainData.std_u)

test_data.data:select(2,3):add(-provider.trainData.mean_v)
test_data.data:select(2,3):div(provider.trainData.std_v)

test_data.data = test_data.data:cuda()
test_data.labels = test_data.labels:cuda()

--load model
file = io.open('res.txt', 'w')
file:write('Id,Prediction\n')
model:evaluate()
confusion = optim.ConfusionMatrix(10)
local bs = 100
for i = 1,test_data.size(),bs do
    xlua.progress(i, test_data.size())
    local outputs = model:forward(test_data.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, test_data.labels:narrow(1,i,bs))
    --print(outputs)       
    for j = 1,bs do
        m,p = outputs[j]:max(1)
        str = string.format('%d,%d\n', i+j-1, p[1]) 
        --file:write(i +j -1, .. ',' .. p[1] .. '\n')
        file:write(str)
    end
end
file:close(file)
confusion:updateValids()
print('\n')
print('accuracy: ', confusion.totalValid)
