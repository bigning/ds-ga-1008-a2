require 'cunn';
require 'nn';
require 'torch';
require 'image'

model = torch.load('logs/vgg_sample/model_300.net')
conv_layer = model:get(1)

weights = conv_layer.weight
save_img = torch.Tensor(3, 8*3, 8*3)
index = 1
for i = 1,8 do
    for j = 1,8 do
        save_img[{{}, {(i-1) * 3 + 1, i*3}, {(j-1) *3 + 1, j*3}}] = weights[index]:double()
        index = index + 1
    end
end

image.save('1st_conv.jpg', save_img)
