require 'nn';
require 'torch';
require 'image'

model = torch.load('model_300.cpu.net')
conv_layer = model:get(1)

weights = conv_layer.weight
save_img = torch.zeros( 8*3 *3, 8*3*3)
index = 1
for i = 1,8 do
    for j = 1,8 do
        save_img[{ {(i-1) * 9 + 1, (i -1 )*9 +3}, {(j-1) *9 + 1, (j - 1)*9 +3}}] = weights[index][1]:double()
        save_img[{ {(i-1) * 9 + 4, (i -1 )*9 +6}, {(j-1) *9 + 4, (j-1)*9 +6}}] = weights[index][2]:double()
        save_img[{ {(i-1) * 9 + 7, (i -1 )*9 +9}, {(j-1) *9 + 7, (j-1)*9 +9}}] = weights[index][3]:double()
        index = index + 1
    end
end
save_img = save_img - save_img:min()
save_img = save_img / save_img:max()
new_img = image.scale(save_img, 501,501, 'simple')

image.save('1st_conv.jpg', new_img)
