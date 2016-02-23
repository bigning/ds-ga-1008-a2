require 'image';
require 'math'

function augment(inputs)
    new_inputs = inputs:clone()
    mini_batch_size = inputs:size(1)
    img_size = inputs:size(3)
    for i = 1, mini_batch_size do
       ori_img = new_inputs[i]

       --rotation
       max_angel = 10
       max_r = 10*3.1415/180
       rnd_r = torch.rand(1) * 2 * max_r
       rnd_r = rnd_r - max_r
       new_img = image.rotate(ori_img, rnd_r)

       --translation
       max_trans_pixels = 10
       rnd_pixel_x = torch.rand(1) * 2 * max_trans_pixels
       rnd_pixel_x = rnd_pixel_x - max_trans_pixels
       rnd_pixel_y = torch.rand(1) * 2 * max_trans_pixels
       rnd_pixel_y = rnd_pixel_y - max_trans_pixels
       new_img = image.translate(new_img, rnd_pixel_x, rnd_pixel_y)

       --scale
       max_ratio = 0.1
       rnd_ratio = torch.rand(1) * 2 * max_ratio
       rnd_ratio = rnd_ratio - max_ratio
       rnd_ratio = 1 + rnd_ratio
       new_size = rnd_ratio * img_size
       new_size = math.floor(new_size + 0.5)
       empty_img = new_img:clone()
       empty_img[{}] = 0
       new_img = image.scale(new_img, new_size, new_size)
       if new_size > img_size then 
           x1 = math.floor((new_size - img_size) / 2)
           if x1 < 1 then x1 = 1 end 
           new_img = image.crop(new_img, x1, x1, x1 + img_size - 1, x1 + img_size - 1)
       else
           x1 = math.floor((img_size - new_size) / 2)
           if x1 < 1 then
               x1 = 1
           end
           empty_img[{{}, {x1, x1 + img_size - 1}, {x1, x1 + img_size -1} }] = new_img
           new_img = empty_img
       end

       new_inputs[i] = new_img
    end
    collectgarbage()
    return new_inputs
end
