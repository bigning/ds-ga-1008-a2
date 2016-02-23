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
       new_img = image.rotate(ori_img, rnd_r[1])

       --translation
       max_trans_pixels = 10
       rnd_pixel_x = torch.rand(1) * 2 * max_trans_pixels
       rnd_pixel_x = rnd_pixel_x - max_trans_pixels
       rnd_pixel_y = torch.rand(1) * 2 * max_trans_pixels
       rnd_pixel_y = rnd_pixel_y - max_trans_pixels
       new_img = image.translate(new_img, rnd_pixel_x[1], rnd_pixel_y[1])

       --scale
       max_ratio = 0.1
       rnd_ratio = torch.rand(1) * 2 * max_ratio
       rnd_ratio = rnd_ratio - max_ratio
       rnd_ratio = rnd_ratio + 1
       new_size = rnd_ratio * img_size
       new_size = math.floor((new_size + 0.5)[1])
       empty_img = new_img:clone()
       empty_img[{}] = 0
       new_img = image.scale(new_img, new_size, new_size)
       if new_size > img_size then 
           x1 = math.floor((new_size - img_size) / 2)
           if x1 < 1 then x1 = 1 end 
           new_img = image.crop(new_img, x1, x1, x1 + img_size , x1 + img_size )
       else
           x1 = math.floor((img_size - new_size) / 2)
           if x1 < 1 then
               x1 = 1
           end
           empty_img[{{}, {x1, x1 + new_size - 1}, {x1, x1 + new_size -1} }] = new_img
           new_img = empty_img
       end

       if new_inputs[i]:size(2) ~= new_img:size(2) then
            print(new_inputs[i]:size(2))
            print(new_img:size(2))
            print(new_size)
       end

       new_inputs[i] = new_img

       a = torch.random(1000)
       if a == 1 then
	   compare = torch.Tensor(3, img_size, img_size * 2)
	   compare[{{}, {1, img_size}, {1, img_size}}] = inputs[i]
	   compare[{{}, {1, img_size}, {img_size + 1, img_size * 2}}] = new_img 
           maxx = compare:max()
           minn = compare:min()
           compare = (compare - minn) / (maxx-minn)
           name = string.format('img/a_%f_%f_%f.png', rnd_r[1], rnd_pixel_x[1], rnd_ratio[1])
           image.save(name, compare)
       end

    end
    collectgarbage()
    return new_inputs
end
