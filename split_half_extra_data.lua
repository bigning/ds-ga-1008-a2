require 'torch'

raw_extra = torch.load('./stl-10/extra.t7b')
extra_data_1 = {}
extra_data_2 = {}

original_size = 100000
shuffle = torch.randperm(original_size)
extra_data_1[1] = {}
extra_data_2[1] = {}
for i = 1, 50000 do
    extra_data_1[1][i] = raw_extra.data[1][i]
    if i % 100 == 0 then
        print(i)
    end
end
for i = 50001, 100000 do
    extra_data_2[1][i - 50000] = raw_extra.data[1][i]
    if i % 100 == 0 then
        print(i)
    end
end

extra_1 = {data = extra_data_1}
extra_2 = {data = extra_data_2}
torch.save('./stl-10/extra_1.t7b', extra_1)
torch.save('./stl-10/extra_2.t7b', extra_2)

