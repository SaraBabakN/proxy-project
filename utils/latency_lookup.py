%cd /content/drive/My Drive/proxyless/search
import yaml 
from search.modules.mylayers import * 
from search.modules.mymix_op import build_candidate_ops
import time
input_channel = 32
first_cell_width = 32
conv_candidates = ['3x3_Conv','5x5_Conv','7x7_Conv','3x3_DepthConv','5x5_DepthConv','7x7_DepthConv']
look_up_table = {}
presence_time = {}
repeat = 1000
torch.manual_seed(0)
f = ConvLayer(3, 32, kernel_size=5, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act' )
i_shape = f(torch.normal(0, 1, size=(64,3,32,32))).shape
n_cell_stages = [2 , 2 , 2  , 1]
width_stages =  [32, 64, 128, 256]
stride_stages = [2 , 2 , 2  , 1]
for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
    for i in range(n_cell):
        stride = s if i == 0 else 1
        if input_channel == width: 
            if stride == 1 :
                modified_conv_candidates = conv_candidates + ['Zero','Identity']   
            else :
                modified_conv_candidates = conv_candidates + ['2x2_avg_pool', '2x2_max_pool']                    
        else:
            modified_conv_candidates = conv_candidates

        candidate_ops=build_candidate_ops(modified_conv_candidates, input_channel, width, stride, 'weight_bn_act')

        for act in candidate_ops: 
            name = "{} {}x{} stride={}".format(act.module_str, input_channel,width,stride)
            act = act.to("cuda:0")
            print(name)
            if (name in look_up_table): 
                total_time = look_up_table[name]
                presence_time[name] += repeat
            else: 
                total_time = 0 
                presence_time[name] = repeat
            for counter in range (repeat): 
                layer_input = (torch.normal(0, 1, size=i_shape)).to("cuda:0")
                # layer_input.device = "cuda:0"
                start = time.time() 
                act(layer_input)
                total_time += (time.time() - start)
            look_up_table[name] = total_time
        i_shape = act(layer_input).shape
        input_channel = width
last_channel = width_stages[-1]

# act = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
# name = "{} {}x{} stride={}".format(act.module_str, input_channel,width,stride)
# act = act.to("cuda:0")
# print(name)
# if (name in look_up_table): 
#     total_time = look_up_table[name]
#     presence_time[name] += repeat
# else: 
#     total_time = 0 
#     presence_time[name] = repeat
# for counter in range (repeat): 
#     layer_input = (torch.normal(0, 1, size=i_shape)).to("cuda:0")
#     # layer_input.device = "cuda:0"
#     start = time.time() 
#     act(layer_input)
#     total_time += (time.time() - start)
# look_up_table[name] = total_time
# i_shape = act(layer_input).shape

for x in look_up_table: 
    look_up_table[x] = look_up_table[x]/presence_time[x] * 1000
    print(x , look_up_table[x])

with open(r'/content/drive/My Drive/proxyless/search/latency.yaml', 'w') as latency_file:
    documents = yaml.dump(look_up_table, latency_file)
