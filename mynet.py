from queue import Queue
import copy

from modules.mymix_op import *
from utils import *
from models.normal_nets import *

class supernet(ProxylessNASNets):
    def __init__(self, conv_candidates):
        self._redundant_modules = None
        self._unused_modules = None
        # first_conv = ConvLayer(3, 32, kernel_size=5, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # input_channel = 32
        # first_cell_width = 32
        # n_cell_stages = [2 , 1 , 1 , 1  , 1  , 1]
        # width_stages =  [32, 64, 64, 128, 128, 256]
        # stride_stages = [2 , 2 , 1 , 2  , 1  , 2]
        # first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
        #     ['3x3_Conv'], input_channel, first_cell_width, 1, 'weight_bn_act'))
        # if first_block_conv.n_choices == 1:
        #     first_block_conv = first_block_conv.candidate_ops[0]
        # first_block = MobileInvertedResidualBlock(first_block_conv, None)
        # net_block = [first_block]
        # # n_cell_stages = [2 , 2 , 2  , 1]
        # # width_stages =  [32, 64, 128, 256]
        # # stride_stages = [2 , 2 , 2  , 1]
        # # net_block = []
        # for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
        #     for i in range(n_cell):
        #         stride = s if i == 0 else 1
        #         if input_channel == width: 
        #             if stride == 1 :
        #                 modified_conv_candidates = conv_candidates + ['Zero']   
        #             else :
        #                 modified_conv_candidates = conv_candidates + ['2x2_avg_pool', '2x2_max_pool']                    
        #         else:
        #             modified_conv_candidates = conv_candidates + ['1x1_Conv']
        #         conv_op = MixedEdge(candidate_ops=build_candidate_ops(
        #             modified_conv_candidates, input_channel, width, stride, 'weight_bn_act') )
        #         # shortcut
        #         if stride == 1 and input_channel == width:
        #             shortcut = IdentityLayer(input_channel, input_channel)
        #         else:
        #             shortcut = ConvLayer(input_channel, width,kernel_size=1, stride=stride)
        #         inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
        #         net_block.append(inverted_residual_block)

        #         input_channel = width
        # blocks = nn.ModuleList(net_block)
        # last_channel = width_stages[-1]
        # feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # classifier = LinearLayer(last_channel, 10, dropout_rate=0.2)
        first_conv = ConvLayer(
            3, 32, kernel_size=5, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        input_channel = 32
        first_cell_width = 32
        # first block
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_Conv'], input_channel, first_cell_width, 1, 'weight_bn_act'))
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width
       # blocks
        net_block = [first_block]
        n_cell_stages = [2, 1, 1, 1, 1, 1]
        width_stages = [32, 64, 64, 128, 256]
        stride_stages = [2, 1, 2, 1, 2, 1]
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                stride = s if i == 0 else 1
                if input_channel == width: 
                    if stride == 1 :
                        modified_conv_candidates = conv_candidates + ['Zero']   
                    else :
                        modified_conv_candidates = conv_candidates + ['2x2_avg_pool', '2x2_max_pool']                    
                else:
                    modified_conv_candidates = conv_candidates + ['1x1_Conv']
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride, 'weight_bn_act') )
                # shortcut
                if stride == 1 and input_channel == width:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut)
                net_block.append(inverted_residual_block)

                input_channel = width
        blocks = nn.ModuleList(net_block)
        last_channel = width_stages[-1]
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        classifier = LinearLayer(last_channel, 10, dropout_rate=0.2)
        
        super(supernet, self).__init__(first_conv, blocks, feature_mix_layer, classifier)
 
    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            # try:
            #     m.set_arch_param_grad()
            # except AttributeError:
            #     print(type(m), ' do not support `set_arch_param_grad()`')
            m.set_arch_param_grad()

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        # print("mynet line 148" , MixedEdge.MODE)
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self, latency_model: LatencyEstimator):
        expected_latency = 0
        # first conv
        # expected_latency += latency_model.predict('Conv', [224, 224, 3], [112, 112, self.first_conv.out_channels])
        # feature mix layer
        # expected_latency += latency_model.predict(
        #     'Conv_1', [7, 7, self.feature_mix_layer.in_channels], [7, 7, self.feature_mix_layer.out_channels]
        # )
        # classifier
        # expected_latency += latency_model.predict(
        #     'Logits', [7, 7, self.classifier.in_features], [self.classifier.out_features]  # 1000
        # )
        # blocks
        # fsize = 112
        # for block in self.blocks:
        #     shortcut = block.shortcut
        #     if shortcut is None or shortcut.is_zero_layer():
        #         idskip = 0
        #     else:
        #         idskip = 1

        #     mb_conv = block.mobile_inverted_conv
        #     if not isinstance(mb_conv, MixedEdge):
        #         if not mb_conv.is_zero_layer():
        #             out_fz = fsize // mb_conv.stride
        #             op_latency = latency_model.predict(
        #                 'expanded_conv', [fsize, fsize, mb_conv.in_channels], [out_fz, out_fz, mb_conv.out_channels],
        #                 expand=mb_conv.expand_ratio, kernel=mb_conv.kernel_size, stride=mb_conv.stride, idskip=idskip
        #             )
        #             expected_latency = expected_latency + op_latency
        #             fsize = out_fz
        #         continue

        #     probs_over_ops = mb_conv.current_prob_over_ops
        #     out_fsize = fsize
        #     for i, op in enumerate(mb_conv.candidate_ops):
        #         if op is None or op.is_zero_layer():
        #             continue
        #         out_fsize = fsize // op.stride
        #         op_latency = latency_model.predict(
        #             'expanded_conv', [fsize, fsize, op.in_channels], [out_fsize, out_fsize, op.out_channels],
        #             expand=op.expand_ratio, kernel=op.kernel_size, stride=op.stride, idskip=idskip
        #         )
        #         expected_latency = expected_latency + op_latency * probs_over_ops[i]
        #     fsize = out_fsize
        for block in self.blocks:
          mb_conv = block.mobile_inverted_conv
          if not isinstance(mb_conv, MixedEdge):
              if not mb_conv.is_zero_layer():
                name = "{} {}x{} stride={}".format(mb_conv.module_str, mb_conv.in_channels, mb_conv.out_channels,mb_conv.stride)
                op_latency = latency_model.predict(name) * 10
                expected_latency = expected_latency + op_latency
              continue
          probs_over_ops = mb_conv.current_prob_over_ops
          for i, op in enumerate(mb_conv.candidate_ops):
              if op is None or op.is_zero_layer():
                  continue
              name = "{} {}x{} stride={}".format(op.module_str, op.in_channels, op.out_channels,op.stride)
              op_latency = latency_model.predict(name) * 10
              expected_latency = expected_latency + op_latency * probs_over_ops[i]
        return expected_latency

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop, x = self.first_conv.get_flops(x)
        expected_flops += flop
        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)
        # feature mix layer
        delta_flop, x = self.feature_mix_layer.get_flops(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return ProxylessNASNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)