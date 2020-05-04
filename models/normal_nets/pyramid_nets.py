# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import json

from utils import *

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_path, "../../../"))
from proxyless_nas.cifar_modules import PyramidTreeNet


def pyramid_base(net_config=None, n_classes=10, bn_param=0, dropout_rate=0):
    net_config_path = download_url(net_config)
    net_config_json = json.load(open(net_config_path, 'r'))
    net_config_json['classifier']['out_features'] = n_classes
    net_config_json['classifier']['dropout_rate'] = dropout_rate
    net = PyramidTreeNet.build_from_config(net_config_json)
    # net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
    # serious question: what do they do in pyramids
    # todo : check what they do
    return net
