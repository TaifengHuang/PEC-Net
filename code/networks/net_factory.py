from networks.pecnet import PEC_Net
import argparse



def net_factory(net_type="unet", in_chns=3, class_num=1):

    if net_type == "pecnet":
        net = PEC_Net(in_chns=in_chns, class_num=class_num).cuda()

    else:
        net = None
    return net
