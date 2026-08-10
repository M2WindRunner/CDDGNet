import torch.nn as nn

from losses.compound_loss import BceDiceLoss


class SelectLoss(nn.Module):
    def __init__(self, loss_name):
        super(SelectLoss, self).__init__()
        if loss_name == "bce+dice":
            self.loss = BceDiceLoss()
        else:
            raise Exception('Error. This loss function hasn\'t been defined: {}'.format(loss_name))

    def forward(self, outs, labels, weight=(1, 1), aux_weight=0.8):
        loss = 0
        aux_loss = 0

        for i, label in enumerate(labels):
            for j, out in enumerate(outs[i::len(labels)]):
                if j == 0:
                    loss += self.loss(out, label) * weight[i]
                else:
                    aux_loss += self.loss(out, label) * weight[i]
        loss = loss / len(labels)
        aux_loss = aux_loss / (len(outs) - len(labels)) * aux_weight if (len(outs) != len(labels)) else 0

        return loss + aux_loss
