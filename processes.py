import torch
from dn3.trainable.processes import StandardClassification


class MultipleParamGroupClassification(StandardClassification):
    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=None, metrics=None, learning_rate=0.01,
                 label_smoothing=None, **kwargs):
        super(MultipleParamGroupClassification, self).__init__(
            classifier, loss_fn, cuda, metrics, learning_rate, label_smoothing, **kwargs)

        self.optimizer = torch.optim.SGD([
            {"params": classifier.init_conv.parameters(), "lr": learning_rate},
            {"params": classifier.depth_conv.parameters(), "lr": learning_rate},
            {"params": classifier.sep_conv.parameters(), "lr": learning_rate},
            {
                "params": classifier.adaptive_input_norm.mean_layer.parameters(),
                "lr": learning_rate * classifier.adaptive_input_norm.mean_lr
            },
            {
                "params": classifier.adaptive_input_norm.scaling_layer.parameters(),
                "lr": learning_rate * classifier.adaptive_input_norm.scale_lr
            },
            {
                "params": classifier.adaptive_input_norm.gating_layer.parameters(),
                "lr": learning_rate * classifier.adaptive_input_norm.gate_lr
            }
        ], lr=learning_rate)
