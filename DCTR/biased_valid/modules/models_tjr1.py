import torch
from modules.layers import MultiLayerPerceptron, FactorizationMachine, FeaturesLinear, FeatureEmbedding
import modules.layers as layer


class BasicModel(torch.nn.Module):
    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.field_num = opt["field_num"]
        self.embedding = FeatureEmbedding(self.feature_num, self.latent_dim)
        self.res_embedding = FeatureEmbedding(self.feature_num, self.latent_dim)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, field_num)``

        """

    def reg(self):
        return 0.0


class FM(BasicModel):
    def __init__(self, opt):
        super(FM, self).__init__(opt)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_embedding = self.embedding(x)
        output_fm = self.fm(x_embedding)
        logit = output_fm
        return logit

    def res_forward(self, x):
        res_x_embedding = torch.sigmoid(self.res_embedding(x))
        res_output_fm = self.fm(res_x_embedding)
        res_logit = res_output_fm
        return res_logit

    def predict(self, x):
        return self.forward(x) - self.res_forward(x)


class DNN(BasicModel):
    def __init__(self, opt):
        super(DNN, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn
        return logit

    def res_forward(self, x):
        res_x_embedding = torch.sigmoid(self.res_embedding(x))
        res_x_dnn = res_x_embedding.view(-1, self.dnn_dim)
        res_output_dnn = self.dnn(res_x_dnn)
        res_logit = res_output_dnn
        return res_logit

    def predict(self, x):
        return self.forward(x) - self.res_forward(x)


class DeepFM(FM):
    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        embed_dims = opt["mlp_dims"]
        dropout = opt["mlp_dropout"]
        use_bn = opt["use_bn"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.dnn = MultiLayerPerceptron(self.dnn_dim, embed_dims, dropout, use_bn=use_bn)

    def forward(self, x):
        x_embedding = self.embedding(x)
        output_fm = self.fm(x_embedding)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_dnn = self.dnn(x_dnn)
        logit = output_dnn + output_fm
        return logit

    def res_forward(self, x):
        res_x_embedding = torch.sigmoid(self.res_embedding(x))
        res_output_fm = self.fm(res_x_embedding)
        res_x_dnn = res_x_embedding.view(-1, self.dnn_dim)
        res_output_dnn = self.dnn(res_x_dnn)
        res_logit = res_output_dnn + res_output_fm
        return res_logit

    def predict(self, x):
        return self.forward(x) - self.res_forward(x)


class DeepCrossNet(BasicModel):
    def __init__(self, opt):
        super(DeepCrossNet, self).__init__(opt)
        cross_num = opt["cross"]
        mlp_dims = opt["mlp_dims"]
        use_bn = opt["use_bn"]
        dropout = opt["mlp_dropout"]
        self.dnn_dim = self.field_num * self.latent_dim
        self.cross = layer.CrossNetwork(self.dnn_dim, cross_num)
        self.dnn = MultiLayerPerceptron(self.dnn_dim, mlp_dims, output_layer=False, dropout=dropout, use_bn=use_bn)
        self.combination = torch.nn.Linear(mlp_dims[-1] + self.dnn_dim, 1, bias=False)

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_dnn = x_embedding.view(-1, self.dnn_dim)
        output_cross = self.cross(x_dnn)
        output_dnn = self.dnn(x_dnn)
        comb_tensor = torch.cat((output_cross, output_dnn), dim=1)
        logit = self.combination(comb_tensor)
        return logit

    def res_forward(self, x):
        res_x_embedding = torch.sigmoid(self.res_embedding(x))
        res_x_dnn = res_x_embedding.view(-1, self.dnn_dim)
        res_output_cross = self.cross(res_x_dnn)
        res_output_dnn = self.dnn(res_x_dnn)
        res_comb_tensor = torch.cat((res_output_cross, res_output_dnn), dim=1)
        res_logit = self.combination(res_comb_tensor)
        return res_logit

    def predict(self, x):
        return self.forward(x) - self.res_forward(x)
