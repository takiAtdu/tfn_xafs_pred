import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


FLOAT_TYPE = torch.float32
EPSILON = 1e-8

def unit_vectors(v, dim=-1):
    return v / torch.sqrt(torch.sum(v ** 2, dim=dim, keepdim=True) + EPSILON)

def Y_2(rij):
    x = rij[:, :, 0]
    y = rij[:, :, 1]
    z = rij[:, :, 2]
    r2 = torch.clamp(torch.sum(rij ** 2, dim=-1), min=EPSILON)
    output = torch.stack([x * y / r2,
                          y * z / r2,
                          (-x ** 2 - y ** 2 + 2. * z ** 2) / (2 * sqrt(3) * r2),
                          z * x / r2,
                          (x ** 2 - y ** 2) / (2. * r2)],
                         dim=-1)
    return output

def get_eijk():
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return torch.tensor(eijk_, dtype=FLOAT_TYPE)


def norm_with_epsilon(input_tensor, dim=None, keepdim=False):
    return torch.sqrt(torch.clamp(torch.sum(input_tensor ** 2, dim=dim, keepdim=keepdim), min=EPSILON))


def ssp(x):
    return torch.log(0.5 * torch.exp(x) + 0.5)


def rotation_equivariant_nonlinearity(x, nonlin=ssp):
    shape = x.shape
    channels = shape[-2]
    representation_index = shape[-1]

    biases = torch.zeros(channels, dtype=FLOAT_TYPE)

    if representation_index == 1:
        return nonlin(x)
    else:
        norm = norm_with_epsilon(x, dim=-1)
        nonlin_out = nonlin(norm + biases)
        factor = nonlin_out / norm
        return x * factor.unsqueeze(-1)


def difference_matrix(geometry):
    ri = geometry.unsqueeze(1)
    rj = geometry.unsqueeze(0)
    rij = ri - rj
    return rij


def distance_matrix(geometry):
    rij = difference_matrix(geometry)
    dij = norm_with_epsilon(rij, dim=-1)
    return dij


class RadialFunction(nn.Module):
    """
    weights_initializerとbiases_initializerはデフォルトでXavier初期化とゼロ初期化を使います。
    forwardメソッドは、指定された非線形関数（デフォルトではReLU）を使用して入力を処理し、最終的な出力を計算します。
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
        super(RadialFunction, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        if weights_initializer is None:
            weights_initializer = nn.init.xavier_uniform_

        if biases_initializer is None:
            biases_initializer = nn.init.constant_

        self.weights1 = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.biases1 = nn.Parameter(torch.Tensor(hidden_dim))
        self.weights2 = nn.Parameter(torch.Tensor(output_dim, hidden_dim))
        self.biases2 = nn.Parameter(torch.Tensor(output_dim))

        weights_initializer(self.weights1)
        weights_initializer(self.weights2)
        biases_initializer(self.biases1, 0.)
        biases_initializer(self.biases2, 0.)

    def forward(self, inputs, nonlin=F.relu):
        # inputs shape: [N, N, input_dim]
        hidden_layer = nonlin(self.biases1 + torch.tensordot(inputs, self.weights1, dims=([-1], [1])))
        radial = self.biases2 + torch.tensordot(hidden_layer, self.weights2, dims=([-1], [1]))

        # output shape: [N, N, output_dim]
        return radial


class F_0(nn.Module):
    """
    RadialFunctionクラスのインスタンスを持ち、forwardメソッドでその出力を計算し、次元を追加します。
    これにより、出力の形状が[N, N, output_dim, 1]になります。
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
        super(F_0, self).__init__()
        self.radial_function = RadialFunction(input_dim, hidden_dim, output_dim, weights_initializer,
                                              biases_initializer)

    def forward(self, inputs, nonlin=F.relu):
        # [N, N, output_dim, 1]
        radial_output = self.radial_function(inputs, nonlin=nonlin)
        return radial_output.unsqueeze(-1)


class F_1(nn.Module):
    """
    unit_vectors関数は、与えられたベクトルの単位ベクトルを計算し、
    F_1クラスのforwardメソッドでは、RadialFunctionの出力を使用して、
    条件付きでマスクされたラジアル成分と単位ベクトルの積を計算します。
    これにより、出力の形状が[N, N, output_dim, 3]になります。
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
        super(F_1, self).__init__()
        self.radial_function = RadialFunction(input_dim, hidden_dim, output_dim, weights_initializer,
                                              biases_initializer)

    def forward(self, inputs, rij, nonlin=F.relu):
        # [N, N, output_dim]
        radial = self.radial_function(inputs, nonlin=nonlin)

        # Mask out for dij = 0
        dij = torch.norm(rij, dim=-1)
        condition = dij < EPSILON
        masked_radial = torch.where(condition.unsqueeze(-1).expand_as(radial), torch.zeros_like(radial), radial)

        # [N, N, output_dim, 3]
        return unit_vectors(rij).unsqueeze(-2) * masked_radial.unsqueeze(-1)


class F_2(nn.Module):
    """
    Y_2関数は、与えられたベクトルrijの単位ベクトルを計算し、
    F_2クラスのforwardメソッドでは、RadialFunctionの出力を使用して、
    条件付きでマスクされたラジアル成分とY_2の積を計算します。
    これにより、出力の形状が[N, N, output_dim, 5]になります。
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
        super(F_2, self).__init__()
        self.radial_function = RadialFunction(input_dim, hidden_dim, output_dim, weights_initializer,
                                              biases_initializer)

    def forward(self, inputs, rij, nonlin=F.relu):
        # [N, N, output_dim]
        radial = self.radial_function(inputs, nonlin=nonlin)

        # Mask out for dij = 0
        dij = torch.norm(rij, dim=-1)
        condition = dij < EPSILON
        masked_radial = torch.where(condition.unsqueeze(-1).expand_as(radial), torch.zeros_like(radial), radial)

        # [N, N, output_dim, 5]
        return Y_2(rij).unsqueeze(-2) * masked_radial.unsqueeze(-1)


class Filter_0(nn.Module):
    """
    RadialFunctionクラスのインスタンスを持ち、
    forwardメソッドでは、F_0の出力を使用して、指定された次元でアインシュタインの縮約を実行します。
    これにより、出力の形状が[N, output_dim]になります。
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, weights_initializer=None, biases_initializer=None):
        super(Filter_0, self).__init__()
        self.f_0 = F_0(input_dim, hidden_dim, output_dim, weights_initializer, biases_initializer)

    def forward(self, layer_input, rbf_inputs, nonlin=F.relu):
        # [N, N, output_dim, 1]
        f_0_out = self.f_0(rbf_inputs, nonlin=nonlin)

        # [N, output_dim]
        input_dim = layer_input.size(-1)

        # Expand filter axis "j"
        cg = torch.eye(input_dim).unsqueeze(-2)

        # L x 0 -> L
        result = torch.einsum('ijk,abfj,bfk->afi', cg, f_0_out, layer_input.view(-1, layer_input.size(-2), layer_input.size(-1)))

        return result


def filter_1_output_0(layer_input, rbf_inputs, rij, nonlin=F.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    """
    次元が3の場合にアインシュタインの縮約を実行します。他の次元については未実装として例外を投げます。
    """
    # [N, N, output_dim, 3]
    f_1 = F_1(rbf_inputs.size(-1), hidden_dim, output_dim, weights_initializer, biases_initializer)
    F_1_out = f_1(rbf_inputs, rij, nonlin=nonlin)

    # [N, output_dim, 3]
    input_dim = layer_input.size(-1)

    if input_dim == 1:
        raise ValueError("0 x 1 cannot yield 0")
    elif input_dim == 3:
        # 1 x 1 -> 0
        cg = torch.eye(3).unsqueeze(0)
        # [N, output_dim, 1]
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input.squeeze())
    else:
        raise NotImplementedError("Other Ls not implemented")


def filter_1_output_1(layer_input, rbf_inputs, rij, nonlin=F.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    """
    次元が1の場合にアインシュタインの縮約を実行します。他の次元については未実装として例外を投げます。
    """
    # [N, N, output_dim, 3]
    f_1 = F_1(rbf_inputs.size(-1), hidden_dim, output_dim, weights_initializer, biases_initializer)
    F_1_out = f_1(rbf_inputs, rij, nonlin=nonlin)

    # [N, output_dim, 3]
    input_dim = layer_input.size(-1)

    if input_dim == 1:
        # 0 x 1 -> 1
        cg = torch.eye(3).unsqueeze(-1)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_1_out, layer_input.view(-1, layer_input.size(-2), 1))
    elif input_dim == 3:
        # 1 x 1 -> 1
        return torch.einsum('ijk,abfj,bfk->afi', get_eijk(), F_1_out, layer_input.squeeze())
    else:
        raise NotImplementedError("Other Ls not implemented")


def filter_2_output_2(layer_input, rbf_inputs, rij, nonlin=F.relu, hidden_dim=None, output_dim=1,
                      weights_initializer=None, biases_initializer=None):
    # [N, N, output_dim, 3]
    f_2 = F_2(rbf_inputs.size(-1), hidden_dim, output_dim, weights_initializer, biases_initializer)
    F_2_out = f_2(rbf_inputs, rij, nonlin=nonlin)

    # [N, output_dim, 5]
    input_dim = layer_input.size(-1)

    if input_dim == 1:
        # 0 x 2 -> 2
        cg = torch.eye(5).unsqueeze(-1)
        return torch.einsum('ijk,abfj,bfk->afi', cg, F_2_out, layer_input)
    else:
        raise NotImplementedError("Other Ls not implemented")


class SelfInteractionLayerWithoutBiases(nn.Module):
    """
    重みの初期化を行い、入力に対してアインシュタインの縮約を実行し、次元を並べ替えます。
    これにより、出力の形状が[N, output_dim, 2L+1]になります。
    """
    def __init__(self, input_dim, output_dim, weights_initializer=None, biases_initializer=None):
        super(SelfInteractionLayerWithoutBiases, self).__init__()

        if weights_initializer is None:
            weights_initializer = nn.init.orthogonal_

        if biases_initializer is None:
            biases_initializer = nn.init.constant_

        self.weights = nn.Parameter(torch.Tensor(output_dim, input_dim))

        weights_initializer(self.weights)

    def forward(self, inputs):
        # inputs has shape [N, C, 2L+1]
        # input_dim is number of channels
        output = torch.einsum('afi,gf->aig', inputs, self.weights)
        # output shape [N, output_dim, 2L+1]
        return output.permute(0, 2, 1)


class SelfInteractionLayerWithBiases(nn.Module):
    """
    重みとバイアスの初期化を行い、入力に対してアインシュタインの縮約を実行し、バイアスを追加して次元を並べ替えます。
    これにより、出力の形状が[N, output_dim, 2L+1]になります。
    """
    def __init__(self, input_dim, output_dim, weights_initializer=None, biases_initializer=None):
        super(SelfInteractionLayerWithBiases, self).__init__()

        if weights_initializer is None:
            weights_initializer = nn.init.orthogonal_

        if biases_initializer is None:
            biases_initializer = nn.init.constant_

        self.weights = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.biases = nn.Parameter(torch.Tensor(output_dim))

        weights_initializer(self.weights)
        biases_initializer(self.biases, 0.0)

    def forward(self, inputs):
        output = torch.einsum('afi,gf->aig', inputs, self.weights) + self.biases.unsqueeze(0)
        # output shape [N, output_dim, 2L+1]
        return output.permute(0, 2, 1)


class Convolution(nn.Module):
    """
    重みとバイアスの初期化を行い、forwardメソッドでは、各フィルタを適用し、結果をoutput_tensor_listに追加します。
    """
    def __init__(self, rbf_count, weights_initializer=None, biases_initializer=None):
        super(Convolution, self).__init__()

        self.weights_initializer = weights_initializer or nn.init.orthogonal_
        self.biases_initializer = biases_initializer or nn.init.constant_

        self.filter_0 = Filter_0(rbf_count, weights_initializer=self.weights_initializer,
                                      biases_initializer=self.biases_initializer)

    def forward(self, input_tensor_list, rbf, unit_vectors):
        output_tensor_list = {0: [], 1: [], 2: []}

        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                output_dim = tensor.size(-2)

                # L x 0 -> L
                # tensor_out = Filter_0(output_dim, weights_initializer=self.weights_initializer,
                #                       biases_initializer=self.biases_initializer)(tensor, rbf)
                tensor_out = self.filter_0(tensor, rbf)
                m = 0 if tensor_out.size(-1) == 1 else 1
                output_tensor_list[m].append(tensor_out)

                if key == 1:
                    # L x 1 -> 0
                    tensor_out = filter_1_output_0(tensor, rbf, unit_vectors,
                                                   output_dim=output_dim,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.biases_initializer)
                    m = 0 if tensor_out.size(-1) == 1 else 1
                    output_tensor_list[m].append(tensor_out)

                if key == 2:
                    tensor_out = filter_2_output_2(tensor, rbf, unit_vectors,
                                                   output_dim=output_dim,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.biases_initializer)
                    if tensor_out.size(-1) == 5:
                        m = 2
                    elif tensor_out.size(-1) == 3:
                        m = 1
                    elif tensor_out.size(-1) == 1:
                        m = 0
                    output_tensor_list[m].append(tensor_out)

                if key == 0 or key == 1:
                    # L x 1 -> 1
                    tensor_out = filter_1_output_1(tensor, rbf, unit_vectors,
                                                   output_dim=output_dim,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.biases_initializer)
                    m = 0 if tensor_out.size(-1) == 1 else 1
                    output_tensor_list[m].append(tensor_out)

        return output_tensor_list


class SelfInteraction(nn.Module):
    """
    重みとバイアスの初期化を行い、
    forwardメソッドでは、各テンソルに対して自己相互作用レイヤーを適用し、
    結果をoutput_tensor_listに追加します。
    ヘルパー関数self_interaction_layer_with_biasesとself_interaction_layer_without_biasesは、
    それぞれバイアスありとバイアスなしの自己相互作用レイヤーを実装しています。
    """
    def __init__(self, output_dim, weights_initializer=None, biases_initializer=None):
        super(SelfInteraction, self).__init__()
        self.output_dim = output_dim
        self.weights_initializer = weights_initializer or nn.init.orthogonal_
        self.biases_initializer = biases_initializer or nn.init.constant_

    # def self_interaction_layer_with_biases(self, tensor, output_dim):
    #     input_dim = tensor.size(-2)
    #     weights = nn.Parameter(torch.Tensor(output_dim, input_dim))
    #     biases = nn.Parameter(torch.Tensor(output_dim))
    #
    #     self.weights_initializer(weights)
    #     self.biases_initializer(biases, 0.0)
    #
    #     output = torch.einsum('afi,gf->aig', tensor, weights) + biases.unsqueeze(-1)
    #     return output.permute(0, 2, 1)

    # def self_interaction_layer_without_biases(self, tensor, output_dim):
    #     input_dim = tensor.size(-2)
    #     weights = nn.Parameter(torch.Tensor(output_dim, input_dim))
    #
    #     self.weights_initializer(weights)
    #
    #     output = torch.einsum('afi,gf->aig', tensor, weights)
    #     return output.permute(0, 2, 1)

    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: []}

        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                if key == 0:
                    tensor_out = SelfInteractionLayerWithBiases(tensor.size(-2), self.output_dim)(tensor)
                    # tensor_out = self.self_interaction_layer_with_biases(tensor, self.output_dim)
                else:
                    tensor_out = SelfInteractionLayerWithoutBiases(tensor.size(-2), self.output_dim)(tensor)
                    # tensor_out = self.self_interaction_layer_without_biases(tensor, self.output_dim)

                m = 0 if tensor_out.size(-1) == 1 else 1
                output_tensor_list[m].append(tensor_out)

        return output_tensor_list


class Nonlinearity(nn.Module):
    """
    非線形性の適用を行い、
    forwardメソッドでは、各テンソルに対してローテーション等価非線形性を適用し、
    結果をoutput_tensor_listに追加します。
    ヘルパー関数rotation_equivariant_nonlinearityは、テンソルに非線形関数を適用します。
    """
    def __init__(self, nonlin=F.elu, biases_initializer=None):
        super(Nonlinearity, self).__init__()
        self.nonlin = nonlin
        self.biases_initializer = biases_initializer or nn.init.constant_

        self.biases0 = nn.Parameter(torch.Tensor(1))
        self.biases_initializer(self.biases0, 0.0)
        self.biases1 = nn.Parameter(torch.Tensor(3))
        self.biases_initializer(self.biases1, 0.0)
        self.biases2 = nn.Parameter(torch.Tensor(5))
        self.biases_initializer(self.biases2, 0.0)

    def rotation_equivariant_nonlinearity(self, tensor, nonlin):
        output_dim = tensor.size(-1)
        if output_dim == 1:
            biases = self.biases0
        elif output_dim == 3:
            biases = self.biases1
        elif output_dim == 5:
            biases = self.biases2
        return nonlin(tensor + biases.unsqueeze(-2))

    def forward(self, input_tensor_list):
        output_tensor_list = {0: [], 1: [], 2: [], 3: []}

        for key in input_tensor_list:
            for i, tensor in enumerate(input_tensor_list[key]):
                if key == 0 or key == 1 or key == 2:
                    tensor_out = self.rotation_equivariant_nonlinearity(tensor, nonlin=self.nonlin)
                    tensor_out = tensor_out.unsqueeze(1)

                if tensor_out.size(-1) == 1:
                    m = 0
                elif tensor_out.size(-1) == 3:
                    m = 1
                elif tensor_out.size(-1) == 5:
                    m = 2
                output_tensor_list[m].append(tensor_out)

        return output_tensor_list


def concatenation(input_tensor_list):
    """
    各テンソルリストを指定されたチャネル軸に沿って連結し、結果をoutput_tensor_listに追加します。
    """
    output_tensor_list = {0: [], 1: [], 2: []}

    for key in input_tensor_list:
        # Concatenate along channel axis
        # [N, channels, M]
        concatenated_tensor = torch.cat(input_tensor_list[key], dim=-2)
        output_tensor_list[key].append(concatenated_tensor)

    return output_tensor_list


class TFN(nn.Module):
    def __init__(self, num_atom_types, output_dim):
        super(TFN, self).__init__()
        self.num_atom_types = num_atom_types
        self.output_dim = output_dim
        self.layer_dims = [self.output_dim, self.output_dim, self.output_dim, self.output_dim]
        self.embedding = SelfInteractionLayerWithBiases(self.num_atom_types, self.layer_dims[0])
        # self.embedding = nn.Linear(self.num_atom_types, self.layer_dims[0])

        self.rbf_low = 0.0
        self.rbf_high = 2.5
        self.rbf_count = 4

        self.conv_layer = Convolution(rbf_count=self.rbf_count)
        self.self_interaction_layers = []
        for idx in range(len(self.layer_dims)):
            self.self_interaction_layers.append(SelfInteraction(self.layer_dims[idx]))
        self.nonlinearity_layer = Nonlinearity()

    def forward(self, inputs_coords, inputs_one_hot, crystal_atom_idx, carbon_idx):
        rbf_spacing = (self.rbf_high - self.rbf_low) / self.rbf_count
        centers = torch.linspace(self.rbf_low, self.rbf_high, self.rbf_count)

        rij = difference_matrix(inputs_coords)
        unit_vectors = rij / (torch.norm(rij, dim=-1, keepdim=True) + EPSILON)
        dij = distance_matrix(inputs_coords)
        gamma = 1.0 / rbf_spacing
        rbf = torch.exp(-gamma * (dij.unsqueeze(-1) - centers) ** 2)

        # EMBEDDING
        # embed_layer = SelfInteractionLayerWithBiases(self.num_atom_types, self.layer_dims[0])
        # embed = embed_layer(inputs_one_hot.view(-1, self.num_atom_types, 1))
        embed = self.embedding(inputs_one_hot.view(-1, self.num_atom_types, 1))
        embed = embed.unsqueeze(1)
        input_tensor_list = {0: [embed]}

        # LAYERS 1-3
        for layer in range(1, len(self.layer_dims)):
            input_tensor_list = self.conv_layer(input_tensor_list, rbf, unit_vectors)
            input_tensor_list = concatenation(input_tensor_list)
            # input_tensor_list = SelfInteraction(layer_dim)(input_tensor_list)
            input_tensor_list = self.self_interaction_layers[layer](input_tensor_list)
            if layer < len(self.layer_dims) - 1:
                input_tensor_list = self.nonlinearity_layer(input_tensor_list)

        predicted_spectra = self.pooling(input_tensor_list[1][0].permute(0, 2, 1), crystal_atom_idx, carbon_idx)

        return predicted_spectra

    def pooling(self, input_tensor, crystal_atom_idx, carbon_idx):
        summed_fea = []
        for i, idx_map in enumerate(crystal_atom_idx):
            crystal_fea_ = input_tensor[idx_map]
            carbon_idx_ = carbon_idx[i]
            carbon_fea = []
            for j in range(len(carbon_idx_)):
                if carbon_idx_[j] == 1:
                    carbon_fea.append(crystal_fea_[j])
            carbon_fea = torch.stack(carbon_fea, dim=0)
            summed_fea.append(torch.mean(carbon_fea, dim=0, keepdim=True))

        return torch.cat(summed_fea, dim=0)

