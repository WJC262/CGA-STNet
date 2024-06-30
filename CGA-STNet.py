import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.GELU = nn.GELU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.reshape(B * T, C, H, W)  # Reshape to combine B and T dimensions
        x = x.float()

        x = self.GELU(self.conv1(x))
        x = self.GELU(self.conv2(x))
        x = self.pool(x)

        x = self.GELU(self.conv3(x))
        x = self.GELU(self.conv4(x))
        x = self.pool(x)

        x = x.reshape(B, T, -1)  # Reshape back to [B, T, -1]
        return x

class SelfAttentionWithGate(nn.Module):
    def __init__(self, d_model, time_data_shape):
        super(SelfAttentionWithGate, self).__init__()
        self.time_data_shape = time_data_shape
        self.d_model = d_model

        self.query = nn.Linear(d_model + time_data_shape, int(d_model/2))
        self.key = nn.Linear(d_model + time_data_shape, int(d_model/2))
        self.value = nn.Linear(d_model + time_data_shape, int(d_model/2))

        self.gate = nn.Linear(d_model + time_data_shape, 1)

    def forward(self, time_features, env_info):
        env_info = env_info.unsqueeze(1).expand(-1, time_features.size(1), -1)
        concat_input = torch.cat((time_features, env_info), dim=2)
        batch_size, seq_len, _ = concat_input.size()

        queries = self.query(concat_input)
        keys = self.key(concat_input)
        values = self.value(concat_input)

        gate_values = torch.sigmoid(self.gate(concat_input))

        scores = torch.matmul(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(float(self.d_model)))
        gated_scores = scores * gate_values
        attention_weights = torch.softmax(gated_scores, dim= -1)

        output = torch.matmul(attention_weights, values)
        return output


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),  # Inception Block V1
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)  # Inception Block V1
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

class Model(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels,
                 enc_in, embed, freq, dropout, e_layers, c_out, time_data_shape):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)])  # TimesBlock模块列表
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.model_cnn = CNNnet()
        self.time_data_shape = time_data_shape
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.SelfAttentionWithGate = SelfAttentionWithGate(enc_in, time_data_shape)

    def forecast(self, x_enc, x_mark_enc=None):
        time_list = x_enc[:, :self.time_data_shape]
        x_enc = x_enc[:, self.time_data_shape:]
        x_enc = x_enc.reshape(-1, self.seq_len, self.enc_in)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        B1,T1,_ = x_enc.size()
        x_enc_1 = self.SelfAttentionWithGate(x_enc, time_list)
        x_enc = x_enc.reshape(B1,T1,2,40,40)
        x_enc_2 = self.model_cnn(x_enc)
        x_enc = torch.cat((x_enc_1,x_enc_2),dim=2)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None):
        dec_out = self.forecast(x_enc, x_mark_enc)

        return dec_out[:, -self.pred_len:, :]

