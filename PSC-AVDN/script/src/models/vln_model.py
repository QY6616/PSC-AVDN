from math import gamma
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AutoModel, BertTokenizerFast


class SoftDotAttention(nn.Module):
    def __init__(self, dim):
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        target = self.linear_in(h).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            attn.data.masked_fill_(mask, -float("inf"))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        lang_embeds = torch.cat((weighted_context, h), 1)
        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class MulAttention(nn.Module):
    def __init__(self, dim):
        super(MulAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim)
        self.tanh = nn.Tanh()

    def forward(self, h, context):
        target = self.linear_in(h)
        attn = torch.mul(context, target)
        attn = self.sm(attn)
        weighted_context = torch.mul(attn, context)
        lang_embeds = torch.cat((weighted_context, h), 1)
        lang_embeds = self.tanh(self.linear_out(lang_embeds))
        return lang_embeds, attn


class pre_direction(nn.Module):
    def __init__(self):
        super(pre_direction, self).__init__()
        self.embedding = nn.Linear(2, 32)
        self.direction_prediction = nn.Linear(512, 2)
        self.linears = nn.Sequential(
            nn.Linear(768 + 32, 512), nn.ReLU(), nn.Dropout(0.1),
        )

    def forward(self, h, current_direct):
        direct_embeds = torch.concat(
            (
                torch.sin(current_direct / 180 * 3.14159),
                torch.cos(current_direct / 180 * 3.14159),
            ),
            axis=1,
        )
        h = torch.cat((self.embedding(direct_embeds), h), 1)
        h = self.linears(h)
        return self.direction_prediction(h)


class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.linears = nn.Sequential(
            nn.Linear(768, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 49), nn.ReLU()
        )

    def forward(self, ids, mask):
        bert_output = self.bert(ids, attention_mask=mask)
        cls_hidden = bert_output["pooler_output"]
        sequence_output = bert_output["last_hidden_state"]
        linear_output = self.linears(cls_hidden)
        return sequence_output, linear_output, cls_hidden


class ViT_LSTM(nn.Module):
    def __init__(
        self,
        args,
        vit_model,
        hidden_size=768,
        dropout_ratio=0.5,
        im_channel_size=512,
        im_feature_size=49,
        embedding_size=32,
    ):
        super().__init__()
        print("\nInitalizing the CLIP_LSTM model ...")
        self.args = args
        self.direction_embedding = nn.Linear(2, embedding_size)
        self.pos_embedding = nn.Linear(2, embedding_size)
        self.vision_model = vit_model
        self.attention_layer_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision = SoftDotAttention(im_feature_size)
        self.vision_lstm = nn.LSTMCell(im_feature_size, 576)
        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, 192)
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(im_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(
        self,
        current_direct,
        im_input,
        pos_input,
        cls_hidden,
        lang_feature,
        h_0=None,
        c_0=None,
        hh_0=None,
        cc_0=None,
    ):
        im_feature = self.vision_model(im_input)
        im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)
        input_lstm_0, beta = self.attention_layer_vision(cls_hidden, im_feature)
        drop = self.drop(input_lstm_0)
        if hh_0 is None or cc_0 is None:
            hh_1, cc_1 = self.vision_lstm(drop)
        else:
            hh_1, cc_1 = self.vision_lstm(drop, (hh_0, cc_0))
        direction = torch.concat(
            (
                torch.sin(current_direct / 180 * 3.14159),
                torch.cos(current_direct / 180 * 3.14159),
            ),
            axis=1,
        )
        direction_embeds = self.direction_embedding(direction)
        if h_0 is None or c_0 is None:
            h_1, c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1, c_1 = self.direct_lstm(direction_embeds, (h_0, c_0))
        action_module_input, alpha = self.attention_layer_lang(
            torch.cat((h_1, hh_1), 1), lang_feature
        )
        h_sali = self.fc(input_lstm_0).view(-1, 1, 8, 8)
        pred_saliency = nn.functional.interpolate(
            h_sali, size=(224, 224), mode="bilinear", align_corners=False
        )
        output = self.decoder_2_action_full(action_module_input)
        return h_1, c_1, hh_1, cc_1, output, pred_saliency


class ViT_LSTM_vision_only(nn.Module):
    def __init__(
        self,
        args,
        vit_model,
        hidden_size=768,
        dropout_ratio=0.5,
        im_channel_size=512,
        im_feature_size=49,
        embedding_size=32,
    ):
        super().__init__()
        print("\nInitalizing the CLIP_LSTM model ...")
        self.args = args
        self.direction_embedding = nn.Linear(2, embedding_size)
        self.pos_embedding = nn.Linear(2, embedding_size)
        self.vision_model = vit_model
        self.attention_layer_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision_lang = SoftDotAttention(hidden_size)
        self.attention_layer_vision = SoftDotAttention(im_feature_size)
        self.vision_lstm = nn.LSTMCell(im_feature_size, 576)
        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, 192)
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(im_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.h_fc = nn.Sequential(nn.Linear(hidden_size, im_feature_size), nn.ReLU())

    def forward(
        self,
        current_direct,
        im_input,
        pos_input,
        h_0=None,
        c_0=None,
        hh_0=None,
        cc_0=None,
    ):
        im_feature = self.vision_model(im_input)
        im_feature = im_feature.view(im_feature.size(0), im_feature.size(1), -1)
        input_lstm_0, beta = self.attention_layer_vision(
            self.h_fc(torch.cat((h_0, hh_0), 1)), im_feature
        )
        drop = self.drop(input_lstm_0)
        if hh_0 is None or cc_0 is None:
            hh_1, cc_1 = self.vision_lstm(drop)
        else:
            hh_1, cc_1 = self.vision_lstm(drop, (hh_0, cc_0))
        direction = torch.concat(
            (
                torch.sin(current_direct / 180 * 3.14159),
                torch.cos(current_direct / 180 * 3.14159),
            ),
            axis=1,
        )
        direction_embeds = self.direction_embedding(direction)
        if h_0 is None or c_0 is None:
            h_1, c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1, c_1 = self.direct_lstm(direction_embeds, (h_0, c_0))
        action_module_input = torch.cat((h_1, hh_1), 1)
        h_sali = self.fc(input_lstm_0).view(-1, 1, 8, 8)
        pred_saliency = nn.functional.interpolate(
            h_sali, size=(224, 224), mode="bilinear", align_corners=False
        )
        output = self.decoder_2_action_full(action_module_input)
        return h_1, c_1, hh_1, cc_1, output, pred_saliency


class ViT_LSTM_lang_only(nn.Module):
    def __init__(
        self,
        args,
        vit_model,
        hidden_size=768,
        dropout_ratio=0.5,
        im_channel_size=512,
        im_feature_size=49,
        embedding_size=32,
    ):
        super().__init__()
        print("\nInitalizing the CLIP_LSTM model ...")
        self.args = args
        self.direction_embedding = nn.Linear(2, embedding_size)
        self.pos_embedding = nn.Linear(2, embedding_size)
        self.attention_layer_lang = SoftDotAttention(hidden_size)
        self.drop = nn.Dropout(p=0.2)
        self.direct_lstm = nn.LSTMCell(embedding_size, hidden_size)
        self.decoder_2_action_full = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(im_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, current_direct, pos_input, lang_feature, h_0=None, c_0=None):
        direction = torch.concat(
            (
                torch.sin(current_direct / 180 * 3.14159),
                torch.cos(current_direct / 180 * 3.14159),
            ),
            axis=1,
        )
        direction_embeds = self.direction_embedding(direction)
        if h_0 is None or c_0 is None:
            h_1, c_1 = self.direct_lstm(direction_embeds)
        else:
            h_1, c_1 = self.direct_lstm(direction_embeds, (h_0, c_0))
        concat_h = h_1
        lang_embeds, alpha = self.attention_layer_lang(concat_h, lang_feature)
        output = self.decoder_2_action_full(lang_embeds)
        return h_1, c_1, output
