# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from allennlp.modules.elmo import Elmo, batch_to_ids

import torch.nn as nn
import torch.nn.functional as F
import torch


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size, ix_to_token):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.USE_GLOVE = __C.USE_GLOVE
        self.USE_ELMO = __C.USE_ELMO
        # Loading the GloVe embedding weights
        if self.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        # load Elmo model
        if __C.ELMO_FEAT_SIZE == 1024:
            options_file = __C.ELMO_CONF_PATH + "elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = __C.ELMO_CONF_PATH + "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        elif __C.ELMO_FEAT_SIZE == 512:
            options_file = __C.ELMO_CONF_PATH + "elmo_2x2048_256_2048cnn_1xhighway_options.json"
            weight_file = __C.ELMO_CONF_PATH + "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

        self.qus_feat_linear = nn.Linear(
            __C.ELMO_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.ix_to_token = ix_to_token


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        if self.USE_GLOVE:
            glove_lang_feat = self.embedding(ques_ix)
            lang_feat, _ = self.lstm(glove_lang_feat)
        elif self.USE_ELMO:
            # elmo word embedding
            ques_content_iter = []
            for single_ques_ix in ques_ix:
                content = []
                for ix in single_ques_ix.numpy():
                    content.append(self.ix_to_token[ix])
                ques_content_iter.append(content)
            character_ids = batch_to_ids(ques_content_iter)
            elmo_embedding_list = self.elmo(character_ids)['elmo_representations'][0]
            elmo_embedding = torch.stack(elmo_embedding_list, 0)
            # lang_feat = torch.cat((glove_lang_feat, elmo_embedding[0]), dim=1)
            lang_feat = self.qus_feat_linear(elmo_embedding)            

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)
        print('初始图像特征：{}，初始语言特征：{}'.format(img_feat.size(), lang_feat.size()))

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
        print('图像特征：{}，语言特征：{}'.format(img_feat.size(), lang_feat.size()))
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
