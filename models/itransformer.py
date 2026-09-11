import torch
import torch.nn as nn



class RevIN(nn.Module):
    """
    Reversible Instance Normalization

    iTransformer 常用组件
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
    ):
        super().__init__()

        self.eps = eps

        self.num_features = num_features


        self.gamma = nn.Parameter(
            torch.ones(
                num_features
            )
        )

        self.beta = nn.Parameter(
            torch.zeros(
                num_features
            )
        )


    def forward(
            self,
            x,
            mode="norm"
    ):

        if mode == "norm":

            self.mean = (
                x.mean(
                    dim=1,
                    keepdim=True
                )
            )

            self.std = (
                x.std(
                    dim=1,
                    keepdim=True
                )
                + self.eps
            )


            x = (
                x - self.mean
            ) / self.std


            x = (
                x
                *
                self.gamma
                +
                self.beta
            )


        elif mode == "denorm":

            x = (
                x
                -
                self.beta
            )

            x = (
                x
                /
                self.gamma
            )

            x = (
                x
                *
                self.std
                +
                self.mean
            )


        return x



class iTransformerForecaster(
        nn.Module
):

    def __init__(
            self,
            input_size,
            lookback,
            pred_len,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
    ):

        super().__init__()


        self.input_size = input_size

        self.lookback = lookback

        self.pred_len = pred_len



        # Variable token embedding
        self.embedding = nn.Linear(
            lookback,
            d_model
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=nhead,

            dim_feedforward=
                dim_feedforward,

            dropout=dropout,

            batch_first=True,

            norm_first=True,

        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers,

        )


        self.revin = RevIN(
            input_size
        )


        self.projection = nn.Linear(

            d_model,

            pred_len

        )



    def forward(
            self,
            x
    ):

        """
        x:

        [B,L,N]

        B:
        batch

        L:
        lookback

        N:
        variables
        """

        # RevIN
        x = self.revin(
            x,
            "norm"
        )


        #
        # iTransformer:
        #
        # variable作为token
        #

        x = x.permute(
            0,
            2,
            1
        )


        # [B,N,L]


        x = self.embedding(
            x
        )


        # [B,N,d_model]


        x = self.encoder(
            x
        )


        out = self.projection(
            x
        )


        # [B,N,pred_len]

        out = out.permute(
            0,
            2,
            1
        )


        # [B,pred_len,N]


        out = self.revin(
            out,
            "denorm"
        )


        return out