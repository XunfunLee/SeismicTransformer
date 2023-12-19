"""
Contains functionality for transformer architecture.
1. Multi-head attention(MSA) block.
2. Multi-Layer perceptron(MLP) block.
3. Transformer encoder(combin MAS and MLP).
4. Seismic Transformer V1.0.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.1
"""

import torch
import torch.nn as nn
import torchinfo
from .embedding import ProjectionModule, PatchEmbedding, ConvLinearModule

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0.): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()
        
        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.num_heads = num_heads

        # Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?
        
    # Create a forward() method to pass the data throguh the layers
    def forward(self, x, mask=None):

        x = self.layer_norm(x)

        if mask is not None:
            # We expand on the sequence dimension to create [batch_size, seq_len, seq_len]
            # `mask` is expanded to match the shape required for `attn_mask` in MultiheadAttention
            # `unsqueeze(1)` adds a singleton dimension at index 1
            # `expand` repeats the tensor across the new dimension to match `seq_len`
            extended_mask = mask.unsqueeze(1).expand(-1, x.size(1), -1)
            # Since `attn_mask` expects a mask that has 0 for tokens to include and float('-inf') for tokens to exclude,
            # we convert our boolean mask to float tensor, where True values will be set to float('-inf').
            extended_mask = extended_mask.float().masked_fill(extended_mask, float('-inf'))
            # We need to replicate the mask for each head. This gives us the proper broadcast shape.
            extended_mask = extended_mask.repeat_interleave(self.num_heads, dim=0)
            # Turn the mask into a list, not torch(user warning: Prefer to use a boolean mask directly.)

            # the final mask shape is [batch_size * num_heads, seq_len, seq_len]   e.g. [23664, 13, 13]    23664 = 1972 * 12
        else:
            extended_mask = None

        attn_output, attn_weights = self.multihead_attn(query=x, # query embeddings 
                                                        key=x, # key embeddings
                                                        value=x, # value embeddings
                                                        attn_mask=extended_mask, # attention mask
                                                        need_weights=True) # do we need the weights or just the layer outputs? Yes, we need!
        return attn_output, attn_weights

    def summary(self,
                batch_size:int=32,
                number_of_patch:int=12,
                hidden_size:int=768):
        """ Summary this class with torchinfo.summary()

        Args:
          batch size: default is 32
          num_of_patch: default is 12
          hidden_size: default is 768
        
        """
        fake_input = (batch_size, number_of_patch, hidden_size)
        summary = torchinfo.summary(self,
                        input_size=fake_input,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])
        print(summary)

class MultiLayerPerceptronBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # Initialize the class with hyperparameters
    def __init__(self,
                 embedding_dim:int=768, # hidden size of the model
                 mlp_size:int=3072, # usually 4 times of the hidden size
                 dropout:float=0.1):
        super().__init__()
        
        #  Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        #  Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )
    
    # Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

    def summary(self,
                batch_size:int=32,
                number_of_patch:int=12,
                hidden_size:int=768):
        """ Summary this class with torchinfo.summary()

        Args:
            batch size: default is 32
            num_of_patch: default is 12
            hidden_size: default is 768

        """
        fake_input = (batch_size, number_of_patch, hidden_size)
        summary = torchinfo.summary(self,
                    input_size=fake_input,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"])
        print(summary)

class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    #  Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=768 * 4, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        # Create MLP block (equation 3)
        self.mlp_block =  MultiLayerPerceptronBlock(embedding_dim=embedding_dim,
                                                    mlp_size=mlp_size,
                                                    dropout=mlp_dropout)
        
    # Create a forward() method
    def forward(self, x, mask=None):
        # Output the attention_weight of the MAS block to see how the model is learning
        mas_output, attn_weights = self.msa_block(x, mask=mask)

        # Create residual connection for MSA block (add the input to the output)
        x = mas_output + x
        
        # Create residual connection for MLP block (add the input to the output)
        mlp_output = self.mlp_block(x) + x
        
        return mlp_output, attn_weights

    def summary(self,
                batch_size:int=32,
                number_of_patch:int=12,
                hidden_size:int=768):
        """ Summary this class with torchinfo.summary()

        Args:
            batch size: default is 32
            num_of_patch: default is 12
            hidden_size: default is 768

        """
        fake_input = (batch_size, number_of_patch, hidden_size)
        summary = torchinfo.summary(self,
                    input_size=fake_input,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"])
        print(summary)

class SeismicTransformerV1(nn.Module):
    """Creates a Seismic Transformer architecture with SeT-Base hyperparameters by default."""
    def __init__(self,
                 len_of_gm:int=3000, # GM_LENGTH: 3000 points (60s - 50Hz)
                 patch_size:int=250, # PATCH_SIZE: 250
                 num_transformer_layers:int=12, # number_of_patch: 3000/250
                 embedding_dim:int=768, # HIDDEN_SIZE
                 mlp_size:int=3072, # MLP size = HIDDEN_SIZE * 4
                 num_heads:int=12, # number of heads in multi-heads
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=5): # Damage state of the building
        
        super().__init__()
        
        # Make the ground motion length is divisble by the patch size 
        assert len_of_gm % patch_size == 0, f"length of ground motion must be divisible by patch size, ground motion size: {len_of_gm}, patch size: {patch_size}."
    
        # Initialize a variable to stroe the attention weights
        self.attention_weights_list = []  # Initialize it here

        # Calculate number of patches (length of GMs / patch size)
        self.num_patches = len_of_gm // patch_size

        # need to put 250 --> 768                                                  [batch_size, 12, 250]  --->  [batch_size, 12, 768]
        self.projection = ProjectionModule(patch_size, embedding_dim)
                 
        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)           [batch_size, 13, 768]
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Create learnable position embedding                                                               [batch_size, 13, 768]
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(num_patches=self.num_patches,
                                              patch_size=patch_size)
        
        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout) for _ in range(num_transformer_layers)]) # '_' means the i is not important
       
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    # Create a forward() method
    def forward(self, x, mask=None):

        # clear the attention weights list
        self.attention_weights_list = []

        # Get batch size
        batch_size = x.shape[0]
        
        # Create class token embedding and expand it to match the batch size (equation 1 in ViT)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # Create patch embedding (equation 1 in ViT)                                        [64, 3000, 1] --> [64, 12, 250]
        x = self.patch_embedding(x)

        # projection (ViT don't have this)                                                      [64, 12, 250] --> [64, 12, 768]
        x = self.projection(x)

        # Concat class embedding and patch embedding (equation 1 in ViT)
        x = torch.cat((class_token, x), dim=1)

        # Add position embedding to patch embedding (equation 1 in ViT) 
        x = self.position_embedding + x

        # Run embedding dropout (Appendix B.1 in ViT)
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3 in ViT)
        # Adding a function to collect the attn_weights of the last encoder layer
        for layer in self.transformer_encoder:
            x, attn_weights = layer(x, mask=mask)
            self.attention_weights_list.append(attn_weights)
        # suppose a list have: num_of_layer * [batch_size, num_of_patch+1, num_of_patch+1]
        # len(self.attention_weights_list) = num_of_layer
        # self.attention_weights_list[0].shape = torch.Size([batch_size, number_of_patch+1, num_of_patch+1])

        # Put 0 index logit through classifier (equation 4 in ViT)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x

    def summary(self,
                batch_size:int=32,
                len_of_gm:int=3000):
        """ Summary this class with torchinfo.summary()

        Args:
            batch size: default is 32
            length_of_each_gm: default is 3000

        """
        fake_input = (batch_size, len_of_gm, 1)
        summary = torchinfo.summary(self,
                                    input_size=fake_input,
                                    col_names=["input_size", "output_size", "num_params", "trainable"],
                                    col_width=20,
                                    row_settings=["var_names"])
        print(summary)

class SeismicTransformerV2(nn.Module):
    """Creates a Seismic Transformer architecture with SeT-Base hyperparameters by default.
    
    V2:
    1. input: 12 time-acceleration data + 1 frequency data          [13, 768]
    2. adding time token, frequency token into input
    
    
    """
    def __init__(self,
                 len_of_gm:int=3000, # GM_LENGTH: 3000 points (60s - 50Hz)
                 patch_size:int=250, # PATCH_SIZE: 250
                 num_transformer_layers:int=12, # number_of_patch: 3000/250
                 embedding_dim:int=768, # HIDDEN_SIZE
                 mlp_size:int=3072, # MLP size = HIDDEN_SIZE * 4
                 num_heads:int=12, # number of heads in multi-heads
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=5): # Damage state of the building
        
        super().__init__()
        
        # Make the ground motion length is divisble by the patch size 
        assert len_of_gm % patch_size == 0, f"length of ground motion must be divisible by patch size, ground motion size: {len_of_gm}, patch size: {patch_size}."
    
        # Initialize a variable to stroe the attention weights
        self.attention_weights_list = []  # Initialize it here

        # Calculate number of patches (length of GMs / patch size)
        self.num_patches = len_of_gm // patch_size

        # gm data need to put 250 --> 768                                                  [batch_size, 12, 250]  --->  [batch_size, 12, 768]
        self.projection = ProjectionModule(patch_size, embedding_dim)

        # frequency need to put 1500 --> 768                                                  [batch_size, 1, 1500]  --->  [batch_size, 1, 768]
        self.conv_linear = ConvLinearModule(conv_output_size=750,
                                           linear_output_size=embedding_dim)
                 
        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)           [batch_size, 13, 768]
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Time token
        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim),requires_grad=True)  # [1, 1, d_model]

        # Frequency token
        self.frequency_token = nn.Parameter(torch.randn(1, 1, embedding_dim),requires_grad=True)  # [1, 1, d_model]

        # Create learnable position embedding                                                               [batch_size, 14, 768]
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1+1, embedding_dim),
                                               requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        # Create patch embedding layer
        self.patch_embedding = PatchEmbedding(num_patches=self.num_patches,
                                              patch_size=patch_size)
        
        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout,
                                                                            attn_dropout=attn_dropout) for _ in range(num_transformer_layers)]) # '_' means the i is not important
       
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes)
        )
    
    # Create a forward() method
    def forward(self, x, mask=None, frequency=None):

        # clear the attention weights list
        self.attention_weights_list = []

        # Get batch size
        batch_size = x.shape[0]
        
        # Create class token embedding and expand it to match the batch size (equation 1 in ViT)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # time tokens
        time_tokens = self.time_token.repeat(batch_size, self.num_patches, 1)  # [batch_size, n_time_parts, d_model]
        
        # frequency tokens
        freq_tokens = self.frequency_token.repeat(batch_size, 1, 1)  # [batch_size, 1, d_model]

        # Create patch embedding (equation 1 in ViT)                                        [64, 3000, 1] --> [64, 12, 250]
        x = self.patch_embedding(x)

        # projection (ViT don't have this)                                                      [64, 12, 250] --> [64, 12, 768]
        x = self.projection(x)

        # add time token to time embedding patches                                              [64, 12, 768] --> [64, 12, 768]
        x = x + time_tokens
        
        # convolution and linear projection (ViT don't have this)                                              [64, 1, 1500] --> [64, 1, 768]
        frequency = self.conv_linear(frequency)
        
        # add frequency token to frequency embedding patches                                                   [64, 1, 768] --> [64, 1, 768]
        freq_data_with_token = freq_tokens + frequency.unsqueeze(1)

        # adding time data with token and frequency data with token                                            [64, 12, 768] --> [64, 13, 768]
        combined_data = torch.cat([x, freq_data_with_token], dim=1)

        # Concat class embedding and patch embedding (equation 1 in ViT)                                       [64, 13, 768] --> [64, 14, 768]
        x = torch.cat([class_token, combined_data], dim=1)

        # Add position embedding to patch embedding (equation 1 in ViT)                                        [64, 14, 768] --> [64, 14, 768]
        x = self.position_embedding + x

        # Run embedding dropout (Appendix B.1 in ViT)
        x = self.embedding_dropout(x)

        # Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3 in ViT)
        # Adding a function to collect the attn_weights of the last encoder layer
        for layer in self.transformer_encoder:
            x, attn_weights = layer(x, mask=mask)
            self.attention_weights_list.append(attn_weights)
        # suppose a list have: num_of_layer * [batch_size, num_of_patch+1, num_of_patch+1]
        # len(self.attention_weights_list) = num_of_layer
        # self.attention_weights_list[0].shape = torch.Size([batch_size, number_of_patch+1, num_of_patch+1])

        # Put 0 index logit through classifier (equation 4 in ViT)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x

    def summary(self,
                batch_size:int=32,
                len_of_gm:int=3000):
        """ Summary this class with torchinfo.summary()

        Args:
            batch size: default is 32
            length_of_each_gm: default is 3000

        """
        fake_input = (batch_size, len_of_gm, 1)
        summary = torchinfo.summary(self,
                                    input_size=fake_input,
                                    col_names=["input_size", "output_size", "num_params", "trainable"],
                                    col_width=20,
                                    row_settings=["var_names"])
        print(summary)