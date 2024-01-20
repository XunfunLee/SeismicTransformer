"""
Contains functionality for transformer architecture.
1. Multi-head attention(MSA) block.
2. Multi-Layer perceptron(MLP) block.
3. Transformer encoder(combin MAS and MLP).
4. Seismic Transformer V1.0.
5. Seismic Transformer V2.0.
5. Seismic Transformer V3.0 with all block rewrited including MHA, MLP, PE, FE, Encoder, Decoder, Classifier, Splicer, Seismic Transformer V3.0.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.1
"""

import torch
import torch.nn as nn
import torchinfo
from .embedding import ProjectionModule, PatchEmbedding, ConvLinearModule
import torch.nn.functional as F
import numpy as np
import random

class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    This class can only be used in the TransformerEncoderBlock class, because decoder need to use encoder-decoder attention.

    Q: Why using self created MSA block instead of nn.MultiheadAttention?
    A: In fact we are using nn.MultiheadAttention, but we need to do some pre-processing to the mask, as well as the position of the layernorm.
    In traditional nn.MultiheadAttention, the layernorm is after the MSA block, but in the research(also in ViT), the layernorm is before the MSA block.
    Pre-NormLayer will make the model more stable and easier to train.
    You can also create nn.MultiheadAttention and add LN inside EncoderBlock.
    There is thounsands of ways to create a model, but the most important thing is to understand the concept of the model.
    The resaon here to create custom class from nn.Module is to make the code more flexible for update and modification.
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

        # pre-LN
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
        # pre-LN
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
        
        '''
        frequency.shape = [batch_size, 1500]
        '''
        # convolution and linear projection (ViT don't have this)                                              [64, 1500] --> [64, 768]
        frequency = self.conv_linear(frequency)
        '''
        frequency.shape = [batch_size, 768]
        '''
        
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


##### ---------------------------------- SeiscmicTransformer V3.0 ---------------------------------- #####
"""
Author: Jason Jiang (Xunfun Lee)

date: 2023.01.14

from v3.0 we are using above class to create the model with more flexible and high modularity.
After testing inside cookbook, newly class doesn't have summary part for the reason is that `torchinfo.summary()` sometimes can't get the summary though the model is correct!
All code directly comes from `SeT_3_CookBook.ipynb`, used in `SeT_3_TrainBook.ipynb` for the first purpose.
Some of the classes have test code at the end of the definition, use if you need.

1. MLP Block (multi-perceptron layer)
2. MHA Block (multi-head attention): three types of MHA can based on this class
3. PE Block (Patch Embedding Block)
3.1 SPE Block (Small Patch Embedding Block)
4. FE Block (Frequency Embedding Block)
5. Encoder Block (also called encoder layer)
6. Encoder (dozens of Encoder Blocks)
7. Decoder Block (also called decoder layer)
8. Decoder (dozens of Decoder Blocks)
9. Classifier (fully connected layer) - for classification task
10. Splicer (convert the patch embedding to the original shape) - (batch_size, 12, 768) --> (batch_size, 3000, 1)
11. Seismic Transformer V3.0 (combine all above blocks and layers)

"""

# 1. MLP Block
class MLPBlock(nn.Module):
    ''' Multi-Layer perceptron block class, including feed forward and dropout layers.

    (64, 14, 768) --> (64, 14, 768)
    
    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        fc_hidden_size (int): Hidden size of the first fully connected layer. Defaults to 3072.
        dropout_rate (float): Dropout rate. Defaults to 0.1.
    '''

    def __init__(self, 
                 hidden_size: int = 768, 
                 fc_hidden_size: int = 3072, 
                 dropout_rate: float = 0.1):

        super(MLPBlock, self).__init__()
        
        # Pre-Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, fc_hidden_size)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(fc_hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Save the residual
        residual = x
        # Apply pre-layer normalization
        x = self.layer_norm(x)
        # First fully connected layer
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        # Second fully connected layer
        x = self.linear2(x)
        x = self.dropout2(x)
        # Add the residual
        x = x + residual
        # Implementing the residual connection
        return x

# 2. MHA Block
class MHABlock(nn.Module):
    """Multi-head attention block class, can be used in the Encoder, Decoder, and Cross-Attention parts of the Transformer model.

    (64, 14, 768) --> (64, 14, 768)

    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, 
                 hidden_size: int = 768, 
                 num_heads: int = 12, 
                 dropout_attn: float = 0.1,
                 batch_first: bool = True):
        
        super(MHABlock, self).__init__()
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, 
                                                    num_heads=num_heads, 
                                                    dropout=dropout_attn,
                                                    batch_first=batch_first)
        self.dropout = nn.Dropout(dropout_attn)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, need_weights=True):
        # Save the residual
        residual = query
        # Apply pre-layer normalization
        normed_query = self.layer_norm(query)
        normed_key = self.layer_norm(key)
        normed_value = self.layer_norm(value)

        # Multi-head attention
        attn_output, attn_output_weights = self.multihead_attn(normed_query,
                                                               normed_key,
                                                               normed_value,
                                                               key_padding_mask=key_padding_mask,
                                                               attn_mask=attn_mask,
                                                               need_weights=need_weights)
        # Apply dropout
        attn_output = self.dropout(attn_output)
        # Add the residual
        attn_output = attn_output + residual
        
        return attn_output, attn_output_weights
    
    # test code
    '''python
    # Instance for three types of MHA
    encoder_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)
    decoder_masked_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)
    cross_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)
    
    # MHA block inside encoder:
    encoder_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)
    input_tensor = torch.rand(batch_size, 14, 768) # (e.g.)
    padding_mask = torch.zeros(batch_size, 14, dtype=torch.bool) # (e.g.)

    # without padding mask:
    output, _ = encoder_mha_instance(input_tensor, input_tensor, input_tensor)
    # with padding mask:
    output, _ = encoder_mha_instance(input_tensor, input_tensor, input_tensor, key_padding_mask=padding_mask)
    
    # MHA block inside decoder:
        # decoder MHA:
    decoder_masked_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)
        # encoder-decoder MHA (cross-MHA):
    cross_mha = MHABlock(hidden_size=768, num_heads=12, dropout_attn=0.1, batch_first=True)

    decoder_input = torch.rand(batch_size, 12, 768)
    seq_length = decoder_input.shape[1]
    attn_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool() # (12, 12)

    # decoder MHA:
    output_decoder = test_decoder_masked_mha(decoder_masked_mha, decoder_input, attn_mask)
    # encoder-decoder MHA (cross-MHA):
    output_cross = test_cross_mha(cross_mha, output_decoder, encoder_output)
    # PS: encoder_output = (batch_size, 12, 768)
    
    '''
    
# 3. PE Block(patch embedding)
class PatchEmbeddingBlock(nn.Module):
    """Patch Embedding Block class, turning 3000 points into 12 patches with 250 points each.

    (64, 3000, 1) --> (64, 12, 768)
    
    Args:
        len_gm (int): Length of the ground motion. Defaults to 3000.
        patch_size (int): Size of the patch. Defaults to 250.
        output_size (int): Output size of the linear layer. Defaults to 768.
    """

    def __init__(self, 
                 len_gm:int=3000, 
                 patch_size:int=250, 
                 output_size:int=768):
        super(PatchEmbeddingBlock, self).__init__()

        # Assume that slice_size is a two-dimensional tuple (seq_len, features)
        self.patch_size = patch_size
        self.output_size = output_size
        self.linear = nn.Linear(patch_size, output_size)
        self.num_of_patches = len_gm // patch_size

    def forward(self, x):
        # verify the input shape
        assert x.shape[1] == self.num_of_patches * self.patch_size, \
            f'Input sequence length should be {self.num_of_patches * self.patch_size}'
        
        # [batch_size, sequence_length, 1] --> [batch_size, num_patches, patch_size]
        # e.g. [64, 3000, 1] --> [64, 12, 250]
        x = x.view(-1, self.num_of_patches, self.patch_size)

        # Reshape for the linear layer
        x = self.linear(x)
        # Reshape to the desired output size (batch_size, seq_len, output_size)
        return x
    
    # test code
    '''python
    PEBlock_Instance = PatchEmbeddingBlock().to(device)
    input_PE = torch.rand(batch_size, 3000, 1).to(device)
    output = PEBlock_Instance(input_PE)

    # output.shape = torch.Size([64, 12, 768])
    
    '''

# 3.1 SPE Block(small patch embedding)
class SmallPatchEmbeddingBlock(nn.Module):
    """Small patch embedding is used inside decoder, to fit the decoder features.
    Inside decoder:
    1. In training mode, we have to decide if the sequence is from teacher forcing or self generated.
    2. In reference mode, all the sequence is self-generated.

    Different from PE block which turn (batch_size, 3000, 1) --> (batch_size, 12, 768),
    SPE block turn (batch_size, 1, 250) --> 12*(batch_size, 1, 768) and cat to (batch_size, 12, 768).
    In other word, instead of turning 3000 points into 12 patches all at once in PE,
    SPE first slice 3000 points into 12 parts and patch them into (1, 768), then cat to (12, 768) for decoder usage.

    (batch_size, 1, 250) --> 12*(batch_size, 1, 768) --> (batch_size, 12, 768)
    
    """
    def __init__(self,
                 len_patch:int=250,
                 output_size:int=768):
        
        super().__init__()

        self.len_patch = len_patch                    # length of each patch, default is 250 = 3000/12
        self.output_size = output_size          # hidden size
        self.linear = nn.Linear(self.len_patch, output_size)

    def forward(self, x):

        assert x.shape[2] == self.len_patch, \
            f'Input sequence length should be {self.len_patch}'
        
        x = self.linear(x)

        return x
    
    # test code
    '''python
    SPE_instance = SmallPatchEmbeddingBlock().to(device)
    input_tensor = torch.rand((1, 1, 250)).to(device)
    output_SPE = SPE_instance(input_tensor)

    # output_SPE.shape = torch.Size([1, 1, 768])
    '''

# 4. FE Block(frequency embedding)
class FreqEmbeddingBlock(nn.Module):
    """Frequency Embedding Block class, turning 3000 points into 768 points.

    (64, 3000, 1) --> (64, 1, 768)

    1. take 3000 input into FFT, then take the real part of the FFT result
    2. take the first 1500 points, then do a convolution with kernel size 2 and stride 2
    3. then do a linear transformation to get the final output.

    Args:
        conv_output_size (int): Output size of the convolution layer. Defaults to 750.
        linear_output_size (int): Output size of the linear layer. Defaults to 768.
    """

    def __init__(self, 
               conv_output_size:int=750, 
               linear_output_size:int=768):
    
        super(FreqEmbeddingBlock, self).__init__()

        self.fft = np.fft.rfft
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.linear = nn.Linear(conv_output_size, linear_output_size)

    def forward(self, x):
        # FFT
        # x = torch.fft.rfft(x, dim=-1).real
        x = torch.fft.rfft(x).real

        # # 确保我们有正确的维度来进行后续操作
        # if x.ndim == 2:
        #     x = x.unsqueeze(1)  # 转换为(batch_size, 1, seq_len//2+1)

        # # Keep only the first 1500 elements
        # x = x[:, :, :1500]  # 现在索引切片应该没问题

        # Keep only the first 1500 elements (as FFT will return N/2+1 elements for real input)
        x = x[:, :1500, :]
        # Convolution
        x = x.permute(0, 2, 1)

        x = self.conv1d(x)
        # Linear transformation
        x = self.linear(x)

        return x
    
    # test code
    '''python
    FEBlock_Instance = FreqEmbeddingBlock().to(device)
    input_FE = torch.rand(64, 3000, 1).to(device)
    output = FEBlock_Instance(input_FE)

    # output.shape = torch.Size([64, 1, 768])    
    '''
    
# 5. Encoder block
class EncoderBlock(nn.Module):
    """Encoder block combined MLP, MHA block

    (batch_size, 14, 768) --> (batch_size, 14, 768)
    
    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        fc_hidden_size (int): Hidden size of the first fully connected layer. Defaults to 3072.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
        dropout_mlp (float): Dropout rate. Defaults to 0.1.
    
    """

    def __init__(self, 
                 hidden_size: int = 768, 
                 num_heads: int = 12, 
                 fc_hidden_size: int = 3072, 
                 dropout_attn: float = 0.1,
                 dropout_mlp: float = 0.1,
                 batch_first: bool = True):
        
        super(EncoderBlock, self).__init__()
        
        self.mha_block = MHABlock(hidden_size=hidden_size, 
                                  num_heads=num_heads, 
                                  dropout_attn=dropout_attn,
                                  batch_first=batch_first)
        
        self.mlp_block = MLPBlock(hidden_size=hidden_size, 
                                  fc_hidden_size=fc_hidden_size, 
                                  dropout_rate=dropout_mlp)
        
    def forward(self, x, key_padding_mask=None, need_weights=True):
        # Multi-head attention block
        x, attn_weights = self.mha_block(x, x, x, key_padding_mask=key_padding_mask, need_weights=need_weights)
        # MLP block
        x = self.mlp_block(x)

        return x, attn_weights
    
    # test code
    '''python
    EncoderBlock_Instance = EncoderBlock().to(device)
    input_EB = torch.rand(64, 14, 768).to(device)
    output, attn_weights = EncoderBlock_Instance(input_EB)

    # output.shape = torch.Size([64, 14, 768])
    # attn_weights.shape = torch.Size([64, 14, 14])
    '''

# 6. Encoder
class EncoderV1(nn.Module):
    """Encoder combined encoder block (MLP + MHA), PE and FE block

    (batch_size, 3000, 1) --> (batch_size, 14, 768)
    
    Args:
        len_gm (int): Length of the ground motion. Defaults to 3000.
        patch_size (int): Size of the patch. Defaults to 250.
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        fc_hidden_size (int): Hidden size of the first fully connected layer. Defaults to 3072.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
        dropout_mlp (float): Dropout rate. Defaults to 0.1.
        dropout_embed (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,
                 len_gm:int=3000,
                 patch_size:int=250,
                 hidden_size:int=768,
                 num_heads:int=12,
                 num_layers:int=12,
                 dropout_attn:float=0.1,
                 dropout_mlp:float=0.1,
                 dropout_embed:float=0.1):

        super().__init__()

        # Calculate the number of patches
        self.num_of_patch = len_gm // patch_size

        # Initialize a variable to stroe the attention weights
        self.attention_weights_list = []  # Initialize it here
        
        # BLOCK
        # patch embedding
        self.PatchEmbedding = PatchEmbeddingBlock(len_gm=len_gm,
                                      patch_size=patch_size,
                                      output_size=hidden_size)
        
        # frequency embedding
        self.FreqEmbedding = FreqEmbeddingBlock(conv_output_size=len_gm // 2 // 2,         # default is 750
                                     linear_output_size=hidden_size)

        # encoder layer
        self.EncoderLayers = nn.Sequential(*[EncoderBlock(hidden_size=hidden_size,
                                                          num_heads=num_heads,
                                                          fc_hidden_size=hidden_size*4,
                                                          dropout_attn=dropout_attn,
                                                          dropout_mlp=dropout_mlp) for _ in range(num_layers)])

        # [TOKEN]
        # [TIME] - time token
        self.time_token = nn.Parameter(torch.randn(1, 1, hidden_size),
                                       requires_grad=True)  # trainable parameter

        # [FREQ] - frequency token
        self.freq_token = nn.Parameter(torch.randn(1, 1, hidden_size),
                                       requires_grad=True)  # trainable parameter

        # [CLS] - class token
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_size),
                                        requires_grad=True)  # trainable parameter

        # POSITION
        # positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_of_patch+2, hidden_size),
                                                  requires_grad=True)  # trainable parameter
        
        # Dropout
        self.embedding_dropout = nn.Dropout(dropout_embed)


    def forward(self, x, key_padding_mask=None, need_weights=True):

        # Get the batch size
        batch_size = x.shape[0]

        # clear the attention weights list
        self.attention_weights_list = []

        # patch embedding
        time_sequence = self.PatchEmbedding(x)

        # [TIME] token
        time_tokens = self.time_token.repeat(batch_size, self.num_of_patch, 1)

        # concatenate the time sequence with the time tokens
        time_sequence_with_token = time_sequence + time_tokens                                      # [batch_size, 12, hidden_size]

        # frequency embedding
        freq_sequence = self.FreqEmbedding(x)

        # [FREQ] token
        freq_tokens = self.freq_token.repeat(batch_size, 1, 1)

        # concatenate the frequency sequence with the frequency tokens
        freq_sequence_with_token = freq_sequence + freq_tokens                                      # [batch_size, 1, hidden_size]

        # cat the time sequence and the frequency sequence
        sequence_combine = torch.cat((time_sequence_with_token, freq_sequence_with_token), dim=1)   # [batch_size, 13, hidden_size]

        # [CLS] token
        class_tokens = self.class_token.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # concatenate the class token with the sequence
        sequence_combine_with_cls = torch.cat((class_tokens, sequence_combine), dim=1)              # [batch_size, 14, hidden_size]

        # embedding dropout
        x = self.embedding_dropout(sequence_combine_with_cls)

        # Encoder Layer
        for layer in self.EncoderLayers:
            x, attn_weights = layer(x, key_padding_mask=key_padding_mask, need_weights=need_weights)
            self.attention_weights_list.append(attn_weights)

        return x
    
    # test code
    '''python
    EncoderV1_Instance = EncoderV1().to(device)
    input_Encoder = torch.rand(64, 3000, 1).to(device)
    output = EncoderV1_Instance(input_Encoder)

    # output.shape = torch.Size([64, 14, 768])
    '''
    
# 7. Decoder Block
class DecoderBlock(nn.Module):
    """Decoder Block: includes two MHA blocks and one MLP block.
    
    (batch_size, 12, 768) --> (batch_size, 12, 768)

    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        fc_hidden_size (int): Hidden size of the first fully connected layer. Defaults to 3072.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
        dropout_mlp (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, 
                 hidden_size: int = 768, 
                 num_heads: int = 12, 
                 fc_hidden_size: int = 3072, 
                 dropout_attn: float = 0.1,
                 dropout_mlp: float = 0.1,
                 batch_first: bool = True):
        
        super(DecoderBlock, self).__init__()
        
        # Masked Multi-head attention block
        self.mmha_block = MHABlock(hidden_size=hidden_size, 
                                  num_heads=num_heads, 
                                  dropout_attn=dropout_attn,
                                  batch_first=batch_first)
    
        # Cross Multi-head attention block (encoder-decoder attention)
        self.cmha_block = MHABlock(hidden_size=hidden_size, 
                                  num_heads=num_heads, 
                                  dropout_attn=dropout_attn,
                                  batch_first=batch_first)
        
        self.mlp_block = MLPBlock(hidden_size=hidden_size, 
                                  fc_hidden_size=fc_hidden_size, 
                                  dropout_rate=dropout_mlp)
        
    def forward(self, query, key, value, output_encoder, attn_mask=None, need_weights=True):

        # Masked Multi-head attention block
        mmha_output, mmha_attn_weights = self.mmha_block(query, key, value, need_weights=need_weights)

        # Cross Multi-head attention block
        cmha_output, cmha_attn_weights = self.cmha_block(mmha_output, output_encoder, output_encoder, attn_mask=attn_mask, need_weights=need_weights)

        # MLP block
        output = self.mlp_block(cmha_output)

        return output, mmha_attn_weights, cmha_attn_weights
    
    
# 8. Decoder
class DecoderV1(nn.Module):
    """Decoder combined decoder block (MLP + MHA), PE block

    (batch_size, 3000, 1) --> (batch_size, 12, 768) 

    Contains two main modes:
    * Training mode: use teacher forcing while decoder has input
    * Inference mode: use decoder output as input for next time step (initialize the sequence)
    
    Args:
        len_gm (int): Length of the ground motion. Defaults to 3000.
        patch_size (int): Size of the patch. Defaults to 250.
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        fc_hidden_size (int): Hidden size of the first fully connected layer. Defaults to 3072.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
        dropout_mlp (float): Dropout rate. Defaults to 0.1.
        dropout_embed (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self,
                 len_gm:int=3000,
                 patch_size:int=250,
                 hidden_size:int=768,
                 num_heads:int=12,
                 num_layers:int=12,
                 dropout_attn:float=0.1,
                 dropout_mlp:float=0.1,
                 dropout_embed:float=0.1,
                 device:torch.device="cuda"):

        super().__init__()

        # Calculate the number of patches
        self.num_of_patch = len_gm // patch_size

        # Initialize a variable to stroe the attention weights
        self.mmha_attn_weights_list = []  # Initialize it here
        self.cmha_attn_weights_list = []  # Initialize it here
        
        # BLOCK
        # patch embedding
        # self.PatchEmbedding = PatchEmbeddingBlock(len_gm=len_gm,
        #                               patch_size=patch_size,
        #                               output_size=hidden_size)

        self.SmallPatchEmbedding = SmallPatchEmbeddingBlock(len_patch=patch_size,
                                                            output_size=hidden_size)

        # encoder layer
        self.DecoderLayers = nn.Sequential(*[DecoderBlock(hidden_size=hidden_size,
                                                          num_heads=num_heads,
                                                          fc_hidden_size=hidden_size*4,
                                                          dropout_attn=dropout_attn,
                                                          dropout_mlp=dropout_mlp) for _ in range(num_layers)])

        # POSITION
        # positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_of_patch, hidden_size),
                                                  requires_grad=True)  # trainable parameter
        
        # Dropout
        self.embedding_dropout = nn.Dropout(dropout_embed)

        # Set device
        self.device = device

    def forward(self, output_encoder, decoder_input=None, attn_mask=None, need_weights=True):
        # Check if target_sequence is provided
        if decoder_input is not None:
            # Training mode
            # Use teacher forcing
            x = decoder_input

            # Patch embedding and position encoding can be done here if needed
            # patch embedding           
            time_sequence = self.PatchEmbedding(x)                      # (batch_size, 3000, 1) --> (batch_size, 12, 768)

            time_sequence = time_sequence + self.positional_embedding

            # embedding dropout
            x = self.embedding_dropout(time_sequence)

        else:
            # Inference mode
            # initialize a variable to store the generated sequence
            x = self._init_sequence(batch_size=output_encoder.shape[0])
            current_sequence = torch.zeros((output_encoder.shape[0], 0, 768)).to(self.device)  # 初始化空序列

            # print(f"Initial sequence shape: {x.shape}")
            # print(f"Current sequence shape: {current_sequence.shape}")
        
        # print(f"decoder x.shape: {x.shape}")                # (batch_size, 12, 768)

        # clear the attention weights list
        self.mmha_attn_weights_list = []
        self.cmha_attn_weights_list = []

        # Encoder Layer
        for layer in self.DecoderLayers:
            x, mmha_attn_weights, cmha_attn_weights = layer(query=x, 
                                                            key=x,
                                                            value=x,
                                                            output_encoder=output_encoder,
                                                            attn_mask=attn_mask, need_weights=need_weights)
            # Store attention weights if needed
            self.mmha_attn_weights_list.append(mmha_attn_weights)
            self.cmha_attn_weights_list.append(cmha_attn_weights)

            if decoder_input is None:
                current_sequence = self._update_sequence(x, current_sequence)
                # print(f"current_sequence.shape: {current_sequence.shape}")
                # print(f"final x.shape: {x.shape}")
        
        return x

    def _init_sequence(self, batch_size):
        # 初始化序列，这里我们使用零张量作为初始序列
        # 注意调整序列的维度以匹配模型的期望输入
        initial_sequence = torch.zeros((batch_size, 12, 768)).to(self.device)
        return initial_sequence

    def _update_sequence(self, decoder_output, current_sequence):
        # 更新序列，这里我们假设解码器输出的最后一个维度是要生成的序列
        # 这里的逻辑需要根据您的具体模型和任务进行调整
        new_sequence = decoder_output[:, -1:, :]  # 获取最后一个时间步的输出
        updated_sequence = torch.cat((current_sequence, new_sequence), dim=1)  # 沿时间维度附加
        return updated_sequence
    
    # # inference mode
    # def generate_sequence(self, output_encoder, attn_mask=None):
    #     # Initial sequence generation for inference mode
    #     print(output_encoder.shape[0])
    #     generated_sequence = self._init_sequence(batch_size=output_encoder.shape[0])
    #     for _ in range(self.num_of_patch):
    #         # Assume that you append to generated_sequence at each step
    #         # You may need to modify this loop to match your actual sequence generation process
    #         generated_sequence = self.forward(output_encoder, generated_sequence, attn_mask)
        
    #     return generated_sequence

    # def _init_sequence(self, batch_size):
    #     # Initialize the sequence for the decoder to start generating the output
    #     # This can be zeros, learned embeddings, or some form of encoder output processing
    #     initial_sequence = torch.zeros((batch_size, 3000, 1)).to(self.device)
    #     # Modify this to suit how you want to start sequence generation
    #     return initial_sequence


    # test code
    '''python
    def test_Decoder(Decoder_Instance, input_decoder, output_encoder, attn_mask=None):
        print("Testing Cross MHABlock with forward pass:")
        output = Decoder_Instance(output_encoder, input_decoder, attn_mask=attn_mask, need_weights=True)
        print("Output shape:", output.shape)
        return output

    DecoderV1_Instance = DecoderV1(decive=device).to(device)
    input_decoder = torch.rand(64, 3000, 1).to(device)
    output_encoder = torch.rand(64, 12, 768).to(device)
    attn_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(device)

    # training mode
    output = test_Decoder(DecoderV1_Instance, input_decoder, output_encoder, attn_mask=attn_mask)
    # output -  Output shape: torch.Size([64, 12, 768])

    # inference mode
    output = test_Decoder(DecoderV1_Instance, None, output_encoder, attn_mask=attn_mask)
    # output - Output shape: torch.Size([64, 12, 768])
    '''

# 9. Classifier
class ClassifierV1(nn.Module):
    """Classifier class, including layer normalization and linear layer.

    (batch_size, 14, 768) --> (batch_size, 14, 5)
    
    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_of_classes (int): Number of classes. Defaults to 5.
    """

    def __init__(self,
                 hidden_size:int=768,
                 num_of_classes:int=5) -> None:

        super().__init__()
        
        # LN
        self.LayerNorm = nn.LayerNorm(normalized_shape=hidden_size)
        # Linear
        self.Linear = nn.Linear(in_features=hidden_size, out_features=num_of_classes)
    
    def forward(self, x):
        # LN
        x = self.LayerNorm(x)
        # Linear                        [N, 768]
        logits = self.Linear(x)

        return logits   # (batch_size, 14, 5)
    
# 10. Splicer
class SplicerV1(nn.Module):
    """Convert the decoder output into the original shape.

    (batch_size, 12, 768) --> (batch_size, 3000, 1)
    
    Args:
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        patch_size (int): Size of the patch. Defaults to 250.
        len_gm (int): Length of the ground motion. Defaults to 3000.
    """

    def __init__(self, 
                 hidden_size:int=768,
                 patch_size:int=250,
                 len_gm:int=3000) -> None:

        super().__init__()

        # LN
        self.LayerNorm = nn.LayerNorm(normalized_shape=hidden_size)

        # linear1: [batch_size, 12, 768] --> [batch_size, 12, 250]
        self.Linear1 = nn.Linear(in_features=hidden_size, out_features=patch_size)

        # linear2: [batch_size, 3000, 1] --> [batch_size, 3000, 1]
        self.Linear2 = nn.Linear(in_features=len_gm, out_features=len_gm)

    def forward(self, x):
        # LN
        x = self.LayerNorm(x)
        # linear1 with GELU
        x = F.gelu(self.Linear1(x))
        # [N, 12, 250] --> [N, 3000]
        x = x.view(x.size(0), -1)  
        # linear2
        x = self.Linear2(x)
        # [N, 3000, 1]
        x = x.view(x.size(0), -1, 1)

        return x
    
# 11. Seismic Transformer V3.0
class SeismicTransformerV3(nn.Module):
    """Seismic Transformer V3.0 class, including encoder, decoder, classifier and splicer.
    
    Args:
        len_gm (int): Length of the ground motion. Defaults to 3000.
        patch_size (int): Size of the patch. Defaults to 250.
        hidden_size (int): Hidden size of the input tensor. Defaults to 768.
        num_heads (int): Number of attention heads. Defaults to 12.
        num_layers (int): Number of layers. Defaults to 12.
        dropout_attn (float): Dropout rate. Defaults to 0.1.
        dropout_mlp (float): Dropout rate. Defaults to 0.1.
        dropout_embed (float): Dropout rate. Defaults to 0.1.
        num_of_classes (int): Number of classes. Defaults to 5.
    """

    def __init__(self,
                 len_gm:int=3000,
                 patch_size:int=250,
                 hidden_size:int=768,
                 num_heads:int=12,
                 num_layers:int=12,
                 dropout_attn:float=0.1,
                 dropout_mlp:float=0.1,
                 dropout_embed:float=0.1,
                 num_of_classes:int=5):

        super().__init__()

        # Encoder
        self.encoder = EncoderV1(len_gm=len_gm,
                                 patch_size=patch_size,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_layers=num_layers,
                                 dropout_attn=dropout_attn,
                                 dropout_mlp=dropout_mlp,
                                 dropout_embed=dropout_embed)
        
        # Decoder
        self.decoder = DecoderV1(len_gm=len_gm,
                                 patch_size=patch_size,
                                 hidden_size=hidden_size,
                                 num_heads=num_heads,
                                 num_layers=num_layers,
                                 dropout_attn=dropout_attn,
                                 dropout_mlp=dropout_mlp,
                                 dropout_embed=dropout_embed)

        # Classifier
        self.classifier = ClassifierV1(hidden_size=hidden_size,
                                       num_of_classes=num_of_classes)
        
        # Splicer
        self.splicer = SplicerV1(hidden_size=hidden_size,
                                 patch_size=patch_size,
                                 len_gm=len_gm)
        
    def forward(self, encoder_input, decoder_input=None, key_padding_mask=None, attn_mask=None):
        # Encoder output
        encoder_output = self.encoder(encoder_input, key_padding_mask=key_padding_mask)

        encoder_output_to_decoder = encoder_output[:,1:13,:]

        # If target sequence is provided, we are in training mode, otherwise we are in inference mode
        if decoder_input is not None:
            # training mode
            # decoder_output = self.decoder(output_encoder=encoder_output_to_decoder, 
            #                               decoder_input=decoder_input, 
            #                               attn_mask=attn_mask,
            #                               need_weights=True)
            # # Splicer forward pass to generate the dynamic response
            # dynamic_response = self.splicer(decoder_output)
            
            # Training mode with Scheduled Sampling
            # ------------------------------------------------------------------------------
            outputs = []
            # input = decoder_input[:, 0:250, :].unsqueeze(1)  # Start with the first input
            input = decoder_input

            print(f"input.shape: {input.shape}")

            for t in range(1, 12):          # for number of patches is 12

                # input_embedded = self.decoder.PatchEmbedding(input)
                # input_embedded = input_embedded + self.decoder.positional_embedding
                # input_embedded = self.decoder.embedding_dropout(input_embedded)

                decoder_output = self.decoder(output_encoder=encoder_output_to_decoder, 
                                              decoder_input=input, 
                                              attn_mask=attn_mask,
                                              need_weights=True)
                outputs.append(decoder_output)
                
                # Decide whether to use teacher forcing
                if random.random() < 0.5:
                    # Use the next true token as the next input
                    next_patch = decoder_input[:, t*250:(t+1)*250, :].unsqueeze(1)

                    

                    input = next_patch          # (batch_size, 1, 250, 1) problem is here! because of decoder only take 3000 as input



                else:
                    # Use the model's own prediction as the next input
                    input = decoder_output

            decoder_output = torch.cat(outputs, dim=1)
            # ------------------------------------------------------------------------------

        else:
            # inference mode
            decoder_output = self.decoder(output_encoder=encoder_output_to_decoder,
                                          decoder_input=None,
                                          attn_mask=attn_mask,
                                          need_weights=True)
            
            dynamic_response = self.splicer(decoder_output)

        # Classifier forward pass to determine the damage state
        # damage_state is logits, put 0 index logit through classifier
        damage_state = self.classifier(encoder_output[:, 0])

        return damage_state, dynamic_response
    
    # test code
    '''python
    SeismicTransformerV3_instance = SeismicTransformerV3().to(device)
    input_gm = torch.rand(64, 3000, 1).to(device)
    input_floorResponse = torch.rand(64, 3000, 1).to(device)
    key_padding_mask = torch.zeros(64, 14, dtype=torch.bool).to(device)
    attn_mask = torch.triu(torch.ones(12, 12), diagonal=1).bool().to(device)

    # training mode (with encoder input)
    damage_state, dynamic_response = SeismicTransformerV3_instance(input_sequence=input_gm, 
                                                                   target_sequence=input_floorResponse, 
                                                                   key_padding_mask=key_padding_mask, 
                                                                   attn_mask=attn_mask)
    
    print(damage_state.shape, dynamic_response.shape)
    # output - (torch.Size([64, 14, 5]), torch.Size([64, 3000, 1]))

    # inference mode (without encoder input)
    with torch.inference_mode():
    damage_state, dynamic_response = SeismicTransformerV3_instance(input_sequence=input_gm, 
                                                                   target_sequence=None, 
                                                                   key_padding_mask=key_padding_mask, 
                                                                   attn_mask=attn_mask)
    print(damage_state.shape, dynamic_response.shape)
    # output - (torch.Size([64, 14, 5]), torch.Size([64, 3000, 1]))
    
    '''