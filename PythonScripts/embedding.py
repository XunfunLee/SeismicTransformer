"""
Contains functionality for embedding.
1. Patch embedding.
2. Projection embedding.

Class token and positional embedding is just one line of the code, don't need to create a class or function.

Author: Jason Jiang (Xunfun Lee)
Date: 2023.12.1
"""

import torch
import torch.nn as nn
import torchinfo

class PatchEmbedding(nn.Module):
    def __init__(self, num_patches:int, patch_size:int):
        """
        Args:
          num_patches: number of patches
          patch_size: size of patches
        """
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size

    def forward(self, x):
        """
        Args:
          x: input shape: [batch_size, sequence_length, 1]
        Returns:
          output shape: [batch_size, num_patches, patch_size]
        """
        # verify the input shape
        assert x.shape[1] == self.num_patches * self.patch_size, \
            f'Input sequence length should be {self.num_patches * self.patch_size}'

        # [batch_size, sequence_length, 1] --> [batch_size, num_patches, patch_size]
        # e.g. [64, 3000, 1] --> [64, 12, 250]
        x = x.view(-1, self.num_patches, self.patch_size)
        return x

    def summary(self,
                batch_size:int=32):
        """ Summary this class with torchinfo.summary()

        Args:
          batch size: default is 32
        
        """
        fake_input = torch.randn((batch_size, self.num_patches * self.patch_size, 1))
        summary = torchinfo.summary(self,
                        input_data=fake_input,
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])
        print(summary)
        
class ProjectionModule(nn.Module):
    def __init__(self, input_size, output_size):
        """ Increase the dimension of the patch size to hidden size

        Args:
          input_size: size of the patches
          hidden_size: embedding size of the model
        """
        super(ProjectionModule, self).__init__()
        self.up_project = nn.Linear(input_size, output_size)
        self.input_size = input_size

    def forward(self, x):
        return self.up_project(x)

    def summary(self,
                batch_size:int=32,
                num_of_patch:int=12):
        """ Summary this class with torchinfo.summary()

        Args:
          batch size: default is 32
          num_of_patch: default is 12
        
        """
        fake_input = torch.randn((batch_size, num_of_patch, self.input_size))
        summary = torchinfo.summary(self,
                                   input_data=fake_input,
                                   col_names=["input_size", "output_size", "num_params", "trainable"],
                                   col_width=20,
                                   row_settings=["var_names"])
        print(summary)
