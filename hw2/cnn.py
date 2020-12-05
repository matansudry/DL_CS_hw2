import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: list,
        pool_every: int,
        hidden_dims: list,
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        if 'kernel_size' not in self.conv_params:
            self.conv_params=dict(kernel_size=3, stride=1, padding=1)
        if 'kernel_size' not in self.pooling_params:
            self.pooling_params=dict(kernel_size=2)

        # - extract number of conv layers
        N = len(self.channels)

        # - first conv layer 
        layers.append( nn.Conv2d (in_channels, self.channels[0], kernel_size= self.conv_params['kernel_size'], stride=self.conv_params['stride'], padding=self.conv_params['padding']))
        if self.activation_type == 'relu':
            layers.append(ACTIVATIONS[self.activation_type](self.activation_params))
        else:
            layers.append(ACTIVATIONS[self.activation_type](self.activation_params['negative_slope']))

        for i in range(1,N):
            if ((i % self.pool_every)==0):
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
            layers.append( nn.Conv2d (self.channels[i-1], self.channels[i], kernel_size= self.conv_params['kernel_size'], stride=self.conv_params['stride'], padding=self.conv_params['padding']))
            if self.activation_type == 'relu':
                layers.append(ACTIVATIONS[self.activation_type](self.activation_params))
            else:
                layers.append(ACTIVATIONS[self.activation_type](self.activation_params['negative_slope']))
        if ((N % self.pool_every)==0):
            layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        in_channels, in_h, in_w, = tuple(self.in_size)
        M = len(self.hidden_dims)
        N = len(self.channels)
        n_pooling = int(int(N - 1) / (self.pool_every)) +1


        #in_h shape
        padding = 1
        pooling_padding = 0
        pooling_kernal_size = self.pooling_params['kernel_size']
        pooling_stride = pooling_kernal_size
        kernel_size = 3
        stride = 1
        negative_slope = None
        if 'padding' in self.conv_params:
            padding = self.conv_params['padding']
        if 'kernel_size' in self.conv_params:
            kernel_size = self.conv_params['kernel_size']
        if 'stride' in self.conv_params:
            stride = self.conv_params['stride']
        if 'negative_slope' in self.activation_params:
            negative_slope = self.activation_params['negative_slope']

        #conv
        #print("in_h = ",in_h)
        in_h = int((in_h+2*padding-1*(kernel_size-1)-1)/(stride)+1)
        for i in range(1,N):
            if ((i % self.pool_every)==0):
                #pooling
                #print("in_h = ",in_h)
                in_h = int((in_h+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)
            #conv
            #print("in_h = ",in_h)
            in_h = int((in_h+2*padding-1*(kernel_size-1)-1)/(stride)+1)

        """if 'kernel_size' in self.conv_params:
            #pooling
            #print("in_h = ",in_h)
            in_h = int((in_h+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)"""
        if ((N % self.pool_every)==0):
            in_h = int((in_h+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)
        #print("in_h = ",in_h)
        #in_w shape
        #conv
        in_w = int((in_w+2*padding-1*(kernel_size-1)-1)/(stride)+1)
        for i in range(1,N):
            if ((i % self.pool_every)==0):
                #pooling
                in_w = int((in_w+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)
            #conv
            in_w = int((in_w+2*padding-1*(kernel_size-1)-1)/(stride)+1)
        """if 'kernel_size' in self.conv_params:
            #pooling
            #print("in_w = ",in_w)
            in_w = int((in_h+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)"""
        if ((N % self.pool_every)==0):
            in_w = int((in_w+2*pooling_padding-pooling_kernal_size)/(pooling_stride)+1)
        
        in_c = self.channels[N-1]
        input_shape = in_c * in_h * in_w

        #layer 1 with input shape
        layers.append(torch.nn.Linear(input_shape, self.hidden_dims[0]))
        if self.activation_type == 'relu':
            layers.append(ACTIVATIONS[self.activation_type](self.activation_params))
        elif negative_slope != None:
            layers.append(ACTIVATIONS[self.activation_type](self.activation_params['negative_slope']))
        else:
            layers.append(ACTIVATIONS[self.activation_type]())

        n_hidden = len(self.hidden_dims)
        for i in range(1,n_hidden):
            layers.append(torch.nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
            if self.activation_type == 'relu':
                layers.append(ACTIVATIONS[self.activation_type](self.activation_params))
            elif negative_slope != None:
                layers.append(ACTIVATIONS[self.activation_type](self.activation_params['negative_slope']))
            else:
                layers.append(ACTIVATIONS[self.activation_type]())
        layers.append(torch.nn.Linear(self.hidden_dims[n_hidden-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        batchnorm=False,
        dropout=0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []
        shortcut_layers = []

        #main path

        # - extract number of conv layers
        N = len(channels)

        # - first conv layer 
        main_layers.append(nn.Conv2d (in_channels, channels[0], kernel_size= kernel_sizes[0],padding=(int((kernel_sizes[0]-1)/2),int((kernel_sizes[0]-1)/2)), bias=True))
        if dropout !=0:
            main_layers.append(torch.nn.Dropout2d(p=dropout))
        if batchnorm == True:
            main_layers.append(torch.nn.BatchNorm2d(channels[0]))
        main_layers.append(ACTIVATIONS[activation_type]())

        #middle layers
        for i in range(1,N-1):
            main_layers.append(nn.Conv2d (channels[i-1], channels[i], kernel_size= kernel_sizes[i],padding=(int((kernel_sizes[i]-1)/2),int((kernel_sizes[i]-1)/2)), bias=True))
            if dropout !=0:
                main_layers.append(torch.nn.Dropout2d(p=dropout))
            if batchnorm == True:
                main_layers.append(torch.nn.BatchNorm2d(channels[i]))
            main_layers.append(ACTIVATIONS[activation_type]())
        if N > 1:
            main_layers.append(nn.Conv2d (channels[N-2], channels[N-1], kernel_size= kernel_sizes[N-1],padding=(int((kernel_sizes[N-1]-1)/2),int((kernel_sizes[N-1]-1)/2)), bias=True))
        if (in_channels != channels[N-1]):
            shortcut_layers.append(nn.Conv2d (in_channels, channels[N-1], kernel_size= 1, bias=False))
        #shortcut_layers.append(torch.nn.BatchNorm2d(channels[N-1]))

        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out = out + self.shortcut_path(x)
        relu = torch.nn.ReLU()
        out = relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        layers = []

        # - extract number of conv layers
        N = len(self.channels)

        """#1st layer
        layers.append(ResidualBlock(in_channels=in_channels, channels=[self.channels[0]], kernel_sizes=[3], batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type))

        #middle layers
        for i in range(1,N-1):
            if ((i % self.pool_every)==0):
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
            layers.append(ResidualBlock(in_channels=self.channels[i-1], channels=[self.channels[i]], kernel_sizes=[3], batchnorm=self.batchnorm, dropout=self.dropout))
        layers.append(ResidualBlock(in_channels=self.channels[N-2], channels=[self.channels[N-1]], kernel_sizes=[3], batchnorm=self.batchnorm, dropout=self.dropout))
        layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))"""

        #1st layer
        #layers.append(ResidualBlock(in_channels=in_channels, channels=[self.channels[0]], kernel_sizes=[3], batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type))
        temp_in_channels = in_channels
        temp_channels = []
        temp_kernel_sizes = []
        #middle layers
        for i in range(1,N):
            temp_channels.append(self.channels[i-1])
            temp_kernel_sizes.append(3)
            if ((i % self.pool_every)==0 and i!=0):
                layers.append(ResidualBlock(in_channels=temp_in_channels, channels=temp_channels, kernel_sizes=temp_kernel_sizes, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type))
                temp_in_channels = self.channels[i-1]
                temp_channels = []
                temp_kernel_sizes = []
                layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))
        temp_channels.append(self.channels[N-1])
        temp_kernel_sizes.append(3)
        layers.append(ResidualBlock(in_channels=temp_in_channels, channels=temp_channels, kernel_sizes=temp_kernel_sizes, batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type))
        if ((N % self.pool_every)==0):
            layers.append(POOLINGS[self.pooling_type](self.pooling_params['kernel_size']))

        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
        in_channels = self.in_size[0]
        main_layers = []
        #shortcut_layers = []

        #main path

        # - extract number of conv layers
        N = len(channels)

        # - first conv layer 
        main_layers.append(nn.Conv2d (in_channels, self.channels[0], kernel_size= 3, stride=1, padding=1, bias=True))
        #main_layers.append(torch.nn.Dropout2d(p=dropout))
        main_layers.append(torch.nn.BatchNorm2d(self.channels[0], eps=1e-05, momentum=0.1, affine=True))
        main_layers.append(torch.nn.ReLU())

        #middle layers
        for i in range(1,N-1):
            if i % pool_every == 0:
                main_layers.append(torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False))
                main_layers.append(torch.nn.Dropout2d(p=0.1))
            main_layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size= 3, stride=1, padding=1, bias=True))
            main_layers.append(nn.BatchNorm2d(channels[i], eps=1e-05, momentum=0.1, affine=True))
            main_layers.append(nn.ReLU())
        if N > 1:
            main_layers.append(nn.Conv2d(channels[N-2], channels[N-1], kernel_size= 3, stride=1, padding=1, bias=True))
        #if (in_channels != channels[N-1]):
            #shortcut_layers.append(nn.Conv2d (in_channels, channels[N-1], kernel_size= 1, bias=False))
        #shortcut_layers.append(torch.nn.BatchNorm2d(channels[N-1]))

        self.main_path = nn.Sequential(*main_layers)
        #self.shortcut_path = nn.Sequential(*shortcut_layers)
    # ========================
