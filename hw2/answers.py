r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. x dim is N x din.

   W dim is Din x dout.
   
   z = xw, and his dim is N x dout
   
   dz/dx dim is (N x din) x (N x dout)
   
   based on that every input meets every output


2. 4 byte per value and we have N^2 * din * dout * values

   (2^7 * 2^7) * (2^10) * (2^11) * (2^2) = (2^37) = 128GB
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.05
    reg = 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.5
    lr_vanilla = 0.05
    lr_momentum = 0.005
    lr_rmsprop = 0.0001
    reg = 0.001
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
#     wstd, lr, = 0.1, 3e-3
    wstd, lr = 10, 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
1. The graphs matched our expectations.
   we expected the model with no dropout to overfit the training set, and it did. it is aparent from the higher train accuracy and lower loss, in contrast to the test results. we can even see the test loss go up from the 2nd epoch.
   the models with dropout showed closer train-test results.

2. surprisingly, the 0.8-dropout performed better then the 0.4-dropout, lower loss and slightly higher accuracy.
   we can see the test accuracy of 0.4-dropout did not significantly improve since epoch 10 and the test loss even went up.
   we can also see the 0.8-dropout loss keeps going down and the accuracy keeps improving, so training for a few more epochs could lead to the 0.8-dropout model performing significantly better then the 0.4-dropout model.
"""

part2_q2 = r"""
it is possible, since the loss and accuracy evaluate the model in a numerically different way.
the loss is calculated based on the raw model outputs, while the accuracy is measured via the argmax over the raw class scores and is not effected by the actual values.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
1. In order to get the number of parameters in CONV2D layer we need to calacute $C_{in}, C_{out}, K_x$ and $K_y.$
    the number of parameters is ($C_{in} * K_x * K_y + 1)C_{out}.$
    a. "Regular" block:
        conv layer with kernal size (3, 3), $K_x = K_y = 3, C_{out}= C_{in} = 256$ and total (256*3*3+1) * 256 = 590,080 parameters.
        we have 2 same layers so in total 590,080 + 590,080 = 1,118,160 parameters.
    b. "Bottleneck" block:
        The $1^{st}$ conv layer with kernal size $(1, 1), K_x=K_y=1, C_{in}=256$ and $C_{out}=64$ and total $(256*1*1+1) * 64 = 16,448$ parameters.
        The $2^{nd}$ conv layer with kernal size $(3, 3), K_x=K_y=3, C_{in}=64$ and $C_{out}=64$ and total $(64*3*3+1) * 64 = 36,928$ parameters.
        The $3^{rd}$ conv layer with kernal size (1, 1), $K_x$=$K_y$=3, $C_{in}$=64 and $C_{out}$=256 and total (64*1*1+1) * 256 = 16,640 parameters.

        total parameters in "Bottleneck" block is 70,016 parameters.

2. In order to get the number of floating point operations in CONV2D layer we need to calacute $C_{in}, I_x, I_y, C_{out}, K_x$ and $K_y.$
    the number of floating point operations is 2 * $C_{in} * I_x * I_y * K_x * K_y * C_{out}.$
    a. "Regular" block:
        conv layer with kernal size (3, 3), $K_x = K_y = 3, C_{out}= C_{in} = 256$ and total $2*256*3*3*256*I_x*I_y = 1,179,648*I_x*I_y$
        we have 2 same layers so $1,179,648*I_x*I_y + 1,179,648*I_x*I_y = 2,359,296 *I_x*I_y$.
        Relu will have $256 * I_x*I_y$.
        Total = $2,359,808*I_x*I_y$.

    b. "Bottleneck" block:
        The $1^{st}$ conv layer with kernal size $(1, 1), K_x=K_y=1, C_{in}=256$ and $C_{out}=64$ and total $2*256*1*1*64*I_x*I_y = 32,768$
        The $2^{nd}$ conv layer with kernal size $(3, 3), K_x=K_y=3, C_{in}=64$ and $C_{out}=64$ and total $2*64*1*1*64*I_x*I_y = 73,728$
        The $3^{rd}$ conv layer with kernal size (1, 1), $K_x$=$K_y$=3, $C_{in}$=64 and $C_{out}$=256 and total $2*64*1*1*256*I_x*I_y = 32,768$
        The shortcat path is $256*I_x*I_y $
        The Relu have $256 * I_x*I_y$.
        Total = $139,776*I_x*I_y$ floating point operations

3. ????????????????????
"""

part3_q2 = r"""
1.  In EXP 1.1 we can see that deeper the depth the ACC is lower, we think it happen beacuse there is more parameters in deeper networks.
    when you have too much parameters you can suffer from overfitting that cause them to lower ACC.
    the best depth is L2 with K32, the shortest one.

2. when L was 16 the network didnt learnd nothing becuase the vanishing gradinent inside the block.
    there is 2 options to improve it,
    a.  is to reduce K any reducing the vanishing problem
    b.  normilze the gradient will not give him to go too high or too low and by that aviod the vanishing problem.
"""

part3_q3 = r"""
In experiment 1.2 we can see few things:
    1. when L=2
        a. almost all the networks start overfitting after 4 iterations, K=128 is still learning and start the pverfitting only in iteration 8
        b. K = 64 is has the best perfomence and got almost 70% ACC on the test and has sharper coeffiecnt in the loss
    2. When L = 4
        a. all the networks start overfitting after 4 iterations
        b. K = 128 is has the best perfomence and got 70% ACC on the test and has sharper coeffiecnt in the loss
    3. When L = 8
        a. all the networks start overfitting after 4-5 iterations
        b. K = 256 got the best ACC with almost 70%
    We can see from the experiment that when you are inceasing the L you need to increase the K in parallel.
"""

part3_q4 = r"""
In experiment 1.3 we can see few things:
    1. all the networks start to overfit after 6-8 iterations
    2. L= 2 got the best ACC ~78%, althourgh L = 3,4 learned more EPOCHs.
    3. when you have more layers it takes more epochs to arrive to overfitting
"""

part3_q5 = r"""
In experiment 1.4 we can see few things:
    1. K=[32] fixed with L=8,16,32 varying per run.
        a. L = 32 has too much parameters and he suffer from vanishing gradient with low ACC
        b. L =8 has the same results has L = 16 but got it faster and in less epochs.
    2. K=[64, 128, 256] fixed with L=2,4,8 varying per run.
        a. In this EXP we can see that lower the number of layers the prefemnce are better, probabaly vanishing gradient
        b. L = 2 achive more than 75% ACC, the best results till this EXP.
        c. L = 2 loss getting much lower than the 2 others
1.1 vs 1.4
    1. We can see that adding the skip connection is improving the ACC significantly from ~53% ACC to almost 70%, 32% imporvment
    2. We got the same loss in 1.4 after 3 epochs instead of 8-9, the gradient is much more efficent

1.3 vs 1.4
    1. In L = 2 and 4,  1.4 has ACC little bit better with less epochs.
    2. shortcut is giving 1.4 better perfomence and avoding vanishing gradient.
"""

part3_q6 = r"""
1. In our network we combined ResidualBlock with finetunning to parameters using Dropout (0.2) and BN.
    we can see that our network got ~80% ACC and didnt suffer from overfitting till EPOCH 35 minimum.
    ES was ajust to 5 epochs in row becuase sometims we have epochs the loss is increasing but we still not in the global minimum
2. in exp 1.5 we got much better results, more than 6% from the best exp in part 1.
   tha main different is because we changed the ES and give the network chance to keep the decresing loss path, it is very importent to not stoping her too soon.
   beside of that we had knewledge from part 1 so we used it in part 2 and got better prefemnce from the start.
"""
# ==============
