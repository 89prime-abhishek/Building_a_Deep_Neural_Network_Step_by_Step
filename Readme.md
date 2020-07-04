{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### <font color='darkblue'> Updates to Assignment <font>\n",
    "\n",
    "#### If you were working on a previous version\n",
    "* The current notebook filename is version \"4a\". \n",
    "* You can find your work in the file directory as version \"4\".\n",
    "* To see the file directory, click on the Coursera logo at the top left of the notebook.\n",
    "\n",
    "#### List of Updates\n",
    "* compute_cost unit test now includes tests for Y = 0 as well as Y = 1.  This catches a possible bug before students get graded.\n",
    "* linear_backward unit test now has a more complete unit test that catches a possible bug before students get graded.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1 - Packages\n",
    "\n",
    "Let's first import all the packages that you will need during this assignment. \n",
    "- [numpy](www.numpy.org) is the main package for scientific computing with Python.\n",
    "- [matplotlib](http://matplotlib.org) is a library to plot graphs in Python.\n",
    "- dnn_utils provides some necessary functions for this notebook.\n",
    "- testCases provides some test cases to assess the correctness of your functions\n",
    "- np.random.seed(1) is used to keep all the random function calls consistent. It will help us grade your work. Please don't change the seed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import h5py\n",
    "import matplotlib.pyplot as plt\n",
    "from testCases_v4a import *\n",
    "from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward\n",
    "\n",
    "%matplotlib inline\n",
    "plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots\n",
    "plt.rcParams['image.interpolation'] = 'nearest'\n",
    "plt.rcParams['image.cmap'] = 'gray'\n",
    "\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "np.random.seed(1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2 - Outline of the Assignment\n",
    "\n",
    "To build your neural network, you will be implementing several \"helper functions\". These helper functions will be used in the next assignment to build a two-layer neural network and an L-layer neural network. Each small helper function you will implement will have detailed instructions that will walk you through the necessary steps. Here is an outline of this assignment, you will:\n",
    "\n",
    "- Initialize the parameters for a two-layer network and for an $L$-layer neural network.\n",
    "- Implement the forward propagation module (shown in purple in the figure below).\n",
    "     - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).\n",
    "     - We give you the ACTIVATION function (relu/sigmoid).\n",
    "     - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.\n",
    "     - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). This gives you a new L_model_forward function.\n",
    "- Compute the loss.\n",
    "- Implement the backward propagation module (denoted in red in the figure below).\n",
    "    - Complete the LINEAR part of a layer's backward propagation step.\n",
    "    - We give you the gradient of the ACTIVATE function (relu_backward/sigmoid_backward) \n",
    "    - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.\n",
    "    - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function\n",
    "- Finally update the parameters.\n",
    "\n",
    "<img src=\"images/final outline.png\" style=\"width:800px;height:500px;\">\n",
    "<caption><center> **Figure 1**</center></caption><br>\n",
    "\n",
    "\n",
    "**Note** that for every forward function, there is a corresponding backward function. That is why at every step of your forward module you will be storing some values in a cache. The cached values are useful for computing gradients. In the backpropagation module you will then use the cache to calculate the gradients. This assignment will show you exactly how to carry out each of these steps. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3 - Initialization\n",
    "\n",
    "You will write two helper functions that will initialize the parameters for your model. The first function will be used to initialize parameters for a two layer model. The second one will generalize this initialization process to $L$ layers.\n",
    "\n",
    "### 3.1 - 2-layer Neural Network\n",
    "\n",
    "**Exercise**: Create and initialize the parameters of the 2-layer neural network.\n",
    "\n",
    "**Instructions**:\n",
    "- The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. \n",
    "- Use random initialization for the weight matrices. Use `np.random.randn(shape)*0.01` with the correct shape.\n",
    "- Use zero initialization for the biases. Use `np.zeros(shape)`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: initialize_parameters\n",
    "\n",
    "def initialize_parameters(n_x, n_h, n_y):\n",
    "    \"\"\"\n",
    "    Argument:\n",
    "    n_x -- size of the input layer\n",
    "    n_h -- size of the hidden layer\n",
    "    n_y -- size of the output layer\n",
    "    \n",
    "    Returns:\n",
    "    parameters -- python dictionary containing your parameters:\n",
    "                    W1 -- weight matrix of shape (n_h, n_x)\n",
    "                    b1 -- bias vector of shape (n_h, 1)\n",
    "                    W2 -- weight matrix of shape (n_y, n_h)\n",
    "                    b2 -- bias vector of shape (n_y, 1)\n",
    "    \"\"\"\n",
    "    \n",
    "    np.random.seed(1)\n",
    "    \n",
    "    ### START CODE HERE ### (≈ 4 lines of code)\n",
    "    W1 = np.random.randn(n_h,n_x)*0.01\n",
    "    b1 = np.zeros((n_h,1))\n",
    "    W2 = np.random.randn(n_y,n_h)*0.01\n",
    "    b2 = np.zeros((n_y,1))\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    assert(W1.shape == (n_h, n_x))\n",
    "    assert(b1.shape == (n_h, 1))\n",
    "    assert(W2.shape == (n_y, n_h))\n",
    "    assert(b2.shape == (n_y, 1))\n",
    "    \n",
    "    parameters = {\"W1\": W1,\n",
    "                  \"b1\": b1,\n",
    "                  \"W2\": W2,\n",
    "                  \"b2\": b2}\n",
    "    \n",
    "    return parameters    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "W1 = [[ 0.01624345 -0.00611756 -0.00528172]\n",
      " [-0.01072969  0.00865408 -0.02301539]]\n",
      "b1 = [[ 0.]\n",
      " [ 0.]]\n",
      "W2 = [[ 0.01744812 -0.00761207]]\n",
      "b2 = [[ 0.]]\n"
     ]
    }
   ],
   "source": [
    "parameters = initialize_parameters(3,2,1)\n",
    "print(\"W1 = \" + str(parameters[\"W1\"]))\n",
    "print(\"b1 = \" + str(parameters[\"b1\"]))\n",
    "print(\"W2 = \" + str(parameters[\"W2\"]))\n",
    "print(\"b2 = \" + str(parameters[\"b2\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output**:\n",
    "       \n",
    "<table style=\"width:80%\">\n",
    "  <tr>\n",
    "    <td> **W1** </td>\n",
    "    <td> [[ 0.01624345 -0.00611756 -0.00528172]\n",
    " [-0.01072969  0.00865408 -0.02301539]] </td> \n",
    "  </tr>\n",
    "\n",
    "  <tr>\n",
    "    <td> **b1**</td>\n",
    "    <td>[[ 0.]\n",
    " [ 0.]]</td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**W2**</td>\n",
    "    <td> [[ 0.01744812 -0.00761207]]</td>\n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td> **b2** </td>\n",
    "    <td> [[ 0.]] </td> \n",
    "  </tr>\n",
    "  \n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.2 - L-layer Neural Network\n",
    "\n",
    "The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep`, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. Thus for example if the size of our input $X$ is $(12288, 209)$ (with $m=209$ examples) then:\n",
    "\n",
    "<table style=\"width:100%\">\n",
    "\n",
    "\n",
    "    <tr>\n",
    "        <td>  </td> \n",
    "        <td> **Shape of W** </td> \n",
    "        <td> **Shape of b**  </td> \n",
    "        <td> **Activation** </td>\n",
    "        <td> **Shape of Activation** </td> \n",
    "    <tr>\n",
    "    \n",
    "    <tr>\n",
    "        <td> **Layer 1** </td> \n",
    "        <td> $(n^{[1]},12288)$ </td> \n",
    "        <td> $(n^{[1]},1)$ </td> \n",
    "        <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td> \n",
    "        \n",
    "        <td> $(n^{[1]},209)$ </td> \n",
    "    <tr>\n",
    "    \n",
    "    <tr>\n",
    "        <td> **Layer 2** </td> \n",
    "        <td> $(n^{[2]}, n^{[1]})$  </td> \n",
    "        <td> $(n^{[2]},1)$ </td> \n",
    "        <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td> \n",
    "        <td> $(n^{[2]}, 209)$ </td> \n",
    "    <tr>\n",
    "   \n",
    "       <tr>\n",
    "        <td> $\\vdots$ </td> \n",
    "        <td> $\\vdots$  </td> \n",
    "        <td> $\\vdots$  </td> \n",
    "        <td> $\\vdots$</td> \n",
    "        <td> $\\vdots$  </td> \n",
    "    <tr>\n",
    "    \n",
    "   <tr>\n",
    "        <td> **Layer L-1** </td> \n",
    "        <td> $(n^{[L-1]}, n^{[L-2]})$ </td> \n",
    "        <td> $(n^{[L-1]}, 1)$  </td> \n",
    "        <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td> \n",
    "        <td> $(n^{[L-1]}, 209)$ </td> \n",
    "    <tr>\n",
    "    \n",
    "    \n",
    "   <tr>\n",
    "        <td> **Layer L** </td> \n",
    "        <td> $(n^{[L]}, n^{[L-1]})$ </td> \n",
    "        <td> $(n^{[L]}, 1)$ </td>\n",
    "        <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>\n",
    "        <td> $(n^{[L]}, 209)$  </td> \n",
    "    <tr>\n",
    "\n",
    "</table>\n",
    "\n",
    "Remember that when we compute $W X + b$ in python, it carries out broadcasting. For example, if: \n",
    "\n",
    "$$ W = \\begin{bmatrix}\n",
    "    j  & k  & l\\\\\n",
    "    m  & n & o \\\\\n",
    "    p  & q & r \n",
    "\\end{bmatrix}\\;\\;\\; X = \\begin{bmatrix}\n",
    "    a  & b  & c\\\\\n",
    "    d  & e & f \\\\\n",
    "    g  & h & i \n",
    "\\end{bmatrix} \\;\\;\\; b =\\begin{bmatrix}\n",
    "    s  \\\\\n",
    "    t  \\\\\n",
    "    u\n",
    "\\end{bmatrix}\\tag{2}$$\n",
    "\n",
    "Then $WX + b$ will be:\n",
    "\n",
    "$$ WX + b = \\begin{bmatrix}\n",
    "    (ja + kd + lg) + s  & (jb + ke + lh) + s  & (jc + kf + li)+ s\\\\\n",
    "    (ma + nd + og) + t & (mb + ne + oh) + t & (mc + nf + oi) + t\\\\\n",
    "    (pa + qd + rg) + u & (pb + qe + rh) + u & (pc + qf + ri)+ u\n",
    "\\end{bmatrix}\\tag{3}  $$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise**: Implement initialization for an L-layer Neural Network. \n",
    "\n",
    "**Instructions**:\n",
    "- The model's structure is *[LINEAR -> RELU] $ \\times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.\n",
    "- Use random initialization for the weight matrices. Use `np.random.randn(shape) * 0.01`.\n",
    "- Use zeros initialization for the biases. Use `np.zeros(shape)`.\n",
    "- We will store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for the \"Planar Data classification model\" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. This means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers! \n",
    "- Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).\n",
    "```python\n",
    "    if L == 1:\n",
    "        parameters[\"W\" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01\n",
    "        parameters[\"b\" + str(L)] = np.zeros((layer_dims[1], 1))\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: initialize_parameters_deep\n",
    "\n",
    "def initialize_parameters_deep(layer_dims):\n",
    "    \"\"\"\n",
    "    Arguments:\n",
    "    layer_dims -- python array (list) containing the dimensions of each layer in our network\n",
    "    \n",
    "    Returns:\n",
    "    parameters -- python dictionary containing your parameters \"W1\", \"b1\", ..., \"WL\", \"bL\":\n",
    "                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])\n",
    "                    bl -- bias vector of shape (layer_dims[l], 1)\n",
    "    \"\"\"\n",
    "    \n",
    "    np.random.seed(3)\n",
    "    parameters = {}\n",
    "    L = len(layer_dims)            # number of layers in the network\n",
    "\n",
    "    for l in range(1, L):\n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01\n",
    "        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))\n",
    "        ### END CODE HERE ###\n",
    "        \n",
    "        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))\n",
    "        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))\n",
    "\n",
    "        \n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "W1 = [[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]\n",
      " [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]\n",
      " [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]\n",
      " [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]\n",
      "b1 = [[ 0.]\n",
      " [ 0.]\n",
      " [ 0.]\n",
      " [ 0.]]\n",
      "W2 = [[-0.01185047 -0.0020565   0.01486148  0.00236716]\n",
      " [-0.01023785 -0.00712993  0.00625245 -0.00160513]\n",
      " [-0.00768836 -0.00230031  0.00745056  0.01976111]]\n",
      "b2 = [[ 0.]\n",
      " [ 0.]\n",
      " [ 0.]]\n"
     ]
    }
   ],
   "source": [
    "parameters = initialize_parameters_deep([5,4,3])\n",
    "print(\"W1 = \" + str(parameters[\"W1\"]))\n",
    "print(\"b1 = \" + str(parameters[\"b1\"]))\n",
    "print(\"W2 = \" + str(parameters[\"W2\"]))\n",
    "print(\"b2 = \" + str(parameters[\"b2\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output**:\n",
    "       \n",
    "<table style=\"width:80%\">\n",
    "  <tr>\n",
    "    <td> **W1** </td>\n",
    "    <td>[[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]\n",
    " [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]\n",
    " [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]\n",
    " [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]</td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**b1** </td>\n",
    "    <td>[[ 0.]\n",
    " [ 0.]\n",
    " [ 0.]\n",
    " [ 0.]]</td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**W2** </td>\n",
    "    <td>[[-0.01185047 -0.0020565   0.01486148  0.00236716]\n",
    " [-0.01023785 -0.00712993  0.00625245 -0.00160513]\n",
    " [-0.00768836 -0.00230031  0.00745056  0.01976111]]</td> \n",
    "  </tr>\n",
    "  \n",
    "  <tr>\n",
    "    <td>**b2** </td>\n",
    "    <td>[[ 0.]\n",
    " [ 0.]\n",
    " [ 0.]]</td> \n",
    "  </tr>\n",
    "  \n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4 - Forward propagation module\n",
    "\n",
    "### 4.1 - Linear Forward \n",
    "Now that you have initialized your parameters, you will do the forward propagation module. You will start by implementing some basic functions that you will use later when implementing the model. You will complete three functions in this order:\n",
    "\n",
    "- LINEAR\n",
    "- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid. \n",
    "- [LINEAR -> RELU] $\\times$ (L-1) -> LINEAR -> SIGMOID (whole model)\n",
    "\n",
    "The linear forward module (vectorized over all the examples) computes the following equations:\n",
    "\n",
    "$$Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}\\tag{4}$$\n",
    "\n",
    "where $A^{[0]} = X$. \n",
    "\n",
    "**Exercise**: Build the linear part of forward propagation.\n",
    "\n",
    "**Reminder**:\n",
    "The mathematical representation of this unit is $Z^{[l]} = W^{[l]}A^{[l-1]} +b^{[l]}$. You may also find `np.dot()` useful. If your dimensions don't match, printing `W.shape` may help."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: linear_forward\n",
    "\n",
    "def linear_forward(A, W, b):\n",
    "    \"\"\"\n",
    "    Implement the linear part of a layer's forward propagation.\n",
    "\n",
    "    Arguments:\n",
    "    A -- activations from previous layer (or input data): (size of previous layer, number of examples)\n",
    "    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)\n",
    "    b -- bias vector, numpy array of shape (size of the current layer, 1)\n",
    "\n",
    "    Returns:\n",
    "    Z -- the input of the activation function, also called pre-activation parameter \n",
    "    cache -- a python tuple containing \"A\", \"W\" and \"b\" ; stored for computing the backward pass efficiently\n",
    "    \"\"\"\n",
    "    \n",
    "    ### START CODE HERE ### (≈ 1 line of code)\n",
    "    Z = np.dot(W,A)+b\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    assert(Z.shape == (W.shape[0], A.shape[1]))\n",
    "    cache = (A, W, b)\n",
    "    \n",
    "    return Z, cache"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Z = [[ 3.26295337 -1.23429987]]\n"
     ]
    }
   ],
   "source": [
    "A, W, b = linear_forward_test_case()\n",
    "\n",
    "Z, linear_cache = linear_forward(A, W, b)\n",
    "print(\"Z = \" + str(Z))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output**:\n",
    "\n",
    "<table style=\"width:35%\">\n",
    "  \n",
    "  <tr>\n",
    "    <td> **Z** </td>\n",
    "    <td> [[ 3.26295337 -1.23429987]] </td> \n",
    "  </tr>\n",
    "  \n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 - Linear-Activation Forward\n",
    "\n",
    "In this notebook, you will use two activation functions:\n",
    "\n",
    "- **Sigmoid**: $\\sigma(Z) = \\sigma(W A + b) = \\frac{1}{ 1 + e^{-(W A + b)}}$. We have provided you with the `sigmoid` function. This function returns **two** items: the activation value \"`a`\" and a \"`cache`\" that contains \"`Z`\" (it's what we will feed in to the corresponding backward function). To use it you could just call: \n",
    "``` python\n",
    "A, activation_cache = sigmoid(Z)\n",
    "```\n",
    "\n",
    "- **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. We have provided you with the `relu` function. This function returns **two** items: the activation value \"`A`\" and a \"`cache`\" that contains \"`Z`\" (it's what we will feed in to the corresponding backward function). To use it you could just call:\n",
    "``` python\n",
    "A, activation_cache = relu(Z)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For more convenience, you are going to group two functions (Linear and Activation) into one function (LINEAR->ACTIVATION). Hence, you will implement a function that does the LINEAR forward step followed by an ACTIVATION forward step.\n",
    "\n",
    "**Exercise**: Implement the forward propagation of the *LINEAR->ACTIVATION* layer. Mathematical relation is: $A^{[l]} = g(Z^{[l]}) = g(W^{[l]}A^{[l-1]} +b^{[l]})$ where the activation \"g\" can be sigmoid() or relu(). Use linear_forward() and the correct activation function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: linear_activation_forward\n",
    "\n",
    "def linear_activation_forward(A_prev, W, b, activation):\n",
    "    \"\"\"\n",
    "    Implement the forward propagation for the LINEAR->ACTIVATION layer\n",
    "\n",
    "    Arguments:\n",
    "    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)\n",
    "    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)\n",
    "    b -- bias vector, numpy array of shape (size of the current layer, 1)\n",
    "    activation -- the activation to be used in this layer, stored as a text string: \"sigmoid\" or \"relu\"\n",
    "\n",
    "    Returns:\n",
    "    A -- the output of the activation function, also called the post-activation value \n",
    "    cache -- a python tuple containing \"linear_cache\" and \"activation_cache\";\n",
    "             stored for computing the backward pass efficiently\n",
    "    \"\"\"\n",
    "    \n",
    "    if activation == \"sigmoid\":\n",
    "        # Inputs: \"A_prev, W, b\". Outputs: \"A, activation_cache\".\n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
    "        A, activation_cache = sigmoid(Z)\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "    elif activation == \"relu\":\n",
    "        # Inputs: \"A_prev, W, b\". Outputs: \"A, activation_cache\".\n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
    "        A, activation_cache = relu(Z)\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "    assert (A.shape == (W.shape[0], A_prev.shape[1]))\n",
    "    cache = (linear_cache, activation_cache)\n",
    "\n",
    "    return A, cache"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "With sigmoid: A = [[ 0.96890023  0.11013289]]\n",
      "With ReLU: A = [[ 3.43896131  0.        ]]\n"
     ]
    }
   ],
   "source": [
    "A_prev, W, b = linear_activation_forward_test_case()\n",
    "\n",
    "A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = \"sigmoid\")\n",
    "print(\"With sigmoid: A = \" + str(A))\n",
    "\n",
    "A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = \"relu\")\n",
    "print(\"With ReLU: A = \" + str(A))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output**:\n",
    "       \n",
    "<table style=\"width:35%\">\n",
    "  <tr>\n",
    "    <td> **With sigmoid: A ** </td>\n",
    "    <td > [[ 0.96890023  0.11013289]]</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td> **With ReLU: A ** </td>\n",
    "    <td > [[ 3.43896131  0.        ]]</td> \n",
    "  </tr>\n",
    "</table>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Note**: In deep learning, the \"[LINEAR->ACTIVATION]\" computation is counted as a single layer in the neural network, not two layers. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### d) L-Layer Model \n",
    "\n",
    "For even more convenience when implementing the $L$-layer Neural Net, you will need a function that replicates the previous one (`linear_activation_forward` with RELU) $L-1$ times, then follows that with one `linear_activation_forward` with SIGMOID.\n",
    "\n",
    "<img src=\"images/model_architecture_kiank.png\" style=\"width:600px;height:300px;\">\n",
    "<caption><center> **Figure 2** : *[LINEAR -> RELU] $\\times$ (L-1) -> LINEAR -> SIGMOID* model</center></caption><br>\n",
    "\n",
    "**Exercise**: Implement the forward propagation of the above model.\n",
    "\n",
    "**Instruction**: In the code below, the variable `AL` will denote $A^{[L]} = \\sigma(Z^{[L]}) = \\sigma(W^{[L]} A^{[L-1]} + b^{[L]})$. (This is sometimes also called `Yhat`, i.e., this is $\\hat{Y}$.) \n",
    "\n",
    "**Tips**:\n",
    "- Use the functions you had previously written \n",
    "- Use a for loop to replicate [LINEAR->RELU] (L-1) times\n",
    "- Don't forget to keep track of the caches in the \"caches\" list. To add a new value `c` to a `list`, you can use `list.append(c)`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: L_model_forward\n",
    "\n",
    "def L_model_forward(X, parameters):\n",
    "    \"\"\"\n",
    "    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation\n",
    "    \n",
    "    Arguments:\n",
    "    X -- data, numpy array of shape (input size, number of examples)\n",
    "    parameters -- output of initialize_parameters_deep()\n",
    "    \n",
    "    Returns:\n",
    "    AL -- last post-activation value\n",
    "    caches -- list of caches containing:\n",
    "                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)\n",
    "    \"\"\"\n",
    "\n",
    "    caches = []\n",
    "    A = X\n",
    "    L = len(parameters) // 2                  # number of layers in the neural network\n",
    "    \n",
    "    # Implement [LINEAR -> RELU]*(L-1). Add \"cache\" to the \"caches\" list.\n",
    "    for l in range(1, L):\n",
    "        A_prev = A \n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        A, cache = linear_activation_forward(A_prev, \n",
    "                                             parameters['W' + str(l)], \n",
    "                                             parameters['b' + str(l)], \n",
    "                                             activation='relu')\n",
    "        caches.append(cache)\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "    # Implement LINEAR -> SIGMOID. Add \"cache\" to the \"caches\" list.\n",
    "    ### START CODE HERE ### (≈ 2 lines of code)\n",
    "    AL, cache = linear_activation_forward(A, \n",
    "                                             parameters['W' + str(L)], \n",
    "                                             parameters['b' + str(L)], \n",
    "                                             activation='sigmoid')\n",
    "    caches.append(cache)\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    assert(AL.shape == (1,X.shape[1]))\n",
    "            \n",
    "    return AL, caches"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AL = [[ 0.03921668  0.70498921  0.19734387  0.04728177]]\n",
      "Length of caches list = 3\n"
     ]
    }
   ],
   "source": [
    "X, parameters = L_model_forward_test_case_2hidden()\n",
    "AL, caches = L_model_forward(X, parameters)\n",
    "print(\"AL = \" + str(AL))\n",
    "print(\"Length of caches list = \" + str(len(caches)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<table style=\"width:50%\">\n",
    "  <tr>\n",
    "    <td> **AL** </td>\n",
    "    <td > [[ 0.03921668  0.70498921  0.19734387  0.04728177]]</td> \n",
    "  </tr>\n",
    "  <tr>\n",
    "    <td> **Length of caches list ** </td>\n",
    "    <td > 3 </td> \n",
    "  </tr>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Great! Now you have a full forward propagation that takes the input X and outputs a row vector $A^{[L]}$ containing your predictions. It also records all intermediate values in \"caches\". Using $A^{[L]}$, you can compute the cost of your predictions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5 - Cost function\n",
    "\n",
    "Now you will implement forward and backward propagation. You need to compute the cost, because you want to check if your model is actually learning.\n",
    "\n",
    "**Exercise**: Compute the cross-entropy cost $J$, using the following formula: $$-\\frac{1}{m} \\sum\\limits_{i = 1}^{m} (y^{(i)}\\log\\left(a^{[L] (i)}\\right) + (1-y^{(i)})\\log\\left(1- a^{[L](i)}\\right)) \\tag{7}$$\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: compute_cost\n",
    "\n",
    "def compute_cost(AL, Y):\n",
    "    \"\"\"\n",
    "    Implement the cost function defined by equation (7).\n",
    "\n",
    "    Arguments:\n",
    "    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)\n",
    "    Y -- true \"label\" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)\n",
    "\n",
    "    Returns:\n",
    "    cost -- cross-entropy cost\n",
    "    \"\"\"\n",
    "    \n",
    "    m = Y.shape[1]\n",
    "\n",
    "    # Compute loss from aL and y.\n",
    "    ### START CODE HERE ### (≈ 1 lines of code)\n",
    "    cost = (-1/m) * np.sum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL)))\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).\n",
    "    assert(cost.shape == ())\n",
    "    \n",
    "    return cost"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cost = 0.279776563579\n"
     ]
    }
   ],
   "source": [
    "Y, AL = compute_cost_test_case()\n",
    "\n",
    "print(\"cost = \" + str(compute_cost(AL, Y)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**:\n",
    "\n",
    "<table>\n",
    "\n",
    "    <tr>\n",
    "    <td>**cost** </td>\n",
    "    <td> 0.2797765635793422</td> \n",
    "    </tr>\n",
    "</table>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6 - Backward propagation module\n",
    "\n",
    "Just like with forward propagation, you will implement helper functions for backpropagation. Remember that back propagation is used to calculate the gradient of the loss function with respect to the parameters. \n",
    "\n",
    "**Reminder**: \n",
    "<img src=\"images/backprop_kiank.png\" style=\"width:650px;height:250px;\">\n",
    "<caption><center> **Figure 3** : Forward and Backward propagation for *LINEAR->RELU->LINEAR->SIGMOID* <br> *The purple blocks represent the forward propagation, and the red blocks represent the backward propagation.*  </center></caption>\n",
    "\n",
    "<!-- \n",
    "For those of you who are expert in calculus (you don't need to be to do this assignment), the chain rule of calculus can be used to derive the derivative of the loss $\\mathcal{L}$ with respect to $z^{[1]}$ in a 2-layer network as follows:\n",
    "\n",
    "$$\\frac{d \\mathcal{L}(a^{[2]},y)}{{dz^{[1]}}} = \\frac{d\\mathcal{L}(a^{[2]},y)}{{da^{[2]}}}\\frac{{da^{[2]}}}{{dz^{[2]}}}\\frac{{dz^{[2]}}}{{da^{[1]}}}\\frac{{da^{[1]}}}{{dz^{[1]}}} \\tag{8} $$\n",
    "\n",
    "In order to calculate the gradient $dW^{[1]} = \\frac{\\partial L}{\\partial W^{[1]}}$, you use the previous chain rule and you do $dW^{[1]} = dz^{[1]} \\times \\frac{\\partial z^{[1]} }{\\partial W^{[1]}}$. During the backpropagation, at each step you multiply your current gradient by the gradient corresponding to the specific layer to get the gradient you wanted.\n",
    "\n",
    "Equivalently, in order to calculate the gradient $db^{[1]} = \\frac{\\partial L}{\\partial b^{[1]}}$, you use the previous chain rule and you do $db^{[1]} = dz^{[1]} \\times \\frac{\\partial z^{[1]} }{\\partial b^{[1]}}$.\n",
    "\n",
    "This is why we talk about **backpropagation**.\n",
    "!-->\n",
    "\n",
    "Now, similar to forward propagation, you are going to build the backward propagation in three steps:\n",
    "- LINEAR backward\n",
    "- LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation\n",
    "- [LINEAR -> RELU] $\\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.1 - Linear backward\n",
    "\n",
    "For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).\n",
    "\n",
    "Suppose you have already calculated the derivative $dZ^{[l]} = \\frac{\\partial \\mathcal{L} }{\\partial Z^{[l]}}$. You want to get $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$.\n",
    "\n",
    "<img src=\"images/linearback_kiank.png\" style=\"width:250px;height:300px;\">\n",
    "<caption><center> **Figure 4** </center></caption>\n",
    "\n",
    "The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$ are computed using the input $dZ^{[l]}$.Here are the formulas you need:\n",
    "$$ dW^{[l]} = \\frac{\\partial \\mathcal{J} }{\\partial W^{[l]}} = \\frac{1}{m} dZ^{[l]} A^{[l-1] T} \\tag{8}$$\n",
    "$$ db^{[l]} = \\frac{\\partial \\mathcal{J} }{\\partial b^{[l]}} = \\frac{1}{m} \\sum_{i = 1}^{m} dZ^{[l](i)}\\tag{9}$$\n",
    "$$ dA^{[l-1]} = \\frac{\\partial \\mathcal{L} }{\\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \\tag{10}$$\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise**: Use the 3 formulas above to implement linear_backward()."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: linear_backward\n",
    "\n",
    "def linear_backward(dZ, cache):\n",
    "    \"\"\"\n",
    "    Implement the linear portion of backward propagation for a single layer (layer l)\n",
    "\n",
    "    Arguments:\n",
    "    dZ -- Gradient of the cost with respect to the linear output (of current layer l)\n",
    "    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer\n",
    "\n",
    "    Returns:\n",
    "    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev\n",
    "    dW -- Gradient of the cost with respect to W (current layer l), same shape as W\n",
    "    db -- Gradient of the cost with respect to b (current layer l), same shape as b\n",
    "    \"\"\"\n",
    "    A_prev, W, b = cache\n",
    "    m = A_prev.shape[1]\n",
    "\n",
    "    ### START CODE HERE ### (≈ 3 lines of code)\n",
    "    dW = (1/m)*np.dot(dZ, A_prev.T)\n",
    "    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)\n",
    "    dA_prev = np.dot(W.T,dZ)\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    assert (dA_prev.shape == A_prev.shape)\n",
    "    assert (dW.shape == W.shape)\n",
    "    assert (db.shape == b.shape)\n",
    "    \n",
    "    return dA_prev, dW, db"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dA_prev = [[-1.15171336  0.06718465 -0.3204696   2.09812712]\n",
      " [ 0.60345879 -3.72508701  5.81700741 -3.84326836]\n",
      " [-0.4319552  -1.30987417  1.72354705  0.05070578]\n",
      " [-0.38981415  0.60811244 -1.25938424  1.47191593]\n",
      " [-2.52214926  2.67882552 -0.67947465  1.48119548]]\n",
      "dW = [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]\n",
      " [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]\n",
      " [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]\n",
      "db = [[-0.14713786]\n",
      " [-0.11313155]\n",
      " [-0.13209101]]\n"
     ]
    }
   ],
   "source": [
    "# Set up some test inputs\n",
    "dZ, linear_cache = linear_backward_test_case()\n",
    "\n",
    "dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
    "print (\"dA_prev = \"+ str(dA_prev))\n",
    "print (\"dW = \" + str(dW))\n",
    "print (\"db = \" + str(db))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Expected Output**:\n",
    "    \n",
    "```\n",
    "dA_prev = \n",
    " [[-1.15171336  0.06718465 -0.3204696   2.09812712]\n",
    " [ 0.60345879 -3.72508701  5.81700741 -3.84326836]\n",
    " [-0.4319552  -1.30987417  1.72354705  0.05070578]\n",
    " [-0.38981415  0.60811244 -1.25938424  1.47191593]\n",
    " [-2.52214926  2.67882552 -0.67947465  1.48119548]]\n",
    "dW = \n",
    " [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]\n",
    " [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]\n",
    " [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]\n",
    "db = \n",
    " [[-0.14713786]\n",
    " [-0.11313155]\n",
    " [-0.13209101]]\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.2 - Linear-Activation backward\n",
    "\n",
    "Next, you will create a function that merges the two helper functions: **`linear_backward`** and the backward step for the activation **`linear_activation_backward`**. \n",
    "\n",
    "To help you implement `linear_activation_backward`, we provided two backward functions:\n",
    "- **`sigmoid_backward`**: Implements the backward propagation for SIGMOID unit. You can call it as follows:\n",
    "\n",
    "```python\n",
    "dZ = sigmoid_backward(dA, activation_cache)\n",
    "```\n",
    "\n",
    "- **`relu_backward`**: Implements the backward propagation for RELU unit. You can call it as follows:\n",
    "\n",
    "```python\n",
    "dZ = relu_backward(dA, activation_cache)\n",
    "```\n",
    "\n",
    "If $g(.)$ is the activation function, \n",
    "`sigmoid_backward` and `relu_backward` compute $$dZ^{[l]} = dA^{[l]} * g'(Z^{[l]}) \\tag{11}$$.  \n",
    "\n",
    "**Exercise**: Implement the backpropagation for the *LINEAR->ACTIVATION* layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: linear_activation_backward\n",
    "\n",
    "def linear_activation_backward(dA, cache, activation):\n",
    "    \"\"\"\n",
    "    Implement the backward propagation for the LINEAR->ACTIVATION layer.\n",
    "    \n",
    "    Arguments:\n",
    "    dA -- post-activation gradient for current layer l \n",
    "    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently\n",
    "    activation -- the activation to be used in this layer, stored as a text string: \"sigmoid\" or \"relu\"\n",
    "    \n",
    "    Returns:\n",
    "    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev\n",
    "    dW -- Gradient of the cost with respect to W (current layer l), same shape as W\n",
    "    db -- Gradient of the cost with respect to b (current layer l), same shape as b\n",
    "    \"\"\"\n",
    "    linear_cache, activation_cache = cache\n",
    "    \n",
    "    if activation == \"relu\":\n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        dZ = relu_backward(dA, cache[1])\n",
    "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
    "        ### END CODE HERE ###\n",
    "        \n",
    "    elif activation == \"sigmoid\":\n",
    "        ### START CODE HERE ### (≈ 2 lines of code)\n",
    "        dZ = sigmoid_backward(dA, cache[1])\n",
    "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
    "        ### END CODE HERE ###\n",
    "    \n",
    "    return dA_prev, dW, db"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sigmoid:\n",
      "dA_prev = [[ 0.11017994  0.01105339]\n",
      " [ 0.09466817  0.00949723]\n",
      " [-0.05743092 -0.00576154]]\n",
      "dW = [[ 0.10266786  0.09778551 -0.01968084]]\n",
      "db = [[-0.05729622]]\n",
      "\n",
      "relu:\n",
      "dA_prev = [[ 0.44090989 -0.        ]\n",
      " [ 0.37883606 -0.        ]\n",
      " [-0.2298228   0.        ]]\n",
      "dW = [[ 0.44513824  0.37371418 -0.10478989]]\n",
      "db = [[-0.20837892]]\n"
     ]
    }
   ],
   "source": [
    "dAL, linear_activation_cache = linear_activation_backward_test_case()\n",
    "\n",
    "dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = \"sigmoid\")\n",
    "print (\"sigmoid:\")\n",
    "print (\"dA_prev = \"+ str(dA_prev))\n",
    "print (\"dW = \" + str(dW))\n",
    "print (\"db = \" + str(db) + \"\\n\")\n",
    "\n",
    "dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = \"relu\")\n",
    "print (\"relu:\")\n",
    "print (\"dA_prev = \"+ str(dA_prev))\n",
    "print (\"dW = \" + str(dW))\n",
    "print (\"db = \" + str(db))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output with sigmoid:**\n",
    "\n",
    "<table style=\"width:100%\">\n",
    "  <tr>\n",
    "    <td > dA_prev </td> \n",
    "           <td >[[ 0.11017994  0.01105339]\n",
    " [ 0.09466817  0.00949723]\n",
    " [-0.05743092 -0.00576154]] </td> \n",
    "\n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > dW </td> \n",
    "           <td > [[ 0.10266786  0.09778551 -0.01968084]] </td> \n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > db </td> \n",
    "           <td > [[-0.05729622]] </td> \n",
    "  </tr> \n",
    "</table>\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected output with relu:**\n",
    "\n",
    "<table style=\"width:100%\">\n",
    "  <tr>\n",
    "    <td > dA_prev </td> \n",
    "           <td > [[ 0.44090989  0.        ]\n",
    " [ 0.37883606  0.        ]\n",
    " [-0.2298228   0.        ]] </td> \n",
    "\n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > dW </td> \n",
    "           <td > [[ 0.44513824  0.37371418 -0.10478989]] </td> \n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > db </td> \n",
    "           <td > [[-0.20837892]] </td> \n",
    "  </tr> \n",
    "</table>\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.3 - L-Model Backward \n",
    "\n",
    "Now you will implement the backward function for the whole network. Recall that when you implemented the `L_model_forward` function, at each iteration, you stored a cache which contains (X,W,b, and z). In the back propagation module, you will use those variables to compute the gradients. Therefore, in the `L_model_backward` function, you will iterate through all the hidden layers backward, starting from layer $L$. On each step, you will use the cached values for layer $l$ to backpropagate through layer $l$. Figure 5 below shows the backward pass. \n",
    "\n",
    "\n",
    "<img src=\"images/mn_backward.png\" style=\"width:450px;height:300px;\">\n",
    "<caption><center>  **Figure 5** : Backward pass  </center></caption>\n",
    "\n",
    "** Initializing backpropagation**:\n",
    "To backpropagate through this network, we know that the output is, \n",
    "$A^{[L]} = \\sigma(Z^{[L]})$. Your code thus needs to compute `dAL` $= \\frac{\\partial \\mathcal{L}}{\\partial A^{[L]}}$.\n",
    "To do so, use this formula (derived using calculus which you don't need in-depth knowledge of):\n",
    "```python\n",
    "dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL\n",
    "```\n",
    "\n",
    "You can then use this post-activation gradient `dAL` to keep going backward. As seen in Figure 5, you can now feed in `dAL` into the LINEAR->SIGMOID backward function you implemented (which will use the cached values stored by the L_model_forward function). After that, you will have to use a `for` loop to iterate through all the other layers using the LINEAR->RELU backward function. You should store each dA, dW, and db in the grads dictionary. To do so, use this formula : \n",
    "\n",
    "$$grads[\"dW\" + str(l)] = dW^{[l]}\\tag{15} $$\n",
    "\n",
    "For example, for $l=3$ this would store $dW^{[l]}$ in `grads[\"dW3\"]`.\n",
    "\n",
    "**Exercise**: Implement backpropagation for the *[LINEAR->RELU] $\\times$ (L-1) -> LINEAR -> SIGMOID* model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: L_model_backward\n",
    "\n",
    "def L_model_backward(AL, Y, caches):\n",
    "    \"\"\"\n",
    "    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group\n",
    "    \n",
    "    Arguments:\n",
    "    AL -- probability vector, output of the forward propagation (L_model_forward())\n",
    "    Y -- true \"label\" vector (containing 0 if non-cat, 1 if cat)\n",
    "    caches -- list of caches containing:\n",
    "                every cache of linear_activation_forward() with \"relu\" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)\n",
    "                the cache of linear_activation_forward() with \"sigmoid\" (it's caches[L-1])\n",
    "    \n",
    "    Returns:\n",
    "    grads -- A dictionary with the gradients\n",
    "             grads[\"dA\" + str(l)] = ... \n",
    "             grads[\"dW\" + str(l)] = ...\n",
    "             grads[\"db\" + str(l)] = ... \n",
    "    \"\"\"\n",
    "    grads = {}\n",
    "    L = len(caches) # the number of layers\n",
    "    m = AL.shape[1]\n",
    "    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL\n",
    "    \n",
    "    # Initializing the backpropagation\n",
    "    ### START CODE HERE ### (1 line of code)\n",
    "    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: \"dAL, current_cache\". Outputs: \"grads[\"dAL-1\"], grads[\"dWL\"], grads[\"dbL\"]\n",
    "    ### START CODE HERE ### (approx. 2 lines)\n",
    "    current_cache = caches[L-1]\n",
    "    grads[\"dA\" + str(L-1)], grads[\"dW\" + str(L)], grads[\"db\" + str(L)] = linear_activation_backward(dAL, current_cache, activation = \"sigmoid\")\n",
    "    ### END CODE HERE ###\n",
    "    \n",
    "    # Loop from l=L-2 to l=0\n",
    "    for l in reversed(range(L-1)):\n",
    "        # lth layer: (RELU -> LINEAR) gradients.\n",
    "        # Inputs: \"grads[\"dA\" + str(l + 1)], current_cache\". Outputs: \"grads[\"dA\" + str(l)] , grads[\"dW\" + str(l + 1)] , grads[\"db\" + str(l + 1)] \n",
    "        ### START CODE HERE ### (approx. 5 lines)\n",
    "        current_cache = caches[l]\n",
    "        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[\"dA\" + str(l + 1)], current_cache, activation = \"relu\")\n",
    "        grads[\"dA\" + str(l)] = dA_prev_temp\n",
    "        grads[\"dW\" + str(l + 1)] = dW_temp\n",
    "        grads[\"db\" + str(l + 1)] = db_temp\n",
    "        ### END CODE HERE ###\n",
    "\n",
    "    return grads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "dW1 = [[ 0.41010002  0.07807203  0.13798444  0.10502167]\n",
      " [ 0.          0.          0.          0.        ]\n",
      " [ 0.05283652  0.01005865  0.01777766  0.0135308 ]]\n",
      "db1 = [[-0.22007063]\n",
      " [ 0.        ]\n",
      " [-0.02835349]]\n",
      "dA1 = [[ 0.12913162 -0.44014127]\n",
      " [-0.14175655  0.48317296]\n",
      " [ 0.01663708 -0.05670698]]\n"
     ]
    }
   ],
   "source": [
    "AL, Y_assess, caches = L_model_backward_test_case()\n",
    "grads = L_model_backward(AL, Y_assess, caches)\n",
    "print_grads(grads)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**\n",
    "\n",
    "<table style=\"width:60%\">\n",
    "  \n",
    "  <tr>\n",
    "    <td > dW1 </td> \n",
    "           <td > [[ 0.41010002  0.07807203  0.13798444  0.10502167]\n",
    " [ 0.          0.          0.          0.        ]\n",
    " [ 0.05283652  0.01005865  0.01777766  0.0135308 ]] </td> \n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > db1 </td> \n",
    "           <td > [[-0.22007063]\n",
    " [ 0.        ]\n",
    " [-0.02835349]] </td> \n",
    "  </tr> \n",
    "  \n",
    "  <tr>\n",
    "  <td > dA1 </td> \n",
    "           <td > [[ 0.12913162 -0.44014127]\n",
    " [-0.14175655  0.48317296]\n",
    " [ 0.01663708 -0.05670698]] </td> \n",
    "\n",
    "  </tr> \n",
    "</table>\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 6.4 - Update Parameters\n",
    "\n",
    "In this section you will update the parameters of the model, using gradient descent: \n",
    "\n",
    "$$ W^{[l]} = W^{[l]} - \\alpha \\text{ } dW^{[l]} \\tag{16}$$\n",
    "$$ b^{[l]} = b^{[l]} - \\alpha \\text{ } db^{[l]} \\tag{17}$$\n",
    "\n",
    "where $\\alpha$ is the learning rate. After computing the updated parameters, store them in the parameters dictionary. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Exercise**: Implement `update_parameters()` to update your parameters using gradient descent.\n",
    "\n",
    "**Instructions**:\n",
    "Update parameters using gradient descent on every $W^{[l]}$ and $b^{[l]}$ for $l = 1, 2, ..., L$. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# GRADED FUNCTION: update_parameters\n",
    "\n",
    "def update_parameters(parameters, grads, learning_rate):\n",
    "    \"\"\"\n",
    "    Update parameters using gradient descent\n",
    "    \n",
    "    Arguments:\n",
    "    parameters -- python dictionary containing your parameters \n",
    "    grads -- python dictionary containing your gradients, output of L_model_backward\n",
    "    \n",
    "    Returns:\n",
    "    parameters -- python dictionary containing your updated parameters \n",
    "                  parameters[\"W\" + str(l)] = ... \n",
    "                  parameters[\"b\" + str(l)] = ...\n",
    "    \"\"\"\n",
    "    \n",
    "    L = len(parameters) // 2 # number of layers in the neural network\n",
    "\n",
    "    # Update rule for each parameter. Use a for loop.\n",
    "    ### START CODE HERE ### (≈ 3 lines of code)\n",
    "    for l in range(L):\n",
    "        parameters[\"W\" + str(l+1)] = parameters[\"W\" + str(l+1)] - learning_rate * grads[\"dW\" + str(l + 1)]\n",
    "        parameters[\"b\" + str(l+1)] = parameters[\"b\" + str(l+1)] - learning_rate * grads[\"db\" + str(l + 1)]\n",
    "    ### END CODE HERE ###\n",
    "    return parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "W1 = [[-0.59562069 -0.09991781 -2.14584584  1.82662008]\n",
      " [-1.76569676 -0.80627147  0.51115557 -1.18258802]\n",
      " [-1.0535704  -0.86128581  0.68284052  2.20374577]]\n",
      "b1 = [[-0.04659241]\n",
      " [-1.28888275]\n",
      " [ 0.53405496]]\n",
      "W2 = [[-0.55569196  0.0354055   1.32964895]]\n",
      "b2 = [[-0.84610769]]\n"
     ]
    }
   ],
   "source": [
    "parameters, grads = update_parameters_test_case()\n",
    "parameters = update_parameters(parameters, grads, 0.1)\n",
    "\n",
    "print (\"W1 = \"+ str(parameters[\"W1\"]))\n",
    "print (\"b1 = \"+ str(parameters[\"b1\"]))\n",
    "print (\"W2 = \"+ str(parameters[\"W2\"]))\n",
    "print (\"b2 = \"+ str(parameters[\"b2\"]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Expected Output**:\n",
    "\n",
    "<table style=\"width:100%\"> \n",
    "    <tr>\n",
    "    <td > W1 </td> \n",
    "           <td > [[-0.59562069 -0.09991781 -2.14584584  1.82662008]\n",
    " [-1.76569676 -0.80627147  0.51115557 -1.18258802]\n",
    " [-1.0535704  -0.86128581  0.68284052  2.20374577]] </td> \n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > b1 </td> \n",
    "           <td > [[-0.04659241]\n",
    " [-1.28888275]\n",
    " [ 0.53405496]] </td> \n",
    "  </tr> \n",
    "  <tr>\n",
    "    <td > W2 </td> \n",
    "           <td > [[-0.55569196  0.0354055   1.32964895]]</td> \n",
    "  </tr> \n",
    "  \n",
    "    <tr>\n",
    "    <td > b2 </td> \n",
    "           <td > [[-0.84610769]] </td> \n",
    "  </tr> \n",
    "</table>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "## 7 - Conclusion\n",
    "\n",
    "Congrats on implementing all the functions required for building a deep neural network! \n",
    "\n",
    "We know it was a long assignment but going forward it will only get better. The next part of the assignment is easier. \n",
    "\n",
    "In the next assignment you will put all these together to build two models:\n",
    "- A two-layer neural network\n",
    "- An L-layer neural network\n",
    "\n",
    "You will in fact use these models to classify cat vs non-cat images!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "coursera": {
   "course_slug": "neural-networks-deep-learning",
   "graded_item_id": "c4HO0",
   "launcher_item_id": "lSYZM"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
