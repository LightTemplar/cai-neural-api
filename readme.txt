CAI - Conscious Artificial Intelligence
========================================
Developed by Joao Paulo Schwarz Schuler.

CAI project contains 2 main subprojects among other prototypes and APIs:
* CAI NEURAL API and
* TEasyLearnAndPredict neural network API.

The best place to find more about CAI NEURAL API is:
https://github.com/joaopauloschuler/neural-api/

These are the APIs:
* A convolutional neural network API implemented at lazarus/neural/neuralnetwork.pas.
* TEasyLearnAndPredict neural network API implemented at lazarus/neural/neuralbyteprediction.pas.
* An ultra fast Single Precision vector processing API supporting AVX and AVX2
  instructions at lazarus/neural/neuralvolume.pas.
* A generic evolutionary algorithm implemented at lazarus/neural/neuralevolutionary.pas.
* CIFAR-10 file support API implemented at lazarus/neural/neuraldatasets.pas.
* An easy to use OpenCL wrapper implemented at lazarus/neural/neuralopencl.pas.

These are some of the prototypes included in the project:
* CIFAR-10 classification examples:
  * lazarus/experiments/testcnnalgo/testcnnalgo.lpr
  * A number of CIFAR-10 classification examples at lazarus/experiments.
* Increase image resolution from 32x32 to 256x256 RGB at lazarus/experiments/IncreaseResolution.
* Web server that allows remote/distributed NN computing and backpropagation at lazarus/experiments/NeuralWebServer.
* Cellular Automatas:
  * John Horton Conway Game of Life.
  * Life Appearance – Cellular Automata showing self replication.
  * 3D Cellular automata sliced in 6 layers.
* Evolutionary Algorithm Example: Magic Square Maker.
* Minimax Algorithm Example: Nine Mans Morris.
* SOM Neural Network Example.
* OpenCL parallel computing example: Trillion Test.
* OpenCL wrapper example: Easy Trillion Test.
* OpenCL convolution implemented at:
  lazarus/experiments/visualCifar10OpenCL/

Available Videos:
Increasing Image Resolution with Neural Networks
https://www.youtube.com/watch?v=jdFixaZ2P4w

Ultra Fast Single Precision Floating Point Computing
https://www.youtube.com/watch?v=qGnfwpKUTIQ

Popperian Mining Agent Artificial Intelligence
https://www.youtube.com/watch?v=qH-IQgYy9zg

The most recent source code can be downloaded from:
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/

Copying
========
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Stable and Development Revisions
=================================
The currently stable revision is 928 and can be found at:
https://sourceforge.net/p/cai/svncode/928/tree/trunk/

The most recent development revision can be found at:
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/

Convolutional Neural Network
=============================
This unit was made to be easy to use and understand.
Implemented at neural/neuralnetwork.pas, this unit offers various layers.

Input Layer:
* TNNetInput (input/output: 1D, 2D or 3D).

Convolutional layers:
* TNNetConvolution (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetConvolutionReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetConvolutionLinear (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetPointwiseConvReLU (input/output: 1D, 2D or 3D).
* TNNetPointwiseConvLinear (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConv (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConvReLU (input/output: 1D, 2D or 3D).
* TNNetDepthwiseConvLinear (input/output: 1D, 2D or 3D).
* TNNet.AddSeparableConvReLU (input/output: 1D, 2D or 3D - separable convolution).
* TNNet.AddSeparableConvLinear (input/output: 1D, 2D or 3D - separable convolution).
* TNNet.AddConvOrSeparableConv (input/output: 1D, 2D or 3D). Adds a convolution or a separable convolutions with/without ReLU and normalization.

Fully connected layers:
* TNNetFullConnect (input/output: 1D, 2D or 3D).
* TNNetFullConnectReLU (input/output: 1D, 2D or 3D).
* TNNetFullConnectLinear (input/output: 1D, 2D or 3D).
* TNNetFullConnectSigmoid (input/output: 1D, 2D or 3D).

Locally connected layers:
* TNNetLocalConnect (input/output: 1D, 2D or 3D - feature size: 1D or 2D). Similar to full connect with individual neurons.
* TNNetLocalConnectReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).

Min / Max pools:
* TNNetAvgPool (input/output: 1D, 2D or 3D).
* TNNetMaxPool (input/output: 1D, 2D or 3D).
* TNNetMinPool (input/output: 1D, 2D or 3D).
* TNNet.AddMinMaxPool (input/output: 1D, 2D or 3D - min and max pools and then concatenates results).
* TNNet.AddAvgMaxPool (input/output: 1D, 2D or 3D - average and max pools and then concatenates results).

Min/Max layers that operate an entire channel and produce only one result per channel:
* TNNetAvgChannel (input: 2D or 3D - output: 1D). Calculates the channel average.
* TNNetMaxChannel (input: 2D or 3D - output: 1D). Calculates the channel max.
* TNNetMinChannel (input: 2D or 3D - output: 1D). Calculates the channel min.
* TNNet.AddMinMaxChannel (input/output: 1D, 2D or 3D - min and max channel and then concatenates results).
* TNNet.AddAvgMaxChannel (input/output: 1D, 2D or 3D - average and max channel and then concatenates results).

Trainable normalization layers allowing faster learning/convergence:
* TNNetChannelZeroCenter (input/output: 1D, 2D or 3D). Trainable zero centering.
* TNNetMovingStdNormalization (input/output: 1D, 2D or 3D). Trainable std. normalization.
* TNNetChannelStdNormalization (input/output: 1D, 2D or 3D). Trainable per channel std. normalization.
* TNNet.AddMovingNorm (input/output: 1D, 2D or 3D). Possible replacement for batch normalization.
* TNNet.AddChannelMovingNorm (input/output: 1D, 2D or 3D). Possible replacement for per batch normalization.

Non trainable and per sample normalization layers:
* TNNetLayerMaxNormalization (input/output: 1D, 2D or 3D).
* TNNetLayerStdNormalization (input/output: 1D, 2D or 3D).
* TNNetLocalResponseNorm2D (input/output: 2D or 3D).
* TNNetLocalResponseNormDepth (input/output: 2D or 3D).
* TNNetRandomMulAdd (input/output: 1D, 2D or 3D). Adds a random multiplication and random bias (shift).
* TNNetChannelRandomMulAdd (input/output: 1D, 2D or 3D). Adds a random multiplication and random bias (shift) per channel.

Concatenation, sum and reshaping layers:
* TNNetConcat (input/output: 1D, 2D or 3D). Allows concatenating the result from previous layers.
* TNNetDeepConcat (input/output: 1D, 2D or 3D). Concatenates into the Depth axis. This is useful with DenseNet like architectures.
* TNNetIdentity (input/output: 1D, 2D or 3D).
* TNNetIdentityWithoutBackprop (input/output: 1D, 2D or 3D). Allows the forward pass to proceed but prevents backpropagation.
* TNNetReshape (input/output: 1D, 2D or 3D).
* TNNetSplitChannels (input: 1D, 2D or 3D / output: 1D, 2D or 3D). Splits layers/channels from input.
* TNNetSum (input/output: 1D, 2D or 3D). Sums outputs from parallel layers allowing ResNet style networks.

Layers with activation functions and no trainable parameter:
* TNNetReLU (input/output: 1D, 2D or 3D).
* TNNetSigmoid (input/output: 1D, 2D or 3D).
* TNNetSoftMax (input/output: 1D, 2D or 3D).

Trainable bias (shift) and multiplication (scale) per cell or channel allowing faster learning and convergence:
* TNNetCellBias (input/output: 1D, 2D or 3D).
* TNNetCellMul (input/output: 1D, 2D or 3D).
* TNNetChannelBias (input/output: 1D, 2D or 3D).
* TNNetChannelMul (input/output: 1D, 2D or 3D).

There are also layers that do opposing operations. They do not share data with above layer types.
* TNNetDeLocalConnect (input/output: 1D, 2D or 3D - feature size: 1D or 2D). Similar to full connect with individual neurons.
* TNNetDeLocalConnectReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeconvolution (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeconvolutionReLU (input/output: 1D, 2D or 3D - feature size: 1D or 2D).
* TNNetDeMaxPool (input/output: 1D, 2D or 3D - max is done on a single layer).

These are the available weight initializers:
* InitUniform(Value: TNeuralFloat = 1).
* InitLeCunUniform(Value: TNeuralFloat = 1).
* InitHeUniform(Value: TNeuralFloat = 1).
* InitHeUniformDepthwise(Value: TNeuralFloat = 1).
* InitHeGaussian(Value: TNeuralFloat = 0.5).
* InitHeGaussianDepthwise(Value: TNeuralFloat = 0.5).
* InitGlorotBengioUniform(Value: TNeuralFloat = 1).

Ready to use data augmentation methods implemented at TVolume:
* procedure FlipX();
* procedure FlipY();
* procedure CopyCropping(Original: TVolume; StartX, StartY, pSizeX, pSizeY: integer);
* procedure CopyResizing(Original: TVolume; NewSizeX, NewSizeY: integer);
* procedure AddGaussianNoise(pMul: TNeuralFloat);
* procedure AddSaltAndPepper(pNum: integer; pSalt: integer = 2; pPepper: integer = -2);

The API allows you to create divergent/parallel and convergent layers as per
example below:

// The Neural Network
NN := TNNet.Create();

// Network that splits into 2 branches and then later concatenated
TA := NN.AddLayer(TNNetInput.Create(32, 32, 3));

// First branch starting from TA (5x5 features)
NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 5, 0, 0),TA);
NN.AddLayer(TNNetMaxPool.Create(2));
NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
NN.AddLayer(TNNetMaxPool.Create(2));
TB := NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));

// Another branch starting from TA (3x3 features)
NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 3, 0, 0),TA);
NN.AddLayer(TNNetMaxPool.Create(2));
NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
NN.AddLayer(TNNetMaxPool.Create(2));
TC := NN.AddLayer(TNNetConvolutionReLU.Create(64, 6, 0, 0));

// Concats both branches so the NN has only one end.
NN.AddLayer(TNNetConcat.Create([TB,TC]));
NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));
NN.AddLayer(TNNetLayerFullConnectReLU.Create(NumClasses));


TEasyLearnAndPredict Neural Network
====================================
The basic code of TEasyLearnAndPredict has been functional since the year of 2001.
This code was firstly developed as part of Joao's study at university.

TEasyLearnAndPredict is inspired in the combinatorial neural network.
The Combinatorial Neural Network concept is explained here:

A Connectionist Model for Knowledge Based Systems
http://dl.acm.org/citation.cfm?id=661467

TEasyLearnAndPredict started with inspiration from the Combinatorial NN described above.
At this moment, TEasyLearnAndPredict is NOT a Combinatorial neural network as
originally invented although it was inspired from.

In the case that you intend to reuse code, you should first look at the neural
network implemented at
libs/ubyteprediction.pas - Universal Byte Prediction Unit.

TEasyLearnAndPredict can be used to predict and/or classify data.

Follows some characteristics implemented in TEasyLearnAndPredict:
  * Neurons resemble microprocessor OPERATIONS and TESTS.
  * Neurons ARE NOT floating point operations. Therefore, this
    implementation doesn't benefit from GPU nor is intended to run on GPU.
  * TEasyLearnAndPredictClass is an easy to use class that you can embed in your
    own project. You can use it with small and large neural networks.
  * After learning, TEasyLearnAndPredictClass can predict/classify future states.
  * There are 2 neural network layers: tests and operations.
  * Some neurons are a "test" that relates a condition (current state) to an
    OPERATION. "tests" are the FIRST NEURAL NETWORK LAYER.
  * Some neurons are "operations" that transform the current state into the next
    predicted state. Operations compose the SECOND NEURAL NETWORK LAYER.
  * The current and the predicted states are arrays of bytes.
  * Neurons that correctly predict future states get stronger.
  * Stronger neurons win the prediction.
  * Neurons are born and killed at runtime! The number of neurons isn't static.
  * ubyteprediction has been tested under:
          Linux (amd64) / Lazarus
          Windows(amd64) / Lazarus
          Android(armv7a) / Laz4Android plus lamw
  * These platforms haven't been tested yet but they will probably work:
          MacOS / Lazarus
          Raspberry PI / Lazarus with armv7a Linux variants.

NEURON TYPES
=============
These are the types of available neurons at libs/uabfun:

// available operations. Some operations are logic/test operations such as <,> and <>.
// Other operations are math operations such as +,- and *.
const csNop = 0;     // no operation
      csEqual = 1;   // NextState[Base] := (Op1 = State[Op2]);
      csEqualM = 2;  // NextState[Base] := (State[Op1] = State[Op2]);
      csDifer = 3;   // NextState[Base] := (State[Op1] <> State[Op2]);
      csGreater = 4; // NextState[Base] := (State[Op1] > State[Op2]);
      csLesser = 5;  // NextState[Base] := (State[Op1] < State[Op2]);
      csTrue = 6;    // NextState[Base] := TRUE;
      csSet = 7;     // NextState[Base] := Op1;
      csInc = 8;     // NextState[Base] := State[Base] + 1;
      csDec = 9;     // NextState[Base] := State[Base] - 1;
      csAdd = 10;    // NextState[Base] := State[Op1] +   State[Op2];
      csSub = 11;    // NextState[Base] := State[Op1] -   State[Op2];
      csMul = 12;    // NextState[Base] := State[Op1] *   State[Op2];
      csDiv = 13;    // NextState[Base] := State[Op1] div State[Op2];
      csMod = 14;    // NextState[Base] := State[Op1] mod State[Op2];
      csAnd = 15;    // NextState[Base] := State[Op1] and State[Op2];
      csOr  = 16;    // NextState[Base] := State[Op1] or  State[Op2];
      csXor = 17;    // NextState[Base] := State[Op1] xor State[Op2];
      csInj = 18;    // NextState[Base] := State[Op1];
      csNot = 19;    // NextState[BASE] := not(PreviousState[BASE])

// An Operation type contains: an operation, 2 operands and boolean operand modifiers.
type
  TOperation = record
    OpCode:byte; //Operand Code
    Op1:integer; //Operand 1
    Op2:integer; //Operand 2
    RelativeOperandPosition1,         //Operand position is relative
    RelativeOperandPosition2:boolean;
    RunOnAction:boolean;
  end;

// "RelativeOperandPosition" Modifier Examples
// As an example, if RelativeOperandPosition1 is false, then we have
// NextState[Base] := State[Op1] + State[Op2];

// If RelativeOperandPosition1 is TRUE, then we have
// NextState[Base] := State[BASE + Op1] + State[Op2];

// If RunOnAction is TRUE and RelativeOperandPosition1 is FALSE, then we have:
// NextState[Base] := State[Op1] + Action[Op2];

// "RunOnAction" modifies first operator in Unary operations and
// modifies second operator in binary operations.

TEasyLearnAndPredict - INTRODUCTORY NEURAL NETWORK EXAMPLE:
============================================================
procedure trainingNeuralNetworkExample();
var
  FNeural:TEasyLearnAndPredictClass;
  aInternalState, aCurrentState, aPredictedState: array of byte;
  internalStateSize, stateSize: integer;
  secondNeuralNetworkLayerSize: integer;
  I,error_cnt: integer;
begin
  secondNeuralNetworkLayerSize := 1000; // 1000 neurons on second layer.

  internalStateSize :=  5; //  the internal state is composed by 5 bytes.
  stateSize         := 10; //  the current and next states are composed by 10 bytes.

  SetLength(aInternalState , internalStateSize);
  SetLength(aCurrentState  , stateSize        );
  SetLength(aPredictedState, stateSize        );

  FNeural.Initiate(internalStateSize, stateSize, false, secondNeuralNetworkLayerSize, {search size} 40, {use cache} false);

  // INCLUDE YOUR CODE HERE: some code here that updates the internal and current states.

  error_cnt := 0;
  for I := 1 to 10000 do
  begin
      // predicts the next state from aInternalState, aCurrentState into aPredictedState
      FNeural.Predict(aInternalState, aCurrentState, aPredictedState);

      // INCLUDE YOUR CODE HERE: some code here that updates the current state.

      // INCLUDE YOUR CODE HERE: some code here that compares aPredictedState
      // with new current state.

      // INCLUDE YOUR CODE HERE: if predicted and current states don't match,
      // then inc(error_cnt);

      // This method is responsible for training. You can use the same code for
      // training and actually predicting.
      FNeural.newStateFound(aCurrentState);
  end;

end;

SIMPLEST NEURAL NETWORK EXAMPLE:
=====================================
// In this example, the NN will learn how to count from 0 to 9 and restart.
procedure countTo9NeuralNetworkExample();
var
  FNeural:TEasyLearnAndPredictClass;
  aInternalState, aCurrentState, aPredictedState: array of byte;
  internalStateSize, stateSize: integer;
  secondNeuralNetworkLayerSize: integer;
  I,error_cnt: integer;
begin
  secondNeuralNetworkLayerSize := 1000; // 1000 neurons on second layer.

  internalStateSize :=  1; //  the internal state is composed by 1 byte.
  stateSize         :=  1; //  the current and next states are composed by 1 byte.

  SetLength(aInternalState , internalStateSize);
  SetLength(aCurrentState  , stateSize        );
  SetLength(aPredictedState, stateSize        );

  FNeural.Initiate(internalStateSize, stateSize, false, secondNeuralNetworkLayerSize, {search size} 40, {use cache} false);

  // INCLUDE YOUR CODE HERE: some code here that updates the internal and current states.
  ABClear(aInternalState);
  ABClear(aCurrentState);
  ABClear(aPredictedState);
  error_cnt := 0;

  writeln('Starting...');

  for I := 1 to 10000 do
  begin
      // predicts the next state from aInternalState, aCurrentState into aPredictedState
      FNeural.Predict(aInternalState, aCurrentState, aPredictedState);

      // INCLUDE YOUR CODE HERE: some code here that updates the current state.
      aCurrentState[0] := (aCurrentState[0] + 1) mod 10;

      // INCLUDE YOUR CODE HERE: some code here that compares aPredictedState
      // with new current state.
      if (not ABCmp(aCurrentState, aPredictedState))

      // INCLUDE YOUR CODE HERE: if predicted and current states don't match,
      then inc(error_cnt);

      // This method is responsible for training. You can use the same code for
      // training and actually predicting.
      FNeural.newStateFound(aCurrentState);
  end;

  // The smaller the number of errors, the faster the NN was able to learn.
  writeln('Finished. Errors found:',error_cnt);
end;

Let Me Know
============
In the case your are using this project or even small parts of it, I would love
to know about. Please post here:
https://sourceforge.net/p/cai/discussion/637501/thread/068972e4/?limit=25#903d
