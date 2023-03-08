(* ubackpropagation.
Copyright (C) 2017 Joao Paulo Schwarz Schuler

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
*)

unit ubackpropagation;
// You can use this unit for learning neural networks. Any real usage should
// be done with uconvolutionneuralnetwork unit instead.

(*
// Generic Backpropagation implemented by Joao Paulo Schwarz Schuler
// https://sourceforge.net/p/cai/

// Usage Example
// This example will train a 3x3 neural network.
// The 3x3 neural netowork learns boolean functions: XOR, AND, OR.
// Each layer is fully connected.

type TBackInputOutput = array[0..3] of array[0..2] of TBackNeuralFloat;

const outputs : TBackInputOutput =
  (// XOR,  AND,   OR
    (-0.9, -0.9, -0.9),
    ( 0.9, -0.9,  0.9),
    ( 0.9, -0.9,  0.9),
    (-0.9,  0.9,  0.9)
  );

const inputs : TBackInputOutput =
  ( // x1,   x2, bias
    (-0.9, -0.9,    1),
    (-0.9,  0.9,    1),
    ( 0.9, -0.9,    1),
    ( 0.9,  0.9,    1)
  );

procedure TForm1.BitBtn5Click(Sender: TObject);
var
  B: TBackPropagation;
  I: integer;
  Cnt: integer;
  pOutPut: array[0..3] of TBackNeuralFloat;
  vInputs, vOutput: TBackInputOutput;
begin
  vInputs := inputs;
  vOutput := outputs;

  B := TBackPropagation.Create(3 {neurons per layer}, 3{layers});
  B.Randomize();

  for I := 1 to 10000 do
  begin
    WriteLn();
    for Cnt := Low(inputs) to High(inputs) do
    begin
      B.Compute(vInputs[cnt], pOutPut);
      B.BackPropagate(vOutput[cnt]);
      WriteLn
      (
        I:7,'x',Cnt,' Output:',
        pOutPut[0]:5:2,' ',
        pOutPut[1]:5:2,' ',
        pOutPut[2]:5:2,' - Training data:',
        vOutput[cnt][0]:5:2,' ',
        vOutput[cnt][1]:5:2,' ' ,
        vOutput[cnt][2]:5:2,' '
      );
    end;
  end;
  B.Free;
end;
*)
{$mode objfpc}{$H+}
{$COPERATORS ON}
{$INLINE ON}

(*
{$FPUTYPE X87}
{$OPTIMIZATION REGVAR}
{$OPTIMIZATION LOOPUNROLL}
{$OPTIMIZATION CSE}
{$OPTIMIZATION ASMCSE}
*)

interface

uses
  Classes, SysUtils, uvolume;

type
  TBackWeights = array of TNeuralFloat;

  { TBackNeuron }
  TBackNeuron = object
    Weights: TBackWeights;
    Output: TNeuralFloat;
    Error: TNeuralFloat;
    ErrorDeriv: TNeuralFloat;
    FNumWeights: integer;

    // sets the number of weights
    procedure SetNumWeights(pNumWeights: integer);

    property NumWeights: integer read FNumWeights;
  end;

  { TBackInternalLayer }
  TBackInternalLayer = object
    Neurons: array of TBackNeuron;
    FSize: integer;

    procedure SetSize(pSize, NumWeights: integer);
    procedure Free;

    property Size: integer read FSize;
  end;

  { TBackPropagation }
  TBackPropagation = class(TObject)
    private
      FLayersCnt: integer;
      FActivationFn: TNeuralActivationFunction;
      FActivationFnDerivative: TNeuralActivationFunction;
      FLayers: array of TBackInternalLayer;
      FLearningRate: TNeuralFloat;

      procedure ComputeNeuron(LayerPos, NeuronPos: integer);
      procedure ComputeAll();
    public
      constructor Create(pNeuronsPerLayer, LayersCnt: integer);
      constructor Create(NeuronsPerLayer: array of integer);
      destructor Destroy; override;
      procedure Randomize();
      procedure Compute(var pInput: array of TNeuralFloat; var pOutput: array of TNeuralFloat);
      procedure BackPropagate(var pOutput: array of TNeuralFloat);

      property ActivationFn: TNeuralActivationFunction read FActivationFn write FActivationFn;
      property ActivationFnDerivative: TNeuralActivationFunction read FActivationFnDerivative write FActivationFnDerivative;
    published
      property LearningRate:TNeuralFloat read FLearningRate write FLearningRate;
  end;

implementation
uses math;
{ TBackInternalLayer }

procedure TBackInternalLayer.SetSize(pSize, NumWeights: integer);
var
  NeuronCnt: integer;
begin
  FSize := pSize;
  SetLength(Neurons,FSize);
  for NeuronCnt := Low(Neurons) to High(Neurons) do
  begin
    Neurons[NeuronCnt].SetNumWeights(NumWeights);
  end;
end;

procedure TBackInternalLayer.Free;
var
  NeuronCnt: integer;
begin
  for NeuronCnt := Low(Neurons) to High(Neurons) do
  begin
    Neurons[NeuronCnt].SetNumWeights(0);
  end;
  SetLength(Neurons,0);
  FSize := 0;
end;

{ TBackNeuron }

procedure TBackNeuron.SetNumWeights(pNumWeights: integer);
begin
  SetLength(Weights,pNumWeights);
  FNumWeights := pNumWeights;
end;

{ TBackPropagation }

procedure TBackPropagation.ComputeNeuron(LayerPos, NeuronPos: integer);
var
  WeightCnt, NumWeights: integer;
  Sum: TNeuralFloat;
begin
  Sum := 0;

  NumWeights := FLayers[LayerPos].Neurons[NeuronPos].NumWeights;

  for WeightCnt := 0 to NumWeights-1 do
  begin
    Sum := Sum +
      FLayers[LayerPos-1].Neurons[WeightCnt].Output *
      FLayers[LayerPos].Neurons[NeuronPos].Weights[WeightCnt];
    //WriteLn(LayerPos, 'x', NeuronPos,'x',WeightCnt,' ',FLayers[LayerPos].Neurons[NeuronPos].Weights[WeightCnt]:5:2);
  end;

  FLayers[LayerPos].Neurons[NeuronPos].Output :=
    FActivationFn(Sum);
end;

procedure TBackPropagation.ComputeAll;
var
  LayerCnt: integer;
  NeuronCnt: integer;
begin
  for LayerCnt := 1 to FLayersCnt-1 do
  begin
    for NeuronCnt := 0 to High(FLayers[LayerCnt].Neurons) do
    begin
      ComputeNeuron(LayerCnt, NeuronCnt);
    end;
  end;
end;

constructor TBackPropagation.Create(pNeuronsPerLayer, LayersCnt: integer);
var
  LayerCnt: integer;
begin
  inherited Create;
  FLayersCnt := LayersCnt;

  FLearningRate := 0.01;

  FActivationFn := @HiperbolicTangent;
  FActivationFnDerivative := @HiperbolicTangentDerivative;

  SetLength(FLayers,FLayersCnt);

  for LayerCnt := 0 to FLayersCnt-1 do
  begin
    FLayers[LayerCnt].SetSize(pNeuronsPerLayer,pNeuronsPerLayer);
  end;
end;

constructor TBackPropagation.Create(NeuronsPerLayer: array of integer);
var
  LayerCnt: integer;
begin
  Create(1,Length(NeuronsPerLayer));

  // define input layer
  FLayers[0].SetSize(NeuronsPerLayer[0], 0);

  // define other layers
  if High(NeuronsPerLayer) > 1 then
  for LayerCnt := 1 to High(NeuronsPerLayer) do
  begin
    FLayers[LayerCnt].SetSize(NeuronsPerLayer[LayerCnt],NeuronsPerLayer[LayerCnt-1]);
  end;
end;

destructor TBackPropagation.Destroy;
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to FLayersCnt-1 do
  begin
    FLayers[LayerCnt].Free;
  end;
  inherited Destroy;
end;

procedure TBackPropagation.Randomize;
var
  LayerCnt: integer;
  NeuronCnt: integer;
  WeightCnt: integer;
  NumWeights: integer;
begin
  for LayerCnt := 1 to FLayersCnt-1 do
  begin
    for NeuronCnt := 0 to High(FLayers[LayerCnt].Neurons) do
    begin
      NumWeights := FLayers[LayerCnt].Neurons[NeuronCnt].NumWeights;
      for WeightCnt := 0 to NumWeights-1 do
      begin
        FLayers[LayerCnt].Neurons[NeuronCnt].Weights[WeightCnt] :=
          (random(10000)-5000) / 5000;
      end;
    end;
  end;
end;

procedure TBackPropagation.Compute(var pInput: array of TNeuralFloat;
  var pOutput: array of TNeuralFloat);
var
  NeuronCnt: integer;
  MaxInput: integer;
  MaxOutput: integer;
begin
  MaxInput := Min(High(pInput), High(FLayers[0].Neurons) );

  // sets input values
  for NeuronCnt := 0 to MaxInput do
  begin
    FLayers[0].Neurons[NeuronCnt].Output := pInput[NeuronCnt];
  end;

  ComputeAll();

  MaxOutput := Min( High(pOutput), High(FLayers[FLayersCnt-1].Neurons) );

  for NeuronCnt := 0 to MaxOutput do
  begin
    pOutput[NeuronCnt] := FLayers[FLayersCnt-1].Neurons[NeuronCnt].Output;
  end;
end;

procedure TBackPropagation.BackPropagate(var pOutput: array of TNeuralFloat);
var
  LayerCnt: integer;
  NeuronCnt: integer;
  WeightCnt: integer;
  OutputCnt: integer;
  NumWeights: integer;
  Delta: TNeuralFloat;
  Error: TNeuralFloat;
  Weight: TNeuralFloat;
  ErrorDeriv: TNeuralFloat;
  LocalLearningErrorDeriv: TNeuralFloat;
  MaxOutput: integer;
begin
  MaxOutput := Min( High(pOutput), High(FLayers[FLayersCnt-1].Neurons) );

  // calculate output layer errors
  for NeuronCnt := 0 to MaxOutput do
  begin
    Error :=
      FLayers[FLayersCnt-1].Neurons[NeuronCnt].Output - pOutput[NeuronCnt];

    ErrorDeriv := Error *
      FActivationFnDerivative(FLayers[FLayersCnt-1].Neurons[NeuronCnt].Output);

    FLayers[FLayersCnt-1].Neurons[NeuronCnt].Error := Error;
    FLayers[FLayersCnt-1].Neurons[NeuronCnt].ErrorDeriv := ErrorDeriv;
  end;

  if FLayersCnt > 2 then
  // calculate intermediate layer errors
  for LayerCnt := FLayersCnt-2 downto 1 do
  begin
    for NeuronCnt := 0 to High(FLayers[LayerCnt].Neurons) do
    begin
      Error := 0;
      for OutputCnt := 0 to High(FLayers[LayerCnt+1].Neurons) do
      begin
        Error := Error +
          FLayers[LayerCnt+1].Neurons[OutputCnt].Weights[NeuronCnt] *
          FLayers[LayerCnt+1].Neurons[OutputCnt].ErrorDeriv;
      end;

      FLayers[LayerCnt].Neurons[NeuronCnt].Error := Error;

      ErrorDeriv := Error *
        FActivationFnDerivative(FLayers[LayerCnt].Neurons[NeuronCnt].Output);

      FLayers[LayerCnt].Neurons[NeuronCnt].ErrorDeriv := ErrorDeriv;
    end;
  end;

  // apply learning
  for LayerCnt := FLayersCnt-1 downto 1 do
  begin
    for NeuronCnt := 0 to High(FLayers[LayerCnt].Neurons) do
    begin
      LocalLearningErrorDeriv :=
        - FLearningRate *
        FLayers[LayerCnt].Neurons[NeuronCnt].ErrorDeriv;

      NumWeights := FLayers[LayerCnt].Neurons[NeuronCnt].NumWeights;
      for WeightCnt := 0 to NumWeights-1 do
      begin
        Delta :=
          LocalLearningErrorDeriv *
          FLayers[LayerCnt-1].Neurons[WeightCnt].Output;   // input from previous layer

        Weight :=
          FLayers[LayerCnt].Neurons[NeuronCnt].Weights[WeightCnt] + Delta;

        FLayers[LayerCnt].Neurons[NeuronCnt].Weights[WeightCnt] := Weight;
      end;
    end;
  end;
end;

end.

