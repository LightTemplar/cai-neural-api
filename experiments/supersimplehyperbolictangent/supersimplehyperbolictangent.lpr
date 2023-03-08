program supersimplehyperbolictangent;
(*
super simple example with Pearson Correlation: learns boolean functions XOR,
AND and OR.
Copyright (C) 2019 Joao Paulo Schwarz Schuler

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
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  uconvolutionneuralnetwork,
  uvolume;

  type TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;
  type TBackOutput = array[0..3] of array[0..2] of TNeuralFloat;

  const inputs : TBackInput =
  ( // x1,   x2, one
    (-0.9, -0.9), // False, False
    (-0.9,  0.9), // False, True
    ( 0.9, -0.9), // True,  False
    ( 0.9,  0.9)  // True,  True
  );

  const outputs : TBackOutput =
  (// XOR, AND,   OR
    (-0.1,-0.1,-0.1),
    ( 0.1,-0.1, 0.1),
    ( 0.1,-0.1, 0.1),
    (-0.1, 0.1, 0.1)
  );

  procedure RunAlgo();
  var
    NN: TNNet;
    EpochCnt: integer;
    Cnt: integer;
    pOutPut: TNNetVolume;
    vInputs: TBackInput;
    vOutput: TBackOutput;
    NNListOutput: TNNetVolumeList;
    NNListFoundOutput: TNNetVolumeList;
    vXor, vAnd, vOr: TNNetVolume;
    vResult: TNNetVolume;
  begin
    NN := TNNet.Create();
    NNListOutput := TNNetVolumeList.Create();
    NNListFoundOutput := TNNetVolumeList.Create();

    vXor := TNNetVolume.Create;
    vAnd := TNNetVolume.Create;
    vOr := TNNetVolume.Create;
    vResult := TNNetVolume.Create;

    NN.AddLayer( TNNetInput.Create(2) );
    NN.AddLayer( TNNetFullConnect.Create(3) );
    NN.AddLayer( TNNetFullConnect.Create(3) );
    NN.SetLearningRate(0.1, 0.9);

    vInputs := inputs;
    vOutput := outputs;
    pOutPut := TNNetVolume.Create(3,1,1,1);

    NNListOutput.AddVolumes(4,1,1,3);
    NNListFoundOutput.AddVolumes(4,1,1,3);

    for Cnt := Low(inputs) to High(inputs) do
    begin
      NNListOutput[Cnt].Copy(vOutput[Cnt]);
    end;

    NNListOutput.GetColumn(vXor, 0);
    NNListOutput.GetColumn(vAnd, 1);
    NNListOutput.GetColumn(vOr,  2);

    for EpochCnt := 1 to 3000 do
    begin
      for Cnt := Low(inputs) to High(inputs) do
      begin
        NN.Compute(vInputs[Cnt]);
        NN.GetOutput(pOutPut);
        NN.Backpropagate(vOutput[Cnt]);
        if EpochCnt mod 300 = 0 then
        WriteLn
        (
          EpochCnt:7,'x',Cnt,
          ' Output:',
          pOutPut.Raw[0]:5:2,' ',
          pOutPut.Raw[1]:5:2,' ',
          pOutPut.Raw[2]:5:2,
          ' - Desired Output:',
          vOutput[Cnt][0]:5:2,' ',
          vOutput[Cnt][1]:5:2,' ' ,
          vOutput[Cnt][2]:5:2,' '
        );
      end;
      if EpochCnt mod 300 = 0 then WriteLn;
    end;

    for Cnt := Low(inputs) to High(inputs) do
    begin
      NN.Compute(vInputs[Cnt]);
      NN.GetOutput(pOutPut);
      NNListFoundOutput[Cnt].Copy(pOutPut);
    end;

    NNListFoundOutput.GetColumn(vResult, 0);
    WriteLn(' Xor Correlation: ', vResult.PearsonCorrelation(vXor):4:2);
    NNListFoundOutput.GetColumn(vResult, 1);
    WriteLn(' And Correlation: ', vResult.PearsonCorrelation(vAnd):4:2);
    NNListFoundOutput.GetColumn(vResult, 2);
    WriteLn(' Or  Correlation: ', vResult.PearsonCorrelation(vOr):4:2);
    WriteLn;
    vResult.Free;
    vXor.Free;
    vAnd.Free;
    vOr.Free;

    NNListFoundOutput.Free;
    NNListOutput.Free;
    NN.DebugWeights();
    NN.DebugErrors();
    pOutPut.Free;
    NN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Super Simple Example With Correlation';
  RunAlgo();
end.
