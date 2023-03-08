program LearnDigitalSumSub;
(*
LearnDigitalSumSub: learns how to sum bytes (X+Y) and subtract(X-Y) bytes.
Copyright (C) 2021 Joao Paulo Schwarz Schuler

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
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralab,
  CustApp;

type
  TTestDigitalLogic = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; var pInput, pOutput: array of byte);
  end;

  procedure TTestDigitalLogic.DoRun;
  var
    NN: TNNetForByteProcessing;
    TrainingCnt: integer;
    ErrorCnt: integer;
    BAInput,BAExpectedOutput,BAOutput: array of byte;
    InputSize, OutputSize: integer;
  begin
    NN := TNNetForByteProcessing.Create();

    InputSize := 2;
    OutputSize := 2;

    SetLength(BAInput, InputSize);
    SetLength(BAExpectedOutput, OutputSize);
    SetLength(BAOutput, OutputSize);

    NN.AddBasicByteProcessingLayers({InputByteCount=}InputSize,
     {OutputByteCount=}OutputSize, {FullyConnectedLayersCnt=}3, {NeuronsPerPath=}32);

    NN.DebugStructure();
    NN.SetLearningRate(0.001,0.0);

    WriteLn('Computing...');
    ErrorCnt := 0;
    for TrainingCnt := 0 to 3000000 do
    begin
      GetTrainingPair(TrainingCnt, {ThreadId=}0, BAInput, BAExpectedOutput);
      NN.Compute(BAInput);
      NN.GetOutput(BAOutput);
      NN.Backpropagate(BAExpectedOutput);
      ErrorCnt := ErrorCnt + ABCountDif(BAExpectedOutput, BAOutput);
      if TrainingCnt mod 10000 = 0 then
      begin
        WriteLn('Samples:', TrainingCnt, ' Error:', ErrorCnt);
        ErrorCnt := 0;
        //NN.DebugWeights();
      end;
    end;

    SetLength(BAInput, 0);
    SetLength(BAExpectedOutput, 0);
    SetLength(BAOutput, 0);
    NN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
    Terminate;
  end;

  procedure TTestDigitalLogic.GetTrainingPair(Idx: integer; ThreadId: integer;
    var pInput, pOutput: array of byte);
  begin
    pInput[0] := Random(128);
    pInput[1] := Random(128);
    pOutput[0] := pInput[0] + pInput[1];
    pOutput[1] := Byte(pInput[0] - pInput[1]);
    //multiplication
    //pOutput[0] := (pInput[0] * pInput[1]) div 256;
    //pOutput[1] := (pInput[0] * pInput[1]) mod 256;
  end;

var
  Application: TTestDigitalLogic;
begin
  Application := TTestDigitalLogic.Create(nil);
  Application.Title:='Logic Example';
  Application.Run;
  Application.Free;
end.

