{
unit uvisualgan
Copyright (C) 2018 Joao Paulo Schwarz Schuler

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
}

unit uvisualgantinyimagenetbirds2;

{$mode objfpc}{$H+}
//{$define MAKESMALL}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Menus, neuralnetwork, neuralvolumev, neuraldatasets, neuraldatasetsv,
  neuralvolume, MTProcs, math, neuralfit

  {$ifdef OpenCL}
  , neuralopencl
  {$endif}
  ;

const
  csLearningRates: array[0..2] of TNeuralFloat = (1, 0.1, 0.01);
  csGeneratorInputSize = 4;
  csMinDiscriminatorError = 0.1; // this prevents too small updates.

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboLearningRate: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabLearningRate: TLabel;
    RadLAB: TRadioButton;
    RadRGB: TRadioButton;
    procedure ButLearnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FDisplay: TNNetVolume;
    FRealPairs, FGeneratedPairs: TNNetVolumePairList;
    FImageCnt: integer;
    iEpochCount, iEpochCountAfterLoading: integer;
    FGenerative: THistoricalNets;
    FGeneratives: TNNetDataParallelism;
    FDiscriminator: THistoricalNets;
    FDiscriminatorClone: TNNet;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FBaseName: string;
    FColorEncoding: byte;
    FRandomSizeX, FRandomSizeY, FRandomDepth: integer;
    FLearningRateProportion: TNeuralFloat;
    {$ifdef OpenCL}
    FEasyOpenCL: TEasyOpenCL;
    FHasOpenCL: boolean;
    {$endif}

    FCritSec: TRTLCriticalSection;
    FFit: TNeuralDataLoadingFit;
    FTrainImages: TClassesAndElements;
    procedure CreateRandomInputSample(InputSample: TNNetVolume);
    function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    function GetDiscriminatorWarmUpPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    procedure GetDiscriminatorTrainingProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetDiscriminatorWarmUpProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure DiscriminatorOnAfterEpoch(Sender: TObject);
    procedure DiscriminatorOnAfterStep(Sender: TObject);
    procedure DiscriminatorAugmentation(pInput: TNNetVolume; ThreadId: integer);
    procedure Learn(Sender: TObject);
    procedure SaveScreenshot(filename: string);
    procedure BuildTrainingPairs();
    procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
    procedure DiscriminatorOnStart(Sender: TObject);
    procedure SendStop;
  public
    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType;


{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;

  if (FRunning) then
  begin
    SendStop;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    ChkBigNetwork.Enabled := false;
    Learn(Sender);
    ChkBigNetwork.Enabled := true;
    ButLearn.Caption := 'Restart';
    FRunning := false;
  end;
end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  SendStop;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  FRunning := false;
  FFit := TNeuralDataLoadingFit.Create();
  FTrainImages := TClassesAndElements.Create();
  InitCriticalSection(FCritSec);
  FGeneratives := nil;
  FRealPairs := TNNetVolumePairList.Create();
  FGeneratedPairs := TNNetVolumePairList.Create();
  FDisplay := TNNetVolume.Create();
  FImageCnt := 0;
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {ImageCount=}32,
    {InputSize=}64, {displaySize=}128, {ImagesPerRow=}8
  );
  {$ifdef OpenCL}
  FEasyOpenCL := TEasyOpenCL.Create();
  {$else}
  ChkRunOnGPU.Visible := false;
  {$endif}
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  SendStop;
  while FFit.Running do Application.ProcessMessages;
  while FRunning do Application.ProcessMessages;
  FreeNeuronImages(aImage, aLabelX, aLabelY);
  DoneCriticalSection(FCritSec);
  FRealPairs.Free;
  FGeneratedPairs.Free;
  FDisplay.Free;
  {$ifdef OpenCL}FEasyOpenCL.Free;{$endif}
  FFit.Free;
  FTrainImages.Free;
end;

procedure TFormVisualLearning.DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
var
  pMin0, pMax0: TNeuralFloat;
  pMin1, pMax1: TNeuralFloat;
  pMin2, pMax2: TNeuralFloat;
begin
  FDisplay.Resize(ImgInput);
  FDisplay.Copy(ImgInput);
  WriteLn('MinMax Image1: ',FDisplay.GetMin():3:2,' ',FDisplay.GetMax():3:2);
  // Makes Bipolar
  FDisplay.Mul(4);
  FDisplay.Sub(2);

  if color_encoding = csEncodeLAB then
  begin
    FDisplay.GetMinMaxAtDepth(0, pMin0, pMax0);
    FDisplay.GetMinMaxAtDepth(1, pMin1, pMax1);
    FDisplay.GetMinMaxAtDepth(2, pMin2, pMax2);
    pMax0 := Max(Abs(pMin0), Abs(pMax0));
    pMax1 := Max(Abs(pMin1), Abs(pMax1));
    pMax2 := Max(Abs(pMin2), Abs(pMax2));

    if pMax0 > 2 then
    begin
      FDisplay.MulAtDepth(0, 2/pMax0);
    end;

    if pMax1 > 2 then
    begin
      FDisplay.MulAtDepth(1, 2/pMax1);
    end;

    if pMax2 > 2 then
    begin
      FDisplay.MulAtDepth(2, 2/pMax2);
    end;
  end
  else if FDisplay.GetMaxAbs() > 2 then
  begin
    FDisplay.NormalizeMax(2);
  end;

  //Debug only: FDisplay.PrintDebugChannel();

  WriteLn('MinMax Image2: ',FDisplay.GetMin():3:2,' ',FDisplay.GetMax():3:2);
  FDisplay.NeuronalInputToRgbImg(color_encoding);
  WriteLn('MinMax Image3: ',FDisplay.GetMin():3:2,' ',FDisplay.GetMax():3:2);

  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt]);
  aImage[FImageCnt].Width := 128;
  aImage[FImageCnt].Height := 128;
  ProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
end;

procedure TFormVisualLearning.DiscriminatorOnStart(Sender: TObject);
begin
  if not(Assigned(FGeneratives)) then
  begin
    FGeneratives := TNNetDataParallelism.Create(FGenerative, FFit.MaxThreadNum);
    //FGeneratives[0].DebugStructure(); ReadLn;
    {$ifdef OpenCL}
    if FHasOpenCL then
    begin
      FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    end;
    {$endif}
  end;
  BuildTrainingPairs();
end;

procedure TFormVisualLearning.CreateRandomInputSample(InputSample: TNNetVolume);
var
  ClassId, ImageId: integer;
begin
  ClassId := FTrainImages.GetRandomClassId();
  ImageId := FTrainImages.List[ClassId].GetRandomIndex();
  InputSample.CopyResizing(FTrainImages.List[ClassId].List[ImageId],csGeneratorInputSize, csGeneratorInputSize);
  if Random(1000)>500 then InputSample.FlipX();
  // Adds a random shift per color channel [-0.1,+0.1]
  InputSample.AddAtDepth(0, (Random(1024)-512) / 5120 );
  InputSample.AddAtDepth(1, (Random(1024)-512) / 5120 );
  InputSample.AddAtDepth(2, (Random(1024)-512) / 5120 );
  //WriteLn(InputSample.GetMax():4:2,' ',InputSample.GetMin():4:2);
end;

procedure TFormVisualLearning.SendStop;
begin
  WriteLn('Sending STOP request');
  FFit.ShouldQuit := true;
end;

procedure TFormVisualLearning.Learn( Sender: TObject);
var
  NeuronMultiplier: integer;
  FirstBranch: TNNetLayer;
begin
  FRandomSizeX :=  csGeneratorInputSize;
  FRandomSizeY :=  csGeneratorInputSize;
  FRandomDepth := 3;
  {$ifdef OpenCL}
  FHasOpenCL := false;
  if ChkRunOnGPU.Checked then
  begin
    if FEasyOpenCL.GetPlatformCount() > 0 then
    begin
      FEasyOpenCL.SetCurrentPlatform(FEasyOpenCL.PlatformIds[0]);
      if FEasyOpenCL.GetDeviceCount() > 0 then
      begin
        FHasOpenCL := true;
      end;
    end;
  end;
  {$endif}
  if ChkBigNetwork.Checked
    then NeuronMultiplier := 2
    else NeuronMultiplier := 1;
  FBaseName := 'IMAGEART'+'-'+IntToStr(NeuronMultiplier)+'-';
  if RadRGB.Checked then
  begin
    FColorEncoding := csEncodeRGB;
    FBaseName += 'RGB-';
  end
  else
  begin
    FColorEncoding := csEncodeLAB;
    FBaseName += 'LAB-';
  end;
  Self.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 10;
  Self.Width  := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  ProcessMessages();
  if FTrainImages.Count = 0 then
  begin
    WriteLn('Loading Tiny ImageNet 200 file names.');
    FTrainImages.LoadFoldersAsClasses('tiny-imagenet-200/train/n02058221/');
    //FTrainImages.LoadFoldersAsClassesProportional('tiny-imagenet-200/train/','images', 0 , {$IFDEF MAKESMALL}0.1{$ELSE}1{$ENDIF});
    WriteLn('Tiny ImageNet 200 loaded file names: ', FTrainImages.CountElements());
    WriteLn('Loading Tiny ImageNet 200 images.');
    FTrainImages.LoadImages(FColorEncoding);
    FTrainImages.MakeMonopolar();
    WriteLn('Loaded.');
  end;
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  writeln('Creating Neural Networks...');
  FGenerative := THistoricalNets.Create();
  FDiscriminator := THistoricalNets.Create();

  FLearningRateProportion := csLearningRates[ComboLearningRate.ItemIndex];

  if Not(FileExists(FBaseName+IntToStr(FRandomDepth)+'-generativeb.nn')) then
  begin
    WriteLn('Creating generative.');
    FGenerative.AddLayer([
      TNNetInput.Create(FRandomSizeX, FRandomSizeY, FRandomDepth)
    ]);
    FGenerative.AddLayer([
      TNNetPad.Create(1),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,0,1,1), //4x4
      TNNetPad.Create(1),
      TNNetConvolution.Create(128 * NeuronMultiplier,3,0,1,1),
      TNNetDeMaxPool.Create(2, 1),
      TNNetPad.Create(2),
      TNNetConvolution.Create(64 * NeuronMultiplier,5,0,1,1), //8x8
      TNNetPad.Create(1),
      TNNetConvolution.Create(64 * NeuronMultiplier,3,0,1,1),
      TNNetDeMaxPool.Create(2, 1),
      TNNetPad.Create(2),
      TNNetConvolution.Create(32 * NeuronMultiplier,5,0,1,1), //16x16
      TNNetPad.Create(1),
      TNNetConvolution.Create(32 * NeuronMultiplier,3,0,1,1),
      TNNetDeMaxPool.Create(2, 1),
      TNNetPad.Create(2),
      TNNetConvolution.Create(32 * NeuronMultiplier,5,0,1,1), //32x32
      TNNetPad.Create(1),
      TNNetConvolution.Create(32 * NeuronMultiplier,3,0,1,1),
      TNNetDeMaxPool.Create(2, 1),
      TNNetPad.Create(2),
      TNNetConvolution.Create(32 * NeuronMultiplier,5,0,1,1), //64x64
      TNNetPad.Create(1),
      TNNetConvolution.Create(32 * NeuronMultiplier,3,0,1,1),
      TNNetConvolutionLinear.Create(3,1,0,1,0),
      TNNetCellBias.Create(),
      TNNetHyperbolicTangent.Create()
      //TNNetReLUL.Create(-40, +40) // Protection against overflow
    ]);
  end
  else
  begin
    WriteLn('Loading generative.');
    FGenerative.LoadFromFile(FBaseName+IntToStr(FRandomDepth)+'-generativeb.nn');
  end;
  FGenerative.DebugStructure();
  FGenerative.SetL2Decay(0.0);

  if Not(FileExists(FBaseName+'discriminatorb.nn')) then
  begin
    FirstBranch := FDiscriminator.AddLayer([
      TNNetInput.Create(64,64,3),
      TNNetPad.Create(1),
      TNNetConvolutionReLU.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}3,{SuppressBias=}1), // downsample to 22x22
      TNNetPad.Create(1),
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}3,{SuppressBias=}0), // downsample to 8x8
      TNNetPad.Create(1),
      TNNetConvolutionReLU.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}1), // downsample to 8x8
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}0), // downsample to 6x6
      TNNetPad.Create(1),
      TNNetConvolutionReLU.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}0), // downsample to 3x3
      TNNetPad.Create(1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}2,{SuppressBias=}1), // downsample to 2x2
      TNNetAvgChannel.Create()
    ]);

    FDiscriminator.Layers[FDiscriminator.GetFirstImageNeuronalLayerIdx()].InitBasicPatterns();

    FDiscriminator.AddLayerAfter([
      TNNetAvgPool.Create(4),
      TNNetPad.Create(1),
      TNNetConvolutionReLU.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}1),  // downsample to 8x8
      TNNetConvolutionReLU.Create({Features=}64  * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}2,{SuppressBias=}1), // downsample to 6x6
      TNNetConvolutionReLU.Create({Features=}64  * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}1), // downsample to 4x4
      TNNetConvolutionReLU.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}0,{Stride=}1,{SuppressBias=}1), // downsample to 2x2
      TNNetAvgChannel.Create()
    ], 0);

    FDiscriminator.AddLayer( TNNetConcat.Create([FDiscriminator.GetLastLayer(), FirstBranch]) );

    FDiscriminator.AddLayer([
      TNNetFullConnectReLU.Create(128),
      TNNetFullConnectLinear.Create(2)
    ]);
  end
  else
  begin
    WriteLn('Loading discriminator.');
    FDiscriminator.LoadFromFile(FBaseName+'discriminatorb.nn');
    TNNetInput(FDiscriminator.Layers[0]).EnableErrorCollection;
    FDiscriminator.DebugStructure();
    FDiscriminator.DebugWeights();
  end;
  FDiscriminator.DebugStructure();

  FDiscriminatorClone := FDiscriminator.Clone();
  TNNetInput(FDiscriminatorClone.Layers[0]).EnableErrorCollection;

  // Discriminator Warm Up
  FFit.DataAugmentationFn := @Self.DiscriminatorAugmentation;
  FFit.EnableClassComparison();
  FFit.LearningRateDecay := 0.0;
  FFit.L2Decay := 0.0;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.001;
  FFit.OnStart := @Self.DiscriminatorOnStart;
  FFit.OnAfterStep := @Self.DiscriminatorOnAfterStep;
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    FGenerative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //Debug only:
  //FFit.MaxThreadNum := 1;
  //{$IFNDEF MAKESMALL}
  //FFit.FitLoading(FDiscriminator, {EpochSize=}12800, 200, 200, {Batch=}64, {Epochs=}50, @GetDiscriminatorWarmUpProc, @GetDiscriminatorWarmUpProc, @GetDiscriminatorWarmUpProc); // This line does the same as above
  //{$ENDIF}

  // GAN TRAINING
  FFit.OnAfterEpoch := @Self.DiscriminatorOnAfterEpoch;
  FFit.FileNameBase := FBaseName+'GenerativeNeuralAPI';
  FFit.FitLoading(FDiscriminator, {EpochSize=}64*1, 500, 500, {Batch=}4, 35000, @GetDiscriminatorTrainingProc, nil, nil); // This line does the same as above

  if Assigned(FGeneratives) then FreeAndNil(FGeneratives);
  FGenerative.Free;
  FDiscriminator.Free;
  FDiscriminatorClone.Free;
end;

function TFormVisualLearning.GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  RandomValue: integer;
  LocalPair: TNNetVolumePair;
  ClassId, ImageId: integer;
begin
  RandomValue := Random(1000);
  if RandomValue < 600 then
  begin
    Result := GetDiscriminatorWarmUpPair(Idx, ThreadId);
  end
  else
  begin
    LocalPair := FGeneratedPairs[ThreadId];
    CreateRandomInputSample(LocalPair.I);
    if ThreadId > 0 then
    begin
      FGeneratives[ThreadId].Compute(LocalPair.I);
      FGeneratives[ThreadId].GetOutput(LocalPair.I);
    end
    else
    begin
      FGenerative.Compute(LocalPair.I);
      FGenerative.GetOutput(LocalPair.I);
    end;
    Result := LocalPair;
    Result.O.SetClassForSoftMax(0);
    if Result.I.Size <> 64*64*3 then
    begin
      WriteLn('ERROR: Generated Pair has wrong size:', Result.I.Size);
    end;
  end;
  // Debug Only:
  //if ((Random(100)=0) and (ThreadId=0)) then DisplayInputImage(Result.I, FColorEncoding);
end;

function TFormVisualLearning.GetDiscriminatorWarmUpPair(Idx: integer;
  ThreadId: integer): TNNetVolumePair;
var
  LocalPair: TNNetVolumePair;
  ClassId, ImageId: integer;
begin
  if (FRealPairs.Count = 0) then
  begin
    WriteLn('Error: discriminator real pairs have no element');
    Result := nil;
    exit;
  end;

  if FGeneratedPairs.Count = 0 then
  begin
    WriteLn('Error: discriminator generated/fake pairs have no element');
    Result := nil;
    exit;
  end;

  LocalPair := FRealPairs[ThreadId];
  ClassId := FTrainImages.GetRandomClassId();
  ImageId := FTrainImages.List[ClassId].GetRandomIndex();
  LocalPair.I.Copy(FTrainImages.List[ClassId].List[ImageId]);
  LocalPair.I.Tag := ClassId;
  Result := LocalPair;
  Result.O.SetClassForSoftMax(1);
  //WriteLn('GetDiscriminatorWarmUpPair:',LocalPair.I.GetMax():3:2);
end;

procedure TFormVisualLearning.GetDiscriminatorTrainingProc(Idx: integer;
  ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  LocalPair: TNNetVolumePair;
begin
  LocalPair := GetDiscriminatorTrainingPair(Idx, ThreadId);
  pInput.Copy(LocalPair.I);
  //WriteLn('GetDiscriminatorTrainingProc:',LocalPair.I.GetMax():3:2);
  pOutput.Copy(LocalPair.O);
end;

procedure TFormVisualLearning.GetDiscriminatorWarmUpProc(Idx: integer;
  ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  LocalPair: TNNetVolumePair;
begin
  LocalPair := GetDiscriminatorWarmUpPair(Idx, ThreadId);
  pInput.Copy(LocalPair.I);
  pOutput.Copy(LocalPair.O);
end;

procedure TFormVisualLearning.DiscriminatorOnAfterEpoch(Sender: TObject);
var
  LoopCnt, MaxLoop, MinCnt: integer;
  ExpectedDiscriminatorOutput, Transitory, DiscriminatorFound, GenerativeInput: TNNetVolume;
  Error, MaxError: TNeuralFloat;
  LocalDiscriminatorAccuracy: TNeuralFloat;
begin
  //DisplayInputImage(FRealPairs[Random(FRealPairs.Count)].I, FColorEncoding);
  MinCnt := 10 + Random(20);
  LocalDiscriminatorAccuracy := 0;
  if FFit.ShouldQuit then exit;
  ExpectedDiscriminatorOutput := TNNetVolume.Create(2, 1, 1);
  ExpectedDiscriminatorOutput.SetClassForSoftMax(1);
  DiscriminatorFound := TNNetVolume.Create(ExpectedDiscriminatorOutput);
  Transitory := TNNetVolume.Create(FDiscriminatorClone.Layers[0].OutputError);
  GenerativeInput := TNNetVolume.Create(FRandomSizeX, FRandomSizeY, FRandomDepth);
  FDiscriminatorClone.CopyWeights(FFit.ThreadNN[0]);
  FGenerative.SetBatchUpdate(true);
  FDiscriminatorClone.SetBatchUpdate(true);
  FGenerative.SetLearningRate(FFit.CurrentLearningRate*FLearningRateProportion, 0);
  FGenerative.SetL2Decay(0.0);
  FDiscriminatorClone.SetL2Decay(0.0);
  MaxLoop := 200+Random(400);
  //if (FFit.TrainingAccuracy >= 0.945) then MaxLoop *= 15;
  begin
    Error := 0;
    FDiscriminatorClone.RefreshDropoutMask();
    for LoopCnt := 1 to MaxLoop do
    begin
      if FFit.ShouldQuit then break;
      FDiscriminatorClone.ClearDeltas();
      FDiscriminatorClone.ClearInertia();
      CreateRandomInputSample(GenerativeInput);
      if FGenerative.Layers[0].Output.Size<>GenerativeInput.Size then
      begin
        Write('.');
        FGenerative.Layers[0].Output.ReSize(GenerativeInput);
        FGenerative.Layers[0].OutputError.ReSize(GenerativeInput);
      end;
      FGenerative.Compute(GenerativeInput);
      FGenerative.GetOutput(Transitory);
      //DisplayInputImage(Transitory, FColorEncoding);
      FDiscriminatorClone.Compute(Transitory);
      FDiscriminatorClone.GetOutput(DiscriminatorFound);
      //if (DiscriminatorFound.GetClass() <> 1) then
      begin
        FDiscriminatorClone.Backpropagate(ExpectedDiscriminatorOutput);
        Error += ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
        if (FDiscriminatorClone.Layers[0].OutputError.Size <> Transitory.Size) then
        begin
          WriteLn('Error - sizes don''t match:',FDiscriminatorClone.Layers[0].OutputError.Size,' ',Transitory.Size);
        end;
        MaxError := FDiscriminatorClone.Layers[0].OutputError.GetMaxAbs();
        // The next 2 lines make small updates impossible.
        if (MaxError>0) and (MaxError<csMinDiscriminatorError)
          then FDiscriminatorClone.Layers[0].OutputError.Mul(csMinDiscriminatorError/MaxError);
        Transitory.Sub(FDiscriminatorClone.Layers[0].OutputError);
        FGenerative.Backpropagate(Transitory);
        FGenerative.NormalizeMaxAbsoluteDelta(0.001);
        FGenerative.UpdateWeights();
      end;
      if LoopCnt mod 10 = 0 then ProcessMessages();
      if (*(Random(1000) < 200) and*) (LoopCnt mod 100 = 1) then
      begin
        DisplayInputImage(Transitory, FColorEncoding);
        if LoopCnt > 1 then WriteLn('Training generative continues at ',LoopCnt,' with: ', LocalDiscriminatorAccuracy:6:4);
      end;
      if LoopCnt = 1 then
      begin
        LocalDiscriminatorAccuracy := DiscriminatorFound.FData[1];
        WriteLn('Training generative starts at ',LoopCnt,' with: ', LocalDiscriminatorAccuracy:6:4,' at epoch ',FFit.CurrentEpoch);
      end;
      LocalDiscriminatorAccuracy := 0.9*LocalDiscriminatorAccuracy + 0.1*DiscriminatorFound.FData[1];
      if (LocalDiscriminatorAccuracy>0.5) and (LoopCnt > MinCnt) then break;
    end;
  end;
  if LocalDiscriminatorAccuracy > 0.95 then Randomize();
  if
    (FFit.TrainingAccuracy > 0.99) and
    (Random(10) = 0) then
  begin
    FGenerative.GetRandomLayer().InitDefault();
  end;
  //Debug:
  //FDiscriminatorClone.Layers[0].OutputError.PrintDebug();WriteLn();
  //WriteLn('Generative error:', Error:6:4);
  //FGenerative.DebugErrors();
  //FGenerative.DebugWeights();
  //FDiscriminatorClone.DebugWeights();
  FGeneratives.CopyWeights(FGenerative);
  {$IFDEF DEBUG}
  if FGenerative.GetWeightSum() <> FGeneratives[0].GetWeightSum() then
  begin
    WriteLn('Error at copying weights');
  end;
  if FDiscriminatorClone.GetWeightSum() <> FFit.ThreadNN[0].GetWeightSum() then
  begin
    WriteLn('Discriminator Clone and Threaded Discriminator aren''t the same.');
  end;
  {$ENDIF}
  GenerativeInput.Free;
  ExpectedDiscriminatorOutput.Free;
  Transitory.Free;
  if FFit.CurrentEpoch mod 100 = 0 then
  begin
    WriteLn('Saving ', FBaseName);
    FGenerative.SaveToFile(FBaseName+IntToStr(FRandomDepth)+'-generativeb.nn');
    FDiscriminator.SaveToFile(FBaseName+'discriminatorb.nn');
    SaveScreenshot(FBaseName+'cai-neural-gan.bmp');
  end;
  WriteLn('Training generative finishes at ',LoopCnt,
    ' with: ', LocalDiscriminatorAccuracy:6:4,
    ' Output: ', DiscriminatorFound.FData[0]:6:4,' ',DiscriminatorFound.FData[1]:6:4);
  DiscriminatorFound.Free;
end;

procedure TFormVisualLearning.DiscriminatorOnAfterStep(Sender: TObject);
begin
  LabClassRate.Caption := PadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+'%',4);
  ProcessMessages();
end;

procedure TFormVisualLearning.DiscriminatorAugmentation(pInput: TNNetVolume;
  ThreadId: integer);
begin
  if Random(1000)>500 then pInput.FlipX();
  if Random(1000)>500 then pInput.FlipY();
  pInput.AddAtDepth(0, (Random(1024)-512) / 5120 );
  pInput.AddAtDepth(1, (Random(1024)-512) / 5120 );
  pInput.AddAtDepth(2, (Random(1024)-512) / 5120 );
  pInput.Add( (Random(1024)-512) / 5120 );

  if Random(1000)>980 then pInput.MakeGray(FColorEncoding);
end;

procedure TFormVisualLearning.SaveScreenshot(filename: string);
begin
  try
    WriteLn(' Saving ',filename,'.');
    SaveHandleToBitmap(filename, Self.Handle);
  except
    // Nothing can be done if this fails.
  end;
end;

procedure TFormVisualLearning.BuildTrainingPairs();
var
  FakePairCnt, RealPairCnt: integer;
  DiscriminatorOutput, GenerativeOutput: TNNetVolume;
begin
  DiscriminatorOutput := TNNetVolume.Create(2, 1, 1);
  GenerativeOutput := TNNetVolume.Create(64, 64, 3);
  if FRealPairs.Count < FFit.MaxThreadNum then
  begin
    for RealPairCnt := 1 to FFit.MaxThreadNum do
    begin
      FRealPairs.Add
      (
        TNNetVolumePair.CreateCopying
        (
          GenerativeOutput,
          DiscriminatorOutput
        )
      );
    end;
  end;

  DiscriminatorOutput.SetClassForSoftMax(0);

  if FGeneratedPairs.Count < FFit.MaxThreadNum then
  begin
    for FakePairCnt := 1 to FFit.MaxThreadNum do
    begin
      FGeneratedPairs.Add
      (
        TNNetVolumePair.CreateCopying
        (
          GenerativeOutput,
          DiscriminatorOutput
        )
      );
    end;
  end;
  GenerativeOutput.Free;
  DiscriminatorOutput.Free;
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.
