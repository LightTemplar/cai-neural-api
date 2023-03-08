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

unit uvisualgantinyimagenettiled;

{$mode objfpc}{$H+}

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
  csGeneratedSize = 16;
  csGeneratorInputSize = 24;
  csSampleOutputDisplaySize = 128;
  csSampleInputDisplaySize = csSampleOutputDisplaySize + (csGeneratorInputSize - csGeneratedSize);

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboLearningRate: TComboBox;
    ComboComplexity: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabComplexity: TLabel;
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
    FGenerativeDisplay: THistoricalNets;
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
    function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    procedure GetDiscriminatorTrainingProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure DiscriminatorOnAfterEpoch(Sender: TObject);
    procedure DiscriminatorOnAfterStep(Sender: TObject);
    procedure DiscriminatorAugmentation(pInput: TNNetVolume; ThreadId: integer);
    procedure Learn(Sender: TObject);
    procedure SaveScreenshot(filename: string);
    procedure BuildTrainingPairs();
    procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
    procedure DisplaySample();
    procedure DiscriminatorOnStart(Sender: TObject);
    procedure SendStop;
    procedure CreateRandomInputSample(InputSample: TNNetVolume);
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
    {ImageCount=}8,
    {InputSize=}csSampleOutputDisplaySize, {displaySize=}csSampleOutputDisplaySize*2, {ImagesPerRow=}4
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
    //FDisplay.NormalizeMax(2);
    FDisplay.ForceMaxRange(2);
  end;

  //Debug only: FDisplay.PrintDebugChannel();

  FDisplay.NeuronalInputToRgbImg(color_encoding);

  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt]);
  aImage[FImageCnt].Width := csSampleOutputDisplaySize*2;
  aImage[FImageCnt].Height := csSampleOutputDisplaySize*2;
  ProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
end;

procedure TFormVisualLearning.DisplaySample();
var
  SampleInput: TNNetVolume;
  SampleOutput: TNNetVolume;
begin
  SampleInput := TNNetVolume.Create(csSampleInputDisplaySize, csSampleInputDisplaySize, FRandomDepth);
  SampleOutput := TNNetVolume.Create(csSampleOutputDisplaySize, csSampleOutputDisplaySize, FRandomDepth);
  SampleInput.Randomize();
  SampleInput.NormalizeMax(2);
  FGenerativeDisplay.CopyWeights(FGenerative);
  FGenerativeDisplay.Compute(SampleInput);
  FGenerativeDisplay.GetOutput(SampleOutput);
  DisplayInputImage(SampleOutput, FColorEncoding);
  SampleInput.Free;
  SampleOutput.Free;
end;

procedure TFormVisualLearning.DiscriminatorOnStart(Sender: TObject);
begin
  FGeneratives := TNNetDataParallelism.Create(FGenerative, FFit.MaxThreadNum);
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  BuildTrainingPairs();
end;

procedure TFormVisualLearning.SendStop;
begin
  WriteLn('Sending STOP request');
  FFit.ShouldQuit := true;
end;

procedure TFormVisualLearning.CreateRandomInputSample(InputSample: TNNetVolume);
begin
  InputSample.Resize(csGeneratorInputSize, csGeneratorInputSize, FRandomDepth);
  InputSample.Randomize();
  InputSample.NormalizeMax(2);
end;

function CreateGenerativeNetwork(InputSizeX, InputSizeY, InputDepth, NeuronMultiplier: integer): THistoricalNets;
var
  GenerativeBlocksCnt: integer;
begin
  Result := THistoricalNets.Create();
  Result.AddLayer([
    TNNetInput.Create(InputSizeX, InputSizeY, InputDepth)
  ]);
  for GenerativeBlocksCnt := 1 to ((csGeneratorInputSize - csGeneratedSize) div 2) do
  begin
    Result.AddLayer(TNNetConvolutionReLU.Create(32 * NeuronMultiplier, 3, 0, 1, 0));
  end;
  Result.AddDenseNetBlockCAI
  (
    8, {pNeurons=}16 * NeuronMultiplier, {supressBias=}0,
    {PointWiseConv=}TNNetConvolutionLinear,
    {IsSeparable=}true,
    {HasNorm=}true,
    {pBefore=}nil,
    {pAfter=}nil,
    {BottleNeck=}8,
    {Compression=}1, // Compression factor. 2 means taking half of channels.
    {DropoutRate=}0,
    {RandomBias=}0,
    {RandomAmplifier=}0
  );
  Result.AddCompression(0.75);
  Result.AddLayer([
    TNNetReLU.Create(),
    TNNetConvolutionLinear.Create(3, 1, 0, 1, 0),
    TNNetReLUL.Create(-40, +40)] // Protection against overflow
  );
end;

procedure TFormVisualLearning.Learn( Sender: TObject);
var
  NeuronMultiplier: integer;
begin
  FRandomSizeX := csGeneratorInputSize;
  FRandomSizeY := csGeneratorInputSize;
  FRandomDepth := StrToInt(ComboComplexity.Text);
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
  FBaseName := 'IMAGEART-TILED-'+IntToStr(FRandomDepth)+'-'+IntToStr(NeuronMultiplier)+'-';
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
    WriteLn('Tiny ImageNet 200 loaded file names: ', FTrainImages.CountElements());
    WriteLn('Loading Tiny ImageNet 200 images.');
    FTrainImages.LoadImages(FColorEncoding);
    WriteLn('Loaded.');
  end;
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  FLearningRateProportion := csLearningRates[ComboLearningRate.ItemIndex];
  writeln('Creating Neural Networks...');
  FDiscriminator := THistoricalNets.Create();
  FGenerativeDisplay := CreateGenerativeNetwork(csSampleInputDisplaySize, csSampleInputDisplaySize, FRandomDepth, NeuronMultiplier);

  if Not(FileExists(FBaseName+'generative.nn')) then
  begin
    WriteLn('Creating generative.');
    FGenerative := CreateGenerativeNetwork(FRandomSizeX, FRandomSizeY, FRandomDepth, NeuronMultiplier);
  end
  else
  begin
    FGenerative := THistoricalNets.Create();
    WriteLn('Loading generative.');
    FGenerative.LoadFromFile(FBaseName+'generative.nn');
  end;
  FGenerative.DebugStructure();
  FGenerative.SetLearningRate(0.001,0);
  FGenerative.SetL2Decay(0.0);

  if Not(FileExists(FBaseName+'discriminator.nn')) then
  begin
    WriteLn('Creating discriminator.');
    FDiscriminator.AddLayer(TNNetInput.Create(csGeneratedSize,csGeneratedSize,3));
    FDiscriminator.AddDenseNetBlockCAI
    (
            2, {pNeurons=}16 * NeuronMultiplier, {supressBias=}0,
            {PointWiseConv=}TNNetConvolutionLinear,
            {IsSeparable=}true,
            {HasNorm=}false,
            {pBefore=}nil,
            {pAfter=}nil,
            {BottleNeck=}8,
            {Compression=}1, // Compression factor. 2 means taking half of channels.
            {DropoutRate=}0,
            {RandomBias=}0,
            {RandomAmplifier=}0
    );
    FDiscriminator.AddCompression(0.75);
    FDiscriminator.AddLayer( TNNetMaxPool.Create(2) );
    FDiscriminator.AddDenseNetBlockCAI
    (
            4, {pNeurons=}16 * NeuronMultiplier, {supressBias=}0,
            {PointWiseConv=}TNNetConvolutionLinear,
            {IsSeparable=}true,
            {HasNorm=}false,
            {pBefore=}nil,
            {pAfter=}nil,
            {BottleNeck=}8,
            {Compression=}1, // Compression factor. 2 means taking half of channels.
            {DropoutRate=}0,
            {RandomBias=}0,
            {RandomAmplifier=}0
    );
    FDiscriminator.AddCompression(0.75);
    FDiscriminator.AddLayer( TNNetMaxPool.Create(2) );
    FDiscriminator.AddLayer([
        TNNetFullConnectReLU.Create(256),
        TNNetFullConnectLinear.Create(2),
        TNNetSoftMax.Create()
    ]);
    FDiscriminator.Layers[FDiscriminator.GetFirstImageNeuronalLayerIdx()].InitBasicPatterns();
  end
  else
  begin
    WriteLn('Loading discriminator.');
    FDiscriminator.LoadFromFile(FBaseName+'discriminator.nn');
    TNNetInput(FDiscriminator.Layers[0]).EnableErrorCollection;
    FDiscriminator.DebugStructure();
    FDiscriminator.DebugWeights();
  end;
  FDiscriminator.DebugStructure();

  FDiscriminatorClone := FDiscriminator.Clone();
  TNNetInput(FDiscriminatorClone.Layers[0]).EnableErrorCollection;

  FFit.EnableClassComparison();
  FFit.OnAfterEpoch := @Self.DiscriminatorOnAfterEpoch;
  FFit.OnAfterStep := @Self.DiscriminatorOnAfterStep;
  FFit.OnStart := @Self.DiscriminatorOnStart;
  FFit.DataAugmentationFn := @Self.DiscriminatorAugmentation;
  FFit.LearningRateDecay := 0.00001;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.001;
  FFit.Inertia := 0.0;
  FFit.FileNameBase := FBaseName+'GenerativeNeuralAPI';
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    FGenerative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //Debug only:
  FFit.MaxThreadNum := 2;
  FFit.FitLoading(FDiscriminator, {EpochSize=}64*1, 500, 500, {Batch=}16, 35000, @GetDiscriminatorTrainingProc, nil, nil); // This line does the same as above

  if Assigned(FGeneratives) then FreeAndNil(FGeneratives);
  FGenerativeDisplay.Free;
  FGenerative.Free;
  FDiscriminator.Free;
  FDiscriminatorClone.Free;
end;

function TFormVisualLearning.GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  RandomValue: integer;
  LocalPair: TNNetVolumePair;
  ClassId, ImageId: integer;
  CropX, CropY: integer;
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

  RandomValue := Random(1000);
  if RandomValue < 500 then
  begin
    LocalPair := FRealPairs[ThreadId];
    ClassId := FTrainImages.GetRandomClassId();
    ImageId := FTrainImages.List[ClassId].GetRandomIndex();
    CropX := Random(FTrainImages.List[ClassId].List[ImageId].SizeX - csGeneratorInputSize);
    CropY := Random(FTrainImages.List[ClassId].List[ImageId].SizeY - csGeneratorInputSize);
    LocalPair.I.CopyCropping(FTrainImages.List[ClassId].List[ImageId],CropX,CropY,csGeneratedSize,csGeneratedSize);
    LocalPair.I.Tag := 1;
    Result := LocalPair;
    Result.O.SetClassForSoftMax(1);
    if Result.I.Size <> csGeneratedSize*csGeneratedSize*3 then
    begin
      WriteLn('ERROR: Real Pair index ',ThreadId,' has wrong size:', Result.I.Size);
    end;
    // Debug Only:
    //if (Random(100)=0) then DisplayInputImage(Result.I, FColorEncoding);
  end
  else
  begin
    LocalPair := FGeneratedPairs[ThreadId];
    CreateRandomInputSample(LocalPair.I);
    FGeneratives[ThreadId].Compute(LocalPair.I);
    FGeneratives[ThreadId].GetOutput(LocalPair.I);
    LocalPair.I.Tag := 0;
    //LocalPair.I.PrintDebug();
    Result := LocalPair;
    Result.O.SetClassForSoftMax(0);
    if Result.I.Size <> csGeneratedSize*csGeneratedSize*3 then
    begin
      WriteLn('ERROR: Generated Pair has wrong size:', Result.I.Size);
    end;
  end;
end;

procedure TFormVisualLearning.GetDiscriminatorTrainingProc(Idx: integer;
  ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  LocalPair: TNNetVolumePair;
begin
  LocalPair := GetDiscriminatorTrainingPair(Idx, ThreadId);
  pInput.Copy(LocalPair.I);
  pOutput.Copy(LocalPair.O);
end;

procedure TFormVisualLearning.DiscriminatorOnAfterEpoch(Sender: TObject);
var
  LoopCnt, MaxLoop: integer;
  ExpectedDiscriminatorOutput, ExpectedGenerativeOutput,
  Transitory, DiscriminatorFound, GenerativeInput: TNNetVolume;
  Error: TNeuralFloat;
begin
  if (FFit.TrainingAccuracy <= 0.545) or FFit.ShouldQuit
  then exit;
  ExpectedDiscriminatorOutput := TNNetVolume.Create(2, 1, 1);
  ExpectedDiscriminatorOutput.SetClassForSoftMax(1);
  DiscriminatorFound := TNNetVolume.Create(ExpectedDiscriminatorOutput);
  Transitory := TNNetVolume.Create(FDiscriminatorClone.Layers[0].OutputError);
  GenerativeInput := TNNetVolume.Create(FRandomSizeX, FRandomSizeY, FRandomDepth);
  ExpectedGenerativeOutput := TNNetVolume.Create(Transitory);
  FDiscriminatorClone.CopyWeights(FDiscriminator);
//  WriteLn(FDiscriminator.GetWeightSum(),' ',FDiscriminatorClone.GetWeightSum());
//  WriteLn(FDiscriminator.GetBiasSum(),' ',FDiscriminatorClone.GetBiasSum());
  FGenerative.SetBatchUpdate(true);
  FGenerative.SetLearningRate(FFit.CurrentLearningRate*FLearningRateProportion, 0);
  FGenerative.SetL2Decay(0.00001);
  FDiscriminatorClone.SetBatchUpdate(true);
  FDiscriminatorClone.SetL2Decay(0.0);
  MaxLoop := 1000;
  if (FFit.TrainingAccuracy >= 0.945) then MaxLoop *= 3;
  // a code block
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
      FDiscriminatorClone.Compute(Transitory);
      FDiscriminatorClone.GetOutput(DiscriminatorFound);
      FDiscriminatorClone.Backpropagate(ExpectedDiscriminatorOutput);
      //Debug:
      (*WriteLn
      (
        DiscriminatorFound.FData[0]:5:3,' ',
        DiscriminatorFound.FData[1]:5:3,' ',
        ExpectedDiscriminatorOutput.FData[0]:5:3,' ',
        ExpectedDiscriminatorOutput.FData[1]:5:3,' '
      );*)
      Error += ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
      if (FDiscriminatorClone.Layers[0].OutputError.Size <> Transitory.Size) then
      begin
        WriteLn('Error - sizes don''t match:',FDiscriminatorClone.Layers[0].OutputError.Size,' ',Transitory.Size);
      end;
      Transitory.Sub(FDiscriminatorClone.Layers[0].OutputError);
      FGenerative.Backpropagate(Transitory);
      FGenerative.NormalizeMaxAbsoluteDelta(0.001);
      FGenerative.UpdateWeights();
      if LoopCnt mod 10 = 0 then ProcessMessages();
      if Random(1000) < 100 then
      begin
        if LoopCnt mod 100 = 1 then DisplaySample();
      end;
      if DiscriminatorFound.FData[1]>0.5 then break;
      if LoopCnt = 1 then WriteLn('Training generative starts at ',LoopCnt,' with: ', DiscriminatorFound.FData[1]:6:4);
    end;
  end;
  //Debug:
  //FDiscriminatorClone.Layers[0].OutputError.PrintDebug();WriteLn();
  //WriteLn('Generative error:', Error:6:4);
  //FGenerative.DebugErrors();
  //FGenerative.DebugWeights();
  //FDiscriminatorClone.DebugWeights();
  FGeneratives.CopyWeights(FGenerative);
  (*
  //Debug generatives:
  CreateRandomInputSample(GenerativeInput);
  FGenerative.Compute(GenerativeInput);
  FGenerative.GetOutput(Transitory);
  FGeneratives[0].Compute(GenerativeInput);
  FGeneratives[0].GetOutput(ExpectedGenerativeOutput);
  Error := ExpectedGenerativeOutput.SumDiff(Transitory);
  if Error > 0 then
  begin
    WriteLn('Generative ERROR: ', Error);
    FGenerative.DebugWeights();
    FGeneratives[0].DebugWeights();
    ReadLn();
    FGenerative.DebugStructure();
    FGeneratives[0].DebugStructure();
    ReadLn();
  end;
  //Debug discriminator:
  FDiscriminatorClone.Compute(Transitory);
  FDiscriminatorClone.GetOutput(DiscriminatorFound);
  FDiscriminator.Compute(Transitory);
  FDiscriminator.GetOutput(ExpectedDiscriminatorOutput);
  Error := ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
  if Error > 0 then
  begin
    WriteLn('Discriminator ERROR: ', Error);
    ReadLn();
  end;
  *)
  GenerativeInput.Free;
  ExpectedDiscriminatorOutput.Free;
  ExpectedGenerativeOutput.Free;
  Transitory.Free;
  if FFit.CurrentEpoch mod 100 = 0 then
  begin
    WriteLn('Saving ', FBaseName);
    FGenerative.SaveToFile(FBaseName+'generative.nn');
    FDiscriminator.SaveToFile(FBaseName+'discriminator.nn');
    SaveScreenshot(FBaseName+'cai-neural-gan.bmp');
  end;
  WriteLn('Training generative finishes at ',LoopCnt,' with: ', DiscriminatorFound.FData[1]:6:4);
  //  WriteLn(FGenerative.GetWeightSum(),' ',FGeneratives[1].GetWeightSum());
  //  WriteLn(FGenerative.GetBiasSum(),' ',FGeneratives[1].GetBiasSum());
  //DisplayInputImage(FRealPairs[Random(FRealPairs.Count)].I, FColorEncoding);
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
  //if Random(1000)>750 then pInput.MakeGray(FColorEncoding);
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
  GenerativeOutput := TNNetVolume.Create(csGeneratedSize, csGeneratedSize, 3);
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
