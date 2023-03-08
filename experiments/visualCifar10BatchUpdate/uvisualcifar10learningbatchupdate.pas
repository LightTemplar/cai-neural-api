{
unit uvisualcifar10learningbatchupdate
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

unit uvisualcifar10learningbatchupdate;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume, MTProcs, math;

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ChkRestartAtEveryEpoch: TCheckBox;
    ChkRandomAtStart: TCheckBox;
    ChkLinearFirstConv: TCheckBox;
    ChkUseSeparableConv: TCheckBox;
    ChkMultipleSamplesAtValidation: TCheckBox;
    CheckReplaceMaxPool: TCheckBox;
    ChkCrop: TCheckBox;
    ChkStaircaseLearningRate: TCheckBox;
    ChkMovingNorm: TCheckBox;
    chkWeightAverage: TCheckBox;
    ComboAlgo: TComboBox;
    ComboAugmentation: TComboBox;
    ComboColor: TComboBox;
    ComboLastLayer: TComboBox;
    EdInertia: TEdit;
    EdFeatures: TEdit;
    EdFeatureSize: TEdit;
    EdFCLayers: TEdit;
    EdConvLayers: TEdit;
    EdDropout: TEdit;
    EdConvPadding: TEdit;
    EdInnerConvFeatureSize: TEdit;
    EdFCNeuronCount: TEdit;
    EdBatchSize: TEdit;
    EdInnerConvNeuronCount: TEdit;
    EdAutosaveName: TEdit;
    EdInputPadding: TEdit;
    EdAvgEpochs: TEdit;
    EdCLR: TEdit;
    EdClipDelta: TEdit;
    EdN: TEdit;
    EdM: TEdit;
    EdNoiseLevel: TEdit;
    EdLearnRateDecay: TEdit;
    EdPhysThreads: TEdit;
    EdThreadCount: TEdit;
    EdL2Decay: TEdit;
    EdMinLearnRate: TEdit;
    EdMaxEpochs: TEdit;
    EdStaircaseEpochs: TEdit;
    EdStride: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabAlgo: TLabel;
    Label1: TLabel;
    LabDataAugType: TLabel;
    LabColor: TLabel;
    LabCounts: TLabel;
    LabBatchSize: TLabel;
    LabComputedL2: TLabel;
    LabConvNeurons: TLabel;
    LabAlgoPar: TLabel;
    LabAutosaveName: TLabel;
    LabCLR: TLabel;
    LabClipDelta: TLabel;
    LabWAD: TLabel;
    LabInputPadding: TLabel;
    LabN: TLabel;
    LabM: TLabel;
    LabStaircase: TLabel;
    LabFCNCount: TLabel;
    LabInnerConvFeatureSize: TLabel;
    LabPadding: TLabel;
    LabNoiseLevel: TLabel;
    LabLearnDecay: TLabel;
    LabPhysThreads: TLabel;
    LabTotalTime: TLabel;
    LabL2Decay: TLabel;
    LabTime: TLabel;
    LabTestRate: TLabel;
    LabConv: TLabel;
    Labdropout: TLabel;
    LabClassRate: TLabel;
    LabLastLayer: TLabel;
    LabMinLearning: TLabel;
    LabMaxEpochs: TLabel;
    LabStride: TLabel;
    LabMaxPool: TLabel;
    LabHiddenNum: TLabel;
    LabFeatureSize: TLabel;
    LabNFeatures: TLabel;
    LabInertia: TLabel;
    LabLearning: TLabel;
    RadL2Conv: TRadioButton;
    RadL2All: TRadioButton;
    procedure ButLearnClick(Sender: TObject);
    procedure chkWeightAverageChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FImgCrop: boolean;
    FThreadNum, FStepSize: integer;
    FThreadNN: TNNetDataParallelism;
    FAvgWeights: TNNetDataParallelism;
    FAvgWeight: TNNet;
    FNoiseLevel: TNeuralFloat;
    FDataAugmentationType: integer;
    ImgVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    ImgWorkingVolumes: TNNetVolumeList;
    globalImgInput: TNNetVolume;
    globalClassHits: TNNetVolume;
    FFinishedThread: TNNetVolume;
    FConfusionMatrix: TNNetVolume;
    globalHit, globalMiss: integer;
    globalErrorSum: TNeuralFloat;
    globalTotalLoss: TNeuralFloat;
    color_encoding: integer;

    bIsSoftmax: boolean;
    bDataAugmentation: boolean;
    bLoadedFile: boolean;
    bMultipleSamplesAtValidation: boolean;

    iEpochCount, iEpochCountAfterLoading: integer;
    NN: THistoricalNets;

    FCritSec: TRTLCriticalSection;
    FMaxCrop: integer;
    procedure LoadCifar10binFilesIfRequired();
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
    procedure SaveScreenshot(filename: string);
    procedure SaveNeuronsImage(filename: string);
  public
    { public declarations }
    procedure RunNNThread(Index: PtrInt; Data: Pointer;
                        Item: TMultiThreadProcItem);

    procedure TestNNThread(Index: PtrInt; Data: Pointer;
                        Item: TMultiThreadProcItem);

    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType;

const
  NumClasses = 10;
  csHit = 1;
  csMiss = 0;
  csThreshold = 0.1;

procedure DecorrelateConcat(NN: TNNet; M,N:integer);
var
  PreviousLayer: TNNetLayer;
  Branch1, Branch2: TNNetLayer;
begin
  PreviousLayer := NN.GetLastLayer();
  NN.AddLayerAfter( TNNetPointwiseConvReLU.Create(M), PreviousLayer );
  Branch1 := NN.AddLayer( TNNetConvolutionReLU.Create(M, 3, 1{padding}, 1{stride}) );

  NN.AddLayerAfter( TNNetPointwiseConvReLU.Create(N), PreviousLayer );
  Branch2 := NN.AddLayer( TNNetConvolutionReLU.Create(N, 3, 1{padding}, 1{stride}) );

  NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
end;

procedure DecorrelateSum(NN: TNNet; M,N:integer);
var
  PreviousLayer: TNNetLayer;
  Branch1, Branch2: TNNetLayer;
begin
  PreviousLayer := NN.GetLastLayer();
  NN.AddLayerAfter( TNNetPointwiseConvReLU.Create(M), PreviousLayer );
  Branch1 := NN.AddLayer( TNNetConvolutionReLU.Create(M+N, 3, 1{padding}, 1{stride}) );

  NN.AddLayerAfter( TNNetPointwiseConvReLU.Create(N), PreviousLayer );
  Branch2 := NN.AddLayer( TNNetConvolutionReLU.Create(M+N, 3, 1{padding}, 1{stride}) );

  NN.AddLayer( TNNetSum.Create([Branch1, Branch2]) );
end;

procedure OutputToClassHits(ClassHits, Predicted, Desired: TNNetVolume; TH: TNeuralFloat = csThreshold);
var
  CntClasses, MaxClasses: integer;
begin
  MaxClasses := Predicted.Size - 1;
  for CntClasses := 0 to MaxClasses do
  begin
    if (Predicted.FData[CntClasses] > TH) then
    begin
      if (Desired.FData[CntClasses] > TH) then
      begin
        ClassHits.Add(0, CntClasses, csHit, 1) // This is a hit
      end
      else
      begin
        ClassHits.Add(0, CntClasses, csMiss, 1);// This is a miss
      end;
    end;
  end;
end;

procedure ClassHitsToFrequency(ClassHits, Frequencies: TNNetVolume);
var
  CntClasses, MaxClasses: integer;
begin
  Frequencies.Resize(1, ClassHits.SizeY, 1);
  Frequencies.Fill(0);
  MaxClasses := ClassHits.SizeY - 1;
  for CntClasses := 0 to MaxClasses do
  begin
    Frequencies[0, CntClasses, 0] := ClassHits[0, CntClasses, csHit] /
      (ClassHits[0, CntClasses, csMiss] + ClassHits[0, CntClasses, csHit] + 1);
  end;
end;

function GetClassFromPredictionAndFrequency(Predicted, Frequencies: TNNetVolume): integer;
var
  CntClasses, MaxClasses: integer;
  MaxValue, Mul: TNeuralFloat;
begin
  Result := -1;
  MaxValue := 0;
  MaxClasses := Predicted.Size - 1;
  for CntClasses := 0 to MaxClasses do
  begin
    if Predicted.FData[CntClasses] > csThreshold then
    begin
      Mul := Frequencies[0, CntClasses, 0] * Predicted.FData[CntClasses];
      if MaxValue < Mul then
      begin
        Result := CntClasses;
        MaxValue := Mul;
      end;
    end;
  end;
end;

procedure PrintFrequencies(ClassHits: TNNetVolume);
var
  CntClasses, MaxClasses: integer;
  ClassTotal: TNeuralFloat;
  TotalIdentified: TNeuralFloat;
begin
  TotalIdentified := 0;
  if ClassHits.Depth <> 2 then
  begin
    Write('Wrong class size:');
    ClassHits.PrintDebug();
  end
  else
  begin
    MaxClasses := ClassHits.SizeY - 1;
    for CntClasses := 0 to MaxClasses do
    begin
      ClassTotal := ClassHits[0,CntClasses,0] + ClassHits[0,CntClasses,1];
      TotalIdentified += ClassTotal;
      Write(' ',CntClasses,':');
      if ClassTotal > 0 then
      begin
        Write( (ClassHits[0,CntClasses,1]/ClassTotal):4:2,' (',Round(ClassTotal),')' );
      end;
    end;
    WriteLn(' Total:', Round(TotalIdentified));
  end;
end;

{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;

  if (FRunning) then
  begin
    FRunning := false;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    EnableComponents(false);
    Learn(Sender);
    EnableComponents(true);
    FRunning := false;
  end;
  ButLearn.Caption := 'Restart';
  LabClassRate.Caption := PadLeft('0%',4);
  LabTestRate.Caption  := PadLeft('0%',4);
  LabTime.Caption := '';
  LabTotalTime.Caption := '';
  LabCounts.Caption := '';
end;

procedure TFormVisualLearning.chkWeightAverageChange(Sender: TObject);
begin
  EdAvgEpochs.Enabled := chkWeightAverage.Checked;
end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  FRunning := false;
  LabTime.Caption := '';
  LabTotalTime.Caption := '';
  LabCounts.Caption := '';
  ImgVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  FFinishedThread := TNNetVolume.Create();
  FConfusionMatrix := TNNetVolume.Create(1,NumClasses,NumClasses);
  EdThreadCount.Text := IntToStr(ProcThreadPool.MaxThreadCount);
  EdPhysThreads.Text := IntToStr(ProcThreadPool.MaxThreadCount);
  //EdLearningRate.Text := FloatToStr(0.001 * TThread.ProcessorCount);
  globalImgInput := TNNetVolume.Create();
  globalClassHits := TNNetVolume.Create(1, NumClasses, 2);
  InitCriticalSection(FCritSec);
  FMaxCrop := 8;
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  globalClassHits.Free;
  FConfusionMatrix.Free;
  FFinishedThread.Free;
  DoneCriticalSection(FCritSec);
  globalImgInput.Free;
  ImgValidationVolumes.Free;
  ImgTestVolumes.Free;
  ImgVolumes.Free;
end;

procedure TFormVisualLearning.LoadCifar10binFilesIfRequired();
var
  I: integer;
begin
  if ImgVolumes.Count = 0 then
  begin
    // creates required volumes to store images
    for I := 0 to 39999 do
    begin
      ImgVolumes.Add( TNNetVolume.Create() );
    end;

    for I := 0 to 9999 do
    begin
      ImgTestVolumes.Add( TNNetVolume.Create() );
      ImgValidationVolumes.Add( TNNetVolume.Create() );
    end;
  end;
end;

procedure AddNorm(NN: TNNet);
begin
  NN.AddLayer( TNNetMovingStdNormalization.Create() );
  NN.AddLayer( TNNetChannelMul.Create() );
  NN.AddLayer( TNNetChannelBias.Create() );
end;

procedure AddBasicPathways(NN: TNNet; NumBranches, pLayers, pConvNeuronCount, pInnerConvFeatureSize, pPadding: integer);
var
  CntPathway: integer;
  CongregateLayer: TNNetLayer;
  CntLayers: integer;
  aL: array of TNNetLayer;
begin
  CongregateLayer := NN.GetLastLayer();
  SetLength(aL, NumBranches);
  for CntPathway := 1 to NumBranches do
  begin
    NN.AddLayerAfter( TNNetConvolutionReLU.Create(pConvNeuronCount,pInnerConvFeatureSize,pPadding,1), CongregateLayer);
    if pLayers > 1 then
    begin
      for CntLayers := 2 to pLayers do
      begin
        NN.AddLayer( TNNetConvolutionReLU.Create(pConvNeuronCount,pInnerConvFeatureSize,pPadding,0,0));
      end;
    end;
    NN.AddLayer( TNNetFullConnectReLU.Create(32) );
    NN.AddLayer( TNNetFullConnectReLU.Create(32) );
    // Creates only one output.
    aL[CntPathway - 1] := NN.AddLayer( TNNetFullConnectLinear.Create(1) );
  end;
  // Concats all outputs
  NN.AddLayer( TNNetConcat.Create(aL) );
end;

procedure AddDensePathways(NN: TNNet; NumBranches, pLayers, pConvNeuronCount, pInnerConvFeatureSize, pPadding: integer);
var
  CntPathway: integer;
  CongregateLayer: TNNetLayer;
  aL: array of TNNetLayer;
begin
  CongregateLayer := NN.GetLastLayer();
  SetLength(aL, NumBranches);
  for CntPathway := 1 to NumBranches do
  begin
    NN.AddLayerAfter( TNNetFullConnectReLU.Create(32), CongregateLayer);
    NN.AddLayer( TNNetFullConnectReLU.Create(32) );
    // Creates only one output.
    aL[CntPathway - 1] := NN.AddLayer( TNNetFullConnectLinear.Create(1) );
  end;
  // Concats all outputs
  NN.AddLayer( TNNetConcat.Create(aL) );
end;

procedure AddDenseBlockLike(NN: TNNet;
  pUnits, k: integer;
  BottleNeck: integer = 0;
  Compression: TNeuralFloat = 1);
var
  UnitCnt: integer;
  PreviousLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := NN.GetLastLayer();
      if BottleNeck > 0 then
      begin
        if PreviousLayer.Output.Depth > BottleNeck then
        begin
          NN.AddLayer( TNNetReLU.Create() );
          NN.AddLayer( TNNetPointwiseConvLinear.Create(BottleNeck, {featuresize}1) );
        end;
      end;
      NN.AddSeparableConvReLU(k, {featuresize}3, {padding}1, {stride}1);
      NN.AddMovingNorm(false);
      NN.AddLayer( TNNetDeepConcat.Create([PreviousLayer, NN.GetLastLayer()]) );
    end;
    if (Compression <> 1)
      then NN.AddLayer( TNNetPointwiseConvReLU.Create(Round(NN.GetLastLayer().Output.Depth * Compression ), {featuresize}1) );
  end;
end;

procedure TFormVisualLearning.Learn(Sender: TObject);
var
  aImage: array of TImage;
  aLabelX, aLabelY: array of TLabel;
  vDisplay: TNNetVolume;

  procedure ShowNeuronsIndependentChannels(firstNeuronalLayer, startImage, filterSize, color_encoding: integer);
  var
    NeuronCount: integer;
    MaxW0, MinW0, MaxW1, MinW1, MaxW2, MinW2: TNeuralFloat;
    AuxVolume: TNNetVolume;
  begin
    MaxW0 := 0;
    MinW0 := 0;
    MaxW1 := 0;
    MinW1 := 0;
    MaxW2 := 0;
    MinW2 := 0;

    AuxVolume := TNNetVolume.Create();

    NN.Layers[firstNeuronalLayer].GetMinMaxAtDepth(0, MinW0, MaxW0);

    if NN.Layers[firstNeuronalLayer].Neurons[0].Weights.Depth > 1 then
    begin
      NN.Layers[firstNeuronalLayer].GetMinMaxAtDepth(1, MinW1, MaxW1);
    end;

    if NN.Layers[firstNeuronalLayer].Neurons[0].Weights.Depth > 2 then
    begin
      NN.Layers[firstNeuronalLayer].GetMinMaxAtDepth(2, MinW2, MaxW2);
    end;

    for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do
    begin
      vDisplay.Copy(NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights);

      //AuxVolume.Copy(vDisplay);
      //vDisplay.CopyChannels(AuxVolume,[1,2]);
      //vDisplay.CopyChannels(AuxVolume,[0]);

      vDisplay.NeuronalWeightToImg3Channel(MaxW0, MinW0, MaxW1, MinW1, MaxW2, MinW2, color_encoding);

      LoadVolumeIntoTImage(vDisplay, aImage[NeuronCount + startImage], color_encoding);
      aImage[NeuronCount + startImage].Width := filterSize;
      aImage[NeuronCount + startImage].Height := filterSize;
    end;

    AuxVolume.Free;
    Application.ProcessMessages;
  end;

  procedure ShowNeurons(firstNeuronalLayer, startImage, filterSize, color_encoding: integer);
  var
    NeuronCount: integer;
    MaxW, MinW: TNeuralFloat;
  begin
    MaxW := NN.Layers[firstNeuronalLayer].GetMaxWeight();
    MinW := NN.Layers[firstNeuronalLayer].GetMinWeight();

    for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do
    begin
      vDisplay.Copy(NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights);

      vDisplay.NeuronalWeightToImg(MaxW, MinW, color_encoding);

      LoadVolumeIntoTImage(vDisplay, aImage[NeuronCount + startImage], color_encoding);
      aImage[NeuronCount + startImage].Width := filterSize;
      aImage[NeuronCount + startImage].Height := filterSize;
    end;
    Application.ProcessMessages;
  end;

  procedure DisplayInputImage(color_encoding: integer);
  begin
    vDisplay.Copy(globalImgInput);

    vDisplay.NeuronalInputToRgbImg(color_encoding);

    LoadVolumeIntoTImage(vDisplay, ImgSample);
    ImgSample.Width := 64;
    ImgSample.Height := 64;
  end;

  procedure CreateNeuronImages(firstNeuronalLayer, filterSize, imagesPerRow, NeuronNum: integer);
  var
    NeuronCount: integer;
    RowCount, ColCount: integer;
    RowNum, ColNum: integer;
    PosTop, PosLeft: integer;
    MaxTop, MaxLeft: integer;
  begin
    PosTop  := 14;
    PosLeft := 22;
    MaxTop  := 0;
    MaxLeft := 0;

    RowNum := NeuronNum div imagesPerRow;
    ColNum := imagesPerRow;

    if (NeuronNum mod imagesPerRow > 0) then
    begin
      Inc(RowNum);
    end;

    SetLength(aImage,  NeuronNum);
    SetLength(aLabelY, RowNum);
    SetLength(aLabelX, ColNum);

    for NeuronCount := 0 to NeuronNum - 1 do
    begin
      aImage[NeuronCount] := TImage.Create(FormVisualLearning);
      aImage[NeuronCount].Parent  := GrBoxNeurons;
      aImage[NeuronCount].Width   := NN.Layers[firstNeuronalLayer].Neurons[0].Weights.SizeX;
      aImage[NeuronCount].Height  := NN.Layers[firstNeuronalLayer].Neurons[0].Weights.SizeY;
      aImage[NeuronCount].Top     := (NeuronCount div imagesPerRow) * (filterSize+4) + PosTop;
      aImage[NeuronCount].Left    := (NeuronCount mod imagesPerRow) * (filterSize+4) + PosLeft;
      aImage[NeuronCount].Stretch := true;
      MaxTop                      := Max(MaxTop, aImage[NeuronCount].Top);
      MaxLeft                     := Max(MaxLeft, aImage[NeuronCount].Left);
    end;

    GrBoxNeurons.Height := MaxTop  + filterSize + 24;
    GrBoxNeurons.Width  := MaxLeft + filterSize + 10;

    Write('Creating cols and rows: ', ColNum, ', ', RowNum,'.');

    for ColCount := 0 to ColNum - 1 do
    begin
      aLabelX[ColCount] := TLabel.Create(FormVisualLearning);
      aLabelX[ColCount].Parent  := GrBoxNeurons;
      aLabelX[ColCount].Top     := (0)        * (filterSize+4) + PosTop - 14;
      aLabelX[ColCount].Left    := (ColCount) * (filterSize+4) + PosLeft;
      aLabelX[ColCount].Caption := Chr(Ord('A') + ColCount);
    end;

    for RowCount := 0 to RowNum - 1 do
    begin
      aLabelY[RowCount] := TLabel.Create(FormVisualLearning);
      aLabelY[RowCount].Parent  := GrBoxNeurons;
      aLabelY[RowCount].Top     := (RowCount) * (filterSize+4) + PosTop;
      aLabelY[RowCount].Left    := (0)        * (filterSize+4) + PosLeft - 16;
      aLabelY[RowCount].Caption := IntToStr(RowCount);
    end;

    WriteLn('Created.');
    Application.ProcessMessages;

  end;

  procedure FreeNeuronImages();
  var
    NeuronCount, RowCount, ColCount: integer;
  begin
    for NeuronCount := Low(aImage) to High(aImage) do
    begin
      aImage[NeuronCount].Free;
    end;

    for RowCount := Low(aLabelX) to High(aLabelX) do
    begin
      aLabelX[RowCount].Free;
    end;

    for ColCount := Low(aLabelY) to High(aLabelY) do
    begin
      aLabelY[ColCount].Free;
    end;

    SetLength(aImage, 0);
    SetLength(aLabelX, 0);
    SetLength(aLabelY, 0);
  end;

var
  MaxPool: integer;
  iConvNeuronCount: integer;
  iInnerConvFeatureSize: integer;
  fDropout: single;

  function AddMaxPool(pAfterLayer: TNNetLayer = nil): TNNetLayer;
  begin
    Result := pAfterLayer;
    if MaxPool > 0 then
    begin
      if fDropout > 0 then Result := NN.AddLayerAfter( TNNetDropout.Create(fDropout), Result );
      if CheckReplaceMaxPool.Checked then
      begin
        Result := NN.AddLayerAfter( TNNetConvolutionReLU.Create(NN.GetLastLayer().Output.Depth, MaxPool + 1, MaxPool div 2, MaxPool), Result );
      end
      else
      begin
        //Result := NN.AddAvgMaxPool(MaxPool, 0.5, true, Result);
        Result := NN.AddLayerAfter( TNNetMaxPool.Create(MaxPool), Result );
        //Result := NN.AddLayerAfter( TNNetMulLearning.Create(MaxPool*MaxPool), Result);
      end;
    end;
  end;

var
  fMinLearnRate, fNewLearningRate: single;
  fLearningRateDecay: single;
  fInitialLearningRate: single;
  iMaxEpochs, iStaircaseEpochs, iCLREpochLen: integer;
  CurrentLearningRate, fLearningRate, fInertia: single;

  procedure CheckLearningRate();
  var
    iStairCount: integer;
  begin
    if ChkStaircaseLearningRate.Checked then
    begin
      iStairCount := (iEpochCount mod iCLREpochLen) div iStaircaseEpochs;
      fNewLearningRate := (fInitialLearningRate * power(fLearningRateDecay,iStaircaseEpochs*iStairCount));
      if ( ( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate <> CurrentLearningRate) ) then
      begin
        CurrentLearningRate := fNewLearningRate;
        FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.ClearInertia();
        WriteLn
        (
          'Learning rate set to:',CurrentLearningRate:7:5
        );
      end;
    end
    else
    begin
      fNewLearningRate := (fInitialLearningRate * power(fLearningRateDecay,iEpochCount mod iCLREpochLen));
      if ( ( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate <> CurrentLearningRate) ) then
      begin
        CurrentLearningRate *= fLearningRateDecay;
        FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.ClearInertia();
        WriteLn
        (
          'Learning rate set to:',CurrentLearningRate:9:7
        );
      end;
    end;
  end;

var
  I: integer;
  pOutput, vOutput: TNNetVolume;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  CurrentRate: double;
  Stride: integer;
  FeaturesNum, FeatureSize: integer;
  InnerConvNum, InnerConvCnt: integer;
  InnerFCNum, InnerFCCnt: integer;
  fileName, fileNameBase, FileNameCSV, fileNameImage, fileNameNeuronsImage: string;
  fileNameConfusion: string;
  firstNeuronalLayer, secondNeuronalLayer, thirdNeuronalLayer: integer;
  iFCNeuronCount: integer;
  CVSError, CVSLoss, CVSRate,
  CVSTestError, CVSTestLoss, CVSTestRate: TNeuralFloat;
  CVSValidationError, CVSValidationLoss, CVSValidationRate: TNeuralFloat;
  CSVFile, CSVConfusion: TextFile;
  fL2Decay: TNeuralFloat;
  iAlgo: integer;
  iInputPadding, iPadding: integer;
  iInputDepth: integer;
  InputLayer, Branch1, Branch2, Branch3,
  Start1, Start2,
  NoBackProp1, NoBackProp2,
  BeforeMax1, BeforeMax2,
  PlusOne, EnhancedInput: TNNetLayer;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  MaxDelta, ClipDelta: TNeuralFloat;
  ValidationRecord: TNeuralFloat;
  EpochsForAverage: integer;
begin
  ValidationRecord := 0.0;
  AccuracyWithInertia := 10;
  FThreadNum := StrToInt(EdThreadCount.Text);
  FDataAugmentationType := ComboAugmentation.ItemIndex;
  FNoiseLevel := StrToFloat(EdNoiseLevel.Text);
  ProcThreadPool.MaxThreadCount := StrToInt(EdPhysThreads.Text);
  EpochsForAverage := StrToInt(EdAvgEpochs.Text);
  FStepSize := StrToInt(EdBatchSize.Text);
  ClipDelta := StrToFloat(EdClipDelta.Text);
  if ChkRandomAtStart.Checked then Randomize;

  iEpochCount := 0;
  iEpochCountAfterLoading := 0;
  bLoadedFile := false;
  bIsSoftmax  := false;
  CVSError    := 0;
  CVSLoss     := 0;
  CVSRate     := 0;
  CurrentRate := 0;
  CVSTestRate := 1;
  CVSTestLoss := 0;
  CVSTestError:= 0;
  CVSValidationRate := 0;
  CVSValidationRate := 0;
  CVSValidationError:= 0;

  color_encoding := ComboColor.ItemIndex;

  if color_encoding = csEncodeGray then
  begin
    iInputDepth := 1;
  end
  else
  begin
    iInputDepth := 3;
  end;

  iAlgo := ComboAlgo.ItemIndex;
  iInputPadding := StrToInt(EdInputPadding.Text);
  iPadding := StrToInt(EdConvPadding.Text);
  iInnerConvFeatureSize := StrToInt(EdInnerConvFeatureSize.Text);

  WriteLn('Number of threads:', FThreadNum);
  WriteLn('Algorithm:',iAlgo,' Color Encoding:', color_encoding,' Input Channels:', iInputDepth, ' Step Size:', FStepSize);

  fileNameBase :=
    EdAutosaveName.Text+'-'+
    'algo'+IntToStr(iAlgo)+'-'+
    EdLearningRate.Text+'-'+
    EdLearnRateDecay.Text+'-'+
    EdInertia.Text+'-'+
    EdL2Decay.Text+'-'+
    EdDropout.Text+'-'+
    EdFeatures.Text+'-'+
    EdFeatureSize.Text+'-'+
    EdStride.Text+'-'+
    EdInputPadding.Text+'-'+
    EdMaxPool.Text+'-'+
    EdInnerConvFeatureSize.Text+'-'+
    EdConvPadding.Text+'-'+
    EdConvLayers.Text+'-'+
    EdInnerConvNeuronCount.Text+'-'+
    EdFCLayers.Text+'-'+
    EdFCNeuronCount.Text+'-'+
    BoolToStr(ChkMovingNorm.Checked,'T','F')+'-'+
    BoolToStr(ChkCrop.Checked,'T','F')+'-'+
    BoolToStr(CheckReplaceMaxPool.Checked,'T','F')+'-'+
    BoolToStr(ChkUseSeparableConv.Checked,'T','F')+'-'+
    EdThreadCount.Text+'-'+
    EdBatchSize.Text+'-'+
    ComboAugmentation.Text+'-'+
    EdNoiseLevel.Text+'-'+
    ComboColor.Text+'-'+
    ComboLastLayer.Text;

  if ( (iAlgo=4) or (iAlgo=5) ) then
  begin
    fileNameBase += '-M'+EdM.Text+'N'+EdN.Text;
  end;

  WriteLn('File name is: ',fileNameBase);

  FileNameCSV := fileNameBase + '.csv';

  FileName := fileNameBase + '.nn';

  fileNameImage := fileNameBase + '.bmp';

  fileNameNeuronsImage := fileNameBase + '-neurons.bmp';
  fileNameConfusion := fileNameBase + '-confusion.csv';

  AssignFile(CSVFile, FileNameCSV);
  AssignFile(CSVConfusion, fileNameConfusion);

  LoadCifar10binFilesIfRequired();

  loadCifar10Dataset(ImgVolumes, 1, 0, color_encoding);
  loadCifar10Dataset(ImgVolumes, 2, 10000, color_encoding);
  loadCifar10Dataset(ImgVolumes, 3, 20000, color_encoding);
  loadCifar10Dataset(ImgVolumes, 4, 30000, color_encoding);
  loadCifar10Dataset(ImgValidationVolumes, 5, 0, color_encoding);
  loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0, color_encoding);
  TranslateCifar10VolumesToMachineAnimal(ImgVolumes);
  TranslateCifar10VolumesToMachineAnimal(ImgValidationVolumes);
  TranslateCifar10VolumesToMachineAnimal(ImgTestVolumes);


  WriteLn('Training Images:', ImgVolumes.Count, ' Test Images:', ImgTestVolumes.Count, ' Validation Images:', ImgValidationVolumes.Count);

  FeaturesNum := StrToInt(EdFeatures.Text);
  FeatureSize := StrToInt(EdFeatureSize.Text);
  FImgCrop := ChkCrop.Checked;
  fLearningRate := StrToFloat(EdLearningRate.Text);
  fInitialLearningRate := fLearningRate;
  fLearningRateDecay := 1 - StrToFloat(EdLearnRateDecay.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fLearningRate;
  fDropout := StrToFloat(EdDropout.Text);
  MaxPool := StrToInt(EdMaxPool.Text);
  Stride := StrToInt(EdStride.Text);
  iMaxEpochs := StrToInt(EdMaxEpochs.Text);
  iStaircaseEpochs := StrToInt(EdStaircaseEpochs.Text);
  iCLREpochLen := StrToInt(EdCLR.Text);
  iFCNeuronCount := StrToInt(EdFCNeuronCount.Text);
  iConvNeuronCount := StrToInt(EdInnerConvNeuronCount.Text);
  fMinLearnRate := StrToFloat(EdMinLearnRate.Text);
  fL2Decay := StrToFloat(EdL2Decay.Text);
  bMultipleSamplesAtValidation := ChkMultipleSamplesAtValidation.Checked;

  if (FStepSize > 1) and (fL2Decay>0) then
  begin
    fL2Decay := (1 - power( 1 - fL2Decay, FStepSize) );
    WriteLn('Actual L2:', fL2Decay:6:4);
  end;

  LabComputedL2.Caption := 'Computed L2: '+FloatToStrF(fL2Decay,ffGeneral,1,6)+' times learning rate.';

  bDataAugmentation := (ComboAugmentation.ItemIndex > 0);

  writeln('Creating Neural Network...');
  NN := THistoricalNets.Create();

  if ( FileExists(fileName) and FileExists(FileNameCSV) ) then
  begin
    writeln('Loading neural network from file: ',fileName);
    Reset(CSVFile);
    Reset(CSVConfusion);
    while not Eof(CSVFile) do
    begin
      ReadLn(CSVFile);
      Inc(iEpochCount);
    end;
    if (iEpochCount > 0) then iEpochCount := iEpochCount - 1;
    CloseFile(CSVFile);
    WriteLn('Precomputed epochs: ', iEpochCount);
    Append(CSVFile);
    Append(CSVConfusion);
    NN.LoadFromFile(fileName);
    // as this is a loaded NN, we'll start measuring and not learning
    NN.SetLearningRate(0, fInertia);
    if ChkStaircaseLearningRate.Checked then bLoadedFile := true;
  end
  else
  begin
    ReWrite(CSVFile);
    WriteLn(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');
    ReWrite(CSVConfusion);
    ConfusionWriteCSVHeader(CSVConfusion, csMachineAnimalCifar10Labels);
    if (FImgCrop) then
    begin
      InputLayer := NN.AddLayer( TNNetInput.Create(24,24,iInputDepth) );
    end
    else
    begin
      InputLayer := NN.AddLayer( TNNetInput.Create(32,32,iInputDepth) );
    end;

    if ( (iAlgo<>15) and (iAlgo<>14) and (iAlgo<>13) and (iAlgo<>8) and (iAlgo<>7) and (iAlgo<>6) and (iAlgo<>5)
      and (iAlgo<>4) and (iAlgo<>3) and (iAlgo<>2) ) then
    begin
      NN.AddConvOrSeparableConv({Separable=}false, {HasRelu=}Not(ChkLinearFirstConv.Checked),
        ChkMovingNorm.Checked, FeaturesNum, FeatureSize,
        iInputPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
        {RandomBias=}1, {RandomAmplifier=}1);
      NN.Layers[ NN.GetFirstImageNeuronalLayerIdx() ].InitBasicPatterns();
    end;

    InnerConvNum := StrToInt(EdConvLayers.Text);
    InnerFCNum := StrToInt(EdFCLayers.Text);

      case iAlgo of
      0:
      begin
        AddMaxPool();

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
              ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
              iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
              {RandomBias=}1, {RandomAmplifier=}1);
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      1:
      begin
        // Very similar to LeCun LeNet5
        // ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(6, 5, 0, 1));
        NN.AddLayer( TNNetMaxPool.Create(2) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
        NN.AddLayer( TNNetConvolution.Create(16, 5, 0, 1) );
        NN.AddLayer( TNNetMaxPool.Create(2) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
        NN.AddLayer( TNNetFullConnect.Create(120) );
        NN.AddLayer( TNNetFullConnect.Create(84) );
      end;
      2:
      begin
        // Inspired on Keras CIFAR-10 example
        // https://keras.io/examples/cifar10_cnn/
        NN.AddLayer( TNNetConvolutionReLU.Create({neurons}32, {featuresize}3, {padding}1, {stride}1) );
        if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
        if ChkUseSeparableConv.Checked then
        begin
          NN.AddSeparableConvReLU({neurons}32, {featuresize}3, {padding}1, {stride}1);
          NN.AddLayer( TNNetMaxPool.Create(2) );
          if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
          NN.AddSeparableConvReLU({neurons}64, {featuresize}3, {padding}1, {stride}1);
          if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
          NN.AddSeparableConvReLU({neurons}64, {featuresize}3, {padding}1, {stride}1);
        end
        else
        begin
          NN.AddLayer( TNNetConvolutionReLU.Create({neurons}32, {featuresize}3, {padding}1, {stride}1) );
          NN.AddLayer( TNNetMaxPool.Create(2) );
          if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
          NN.AddLayer( TNNetConvolutionReLU.Create({neurons}64, {featuresize}3, {padding}1, {stride}1) );
          if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
          NN.AddLayer( TNNetConvolutionReLU.Create({neurons}64, {featuresize}3, {padding}1, {stride}1) );
        end;
        NN.AddLayer( TNNetMaxPool.Create(2) );
        NN.AddLayer( TNNetFullConnectReLU.Create({neurons}512) );
      end;
      3:
      begin
        // INSPIRED ON TENSOR FLOW EXAMPLE
        // https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
        NN.AddLayer( TNNetConvolutionReLU.Create({neurons}64, {featuresize}5, {padding}2, {stride}1) );
        NN.AddLayer( TNNetMaxPool.Create({size}3,{stride}2) );
        if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
        if ChkUseSeparableConv.Checked then
        begin
          NN.AddSeparableConvReLU({neurons}64, {featuresize}5, {padding}2, {stride}1);
        end
        else
        begin
          NN.AddLayer( TNNetLocalResponseNormDepth.Create({diameter}9) );
          NN.AddLayer( TNNetConvolutionReLU.Create({neurons}64, {featuresize}5, {padding}2, {stride}1) );
          NN.AddLayer( TNNetLocalResponseNormDepth.Create({diameter}9) );
        end;
        NN.AddLayer( TNNetMaxPool.Create({size}3,{stride}2) );
        NN.AddLayer( TNNetFullConnectReLU.Create({neurons}384) );
        NN.AddLayer( TNNetFullConnectReLU.Create({neurons}192) );
      end;
      4:
      begin
        // Branch1
        if StrToInt(EdM.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
          if ChkLinearFirstConv.Checked
            then NN.AddLayer( TNNetConvolutionLinear.Create(StrToInt(EdM.Text), FeatureSize, iInputPadding, Stride) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(  StrToInt(EdM.Text), FeatureSize, iInputPadding, Stride) );
          Branch1 := AddMaxPool();
        end;

        // Branch2
        if StrToInt(EdN.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
          if ChkLinearFirstConv.Checked
            then NN.AddLayer( TNNetConvolutionLinear.Create(StrToInt(EdN.Text), FeatureSize, iInputPadding, Stride) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(  StrToInt(EdN.Text), FeatureSize, iInputPadding, Stride) );
          Branch2 := AddMaxPool();
        end;

        if ( (StrToInt(EdM.Text) > 0) and (StrToInt(EdN.Text) > 0) ) then
        begin
          NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
        end;

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
            if ChkUseSeparableConv.Checked
              then NN.AddSeparableConvReLU(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 1)
              else NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      5:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );
        Branch1 := AddMaxPool();

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );
        Branch2 := AddMaxPool();

        NN.AddLayer( TNNetSum.Create([Branch1, Branch2]) );

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
            if ChkUseSeparableConv.Checked
              then NN.AddSeparableConvReLU(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 1)
              else NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      6:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch1 := NN.AddLayer(TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0));

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch2 := NN.AddLayer(TNNetConvolutionReLU.Create(22, iInnerConvFeatureSize, iPadding, 0));

        // Branch3
        NN.AddLayerAfter( TNNetSplitChannels.Create([2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch3 := NN.AddLayer( TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2, Branch3]) );

        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetLayerStdNormalization.Create() );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
      end;
      7:
      begin
        // Branch1 (default is 3x3 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize-2, iInputPadding + 0, Stride),InputLayer);
        Branch1 := AddMaxPool();

        // Branch2 (default is 5x5 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding + 1, Stride),InputLayer);
        Branch2 := AddMaxPool();

        // Branch3 (default is 7x7 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize+2, iInputPadding + 2, Stride),InputLayer);
        Branch3 := AddMaxPool();

        // Concats both branches so the NN has only one end.
        NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2, Branch3]));
        //NN.AddLayer(TNNetSum.Create([Branch1, Branch2, Branch3]));

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
            if ChkUseSeparableConv.Checked
              then NN.AddSeparableConvReLU(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 1)
              else NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      8:
      begin
        InputLayer := NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, iInputPadding, Stride) );

        // Branch1
        Branch1 := AddMaxPool(InputLayer);

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
            if ChkUseSeparableConv.Checked
              then Branch1 := NN.AddSeparableConvReLU(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 1)
              else Branch1 := NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            Branch1 := NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;

        // Branch2
        Branch2 := AddMaxPool(InputLayer);

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if ChkMovingNorm.Checked then NN.AddMovingNorm(true);
            if ChkUseSeparableConv.Checked
              then Branch2 := NN.AddSeparableConvReLU(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 1)
              else Branch2 := NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0, 0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            Branch2 := NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;

        // Concats both branches so the NN has only one end.
        NN.AddLayer(TNNetSum.Create([Branch1, Branch2]));
      end;
      9:
      begin
        AddMaxPool();

        if (InnerConvNum div 2 > 0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum div 2 do
          begin
            NN.AddResNetUnit(iConvNeuronCount);
          end;
        end;

        if (InnerFCNum > 0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
        //NN.AddLayer( TNNetAvgChannel.Create() );
      end;
      10:
      begin
        AddMaxPool();

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0,0) );
            if InnerConvCnt < InnerConvNum
              then NN.AddLayer( TNNetMaxPool.Create(2) )
              else NN.AddLayer( TNNetMaxChannel.Create() );
          end;
        end;
      end;
      11:
      begin
        AddMaxPool();

        if (InnerConvNum div 2 > 0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            DecorrelateConcat(NN, StrToInt(EdM.Text), StrToInt(EdN.Text));
          end;
        end;

        if (InnerFCNum > 0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      12:
      begin
        AddMaxPool();
        if (InnerConvNum div 2 > 0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            DecorrelateSum(NN, StrToInt(EdM.Text), StrToInt(EdN.Text));
          end;
        end;

        if (InnerFCNum > 0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      13:
      begin
        // Branch1
        if StrToInt(EdM.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
          NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdM.Text), FeatureSize-2, iInputPadding, Stride) );
          Branch1 := AddMaxPool();
        end;

        // Branch2
        if StrToInt(EdN.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
          NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdN.Text), FeatureSize, iInputPadding+1, Stride) );
          Branch2 := AddMaxPool();
        end;

        if ( (StrToInt(EdM.Text) > 0) and (StrToInt(EdN.Text) > 0) ) then
        begin
          NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
        end;

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
          end;
        end;
      end;
      14:
      begin
        NN.AddLayer( TNNetConvolutionLinear.Create(24, 3, 1, 1, 1) ).InitBasicPatterns();
        NN.AddDenseNetBlock(6, 12, 48);
        NN.AddDenseNetTransition(0.5, 1, false);
        NN.AddDenseNetBlock(6, 12, 48);
        NN.AddDenseNetTransition(0.5, 1, false);
        NN.AddDenseNetBlock(6, 12, 48);
        NN.AddMovingNorm(false, 0, 0);
        NN.AddLayer( TNNetReLU.Create() );
        //NN.AddLayer( TNNetAvgChannel.Create() );
        NN.AddLayer( TNNetMaxChannel.Create() );
      end;
      15:
       begin
         TNNetInput(InputLayer).EnableErrorCollection;
         NN.AddDenseNetBlockCAI(InnerConvNum div 6, iConvNeuronCount, 0, TNNetConvolutionLinear, false, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         NN.AddDenseNetBlockCAI(InnerConvNum div 6, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         //NN.AddLayer( TNNetMaxPool.Create(2) );
         NN.AddAvgMaxPool(2, 0.25, true);
         NN.AddDenseNetBlockCAI(InnerConvNum div 3, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         //NN.AddLayer( TNNetMaxPool.Create(2) );
         NN.AddAvgMaxPool(2, 0.25, true);
         NN.AddDenseNetBlockCAI(InnerConvNum div 3, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         //NN.AddLayer( TNNetMaxPool.Create(2) );
         //NN.AddLayer( TNNetAvgChannel.Create() );
         NN.AddAvgMaxChannel(0.5, false);
       end;
       16:
       begin
         if ChkLinearFirstConv.Checked
         then NN.AddMinMaxPool(MaxPool)
         else AddMaxPool();
         NN.AddDenseNetBlockCAI(InnerConvNum div 3, iConvNeuronCount, 0, TNNetConvolutionReLU, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, {bottleneck=}iConvNeuronCount, 2, fDropout);
         NN.AddLayer( TNNetMaxPool.Create(2) );
         NN.AddDenseNetBlockCAI(InnerConvNum div 3, iConvNeuronCount, 0, TNNetConvolutionReLU, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, {bottleneck=}iConvNeuronCount, 2, fDropout);
         NN.AddDenseNetBlockCAI(InnerConvNum div 3, iConvNeuronCount, 0, TNNetConvolutionReLU, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, {bottleneck=}iConvNeuronCount, 2, fDropout);
         NN.AddLayer( TNNetMaxChannel.Create() );
       end;
       17:
       begin
         AddMaxPool();
         AddBasicPathways(NN, NumClasses, InnerConvNum, iConvNeuronCount, iInnerConvFeatureSize, iPadding);
       end;
       18:
       begin
         AddMaxPool();
         if (InnerConvNum>0) then
         begin
           for InnerConvCnt := 1 to InnerConvNum do
           begin
             NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0,0) );
           end;
         end;
         AddDensePathways(NN, NumClasses, InnerConvNum, iConvNeuronCount, iInnerConvFeatureSize, iPadding);
       end;
       19:
       begin
        AddMaxPool();
        Start1 := NN.GetLastLayer();
        for InnerConvCnt := 1 to InnerConvNum do
          InputLayer := NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
            ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
            iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
            {RandomBias=}1, {RandomAmplifier=}1);

        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
            ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
            iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
            {RandomBias=}1, {RandomAmplifier=}1, InputLayer);
        for InnerConvCnt := 2 to 2 do
          NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
            ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
            iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
            {RandomBias=}1, {RandomAmplifier=}1);
        Branch1 := NN.AddLayer(TNNetFullConnectLinear.Create(4));  // 4 machines

        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
            ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
            iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
            {RandomBias=}1, {RandomAmplifier=}1, InputLayer);
        for InnerConvCnt := 2 to 2 do
          NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
            ChkMovingNorm.Checked, iConvNeuronCount, iInnerConvFeatureSize,
            iPadding, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
            {RandomBias=}1, {RandomAmplifier=}1);
        Branch2 := NN.AddLayer(TNNetFullConnectLinear.Create(6));  // 6 animals

        NN.AddLayer( TNNetConcat.Create([Branch1, Branch2]) );
       end;
       20:
       begin
         //VGG Like
         NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
           ChkMovingNorm.Checked, 64, 3,
           1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
           {RandomBias=}1, {RandomAmplifier=}1);

        //112x112x64
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 128, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 128, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);

        //56x56x128
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 256, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 256, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 256, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);

        //28x28x256
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddLayer( TNNetMaxPool.Create(2) );

        //14x14x512
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddConvOrSeparableConv(ChkUseSeparableConv.Checked, {HasRelu=}true,
          ChkMovingNorm.Checked, 512, 3,
          1, {Stride=}1, {PerCell=}false, {SuppressBias=}0,
          {RandomBias=}1, {RandomAmplifier=}1);
        NN.AddLayer( TNNetMaxPool.Create(2) );
       end;
       21: // Custom Dense
       begin
         if ChkLinearFirstConv.Checked
         then NN.AddMinMaxPool(MaxPool)
         else AddMaxPool();
         NN.AddDenseNetBlockCAI(InnerConvNum div 2, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         NN.AddLayer( TNNetMaxPool.Create(2) );
         NN.AddDenseNetBlockCAI(InnerConvNum div 2, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         NN.AddLayer( TNNetMaxChannel.Create() );
         if (InnerFCNum>0) then
         begin
           NN.AddDenseFullyConnected(InnerFCNum, iFCNeuronCount, 0, nil, ChkMovingNorm.Checked, {HasReLU=}true);
         end;
       end;
       22: // Simple Dense
       begin
         if ChkLinearFirstConv.Checked
         then NN.AddMinMaxPool(MaxPool)
         else AddMaxPool();
         NN.AddDenseNetBlockCAI(InnerConvNum, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
         NN.AddLayer( TNNetMaxChannel.Create() );
         if ( InnerFCNum > 0 ) then
         begin
           NN.AddDenseFullyConnected(InnerFCNum, iFCNeuronCount, 0, nil, ChkMovingNorm.Checked, {HasReLU=}true);
         end;
       end;
       23: // Machine Animal Dense
       begin
         if ChkLinearFirstConv.Checked
         then NN.AddMinMaxPool(MaxPool)
         else AddMaxPool();
         InputLayer := NN.AddDenseNetBlockCAI(InnerConvNum, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         NN.AddLayer( TNNetPointwiseConvReLU.Create(64) );
         NN.AddDenseNetBlockCAI(2, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         Start1 := NN.AddLayer( TNNetMaxChannel.Create() );
         NN.AddLayerAfter( TNNetPointwiseConvReLU.Create(64), InputLayer );
         NN.AddDenseNetBlockCAI(2, iConvNeuronCount, 0, TNNetConvolutionLinear, ChkUseSeparableConv.Checked, ChkMovingNorm.Checked, nil, nil, 0, 1, fDropout);
         Start2 := NN.AddLayer( TNNetMaxChannel.Create() );
         NoBackProp1 := NN.AddLayerAfter( TNNetIdentityWithoutBackprop.Create(), Start1);
         NoBackProp2 := NN.AddLayerAfter( TNNetIdentityWithoutBackprop.Create(), Start2);
         BeforeMax1  := NN.AddLayer( TNNetConcat.Create([Start1, NoBackProp2]) );
         BeforeMax2  := NN.AddLayer( TNNetConcat.Create([NoBackProp1, Start2]) );
         Branch1 := NN.AddLayerAfter(TNNetFullConnectLinear.Create(4), BeforeMax1);  // 4 machines
         Branch2 := NN.AddLayerAfter(TNNetFullConnectLinear.Create(6), BeforeMax2);  // 6 animals
         NN.AddLayer( TNNetConcat.Create([Branch1, Branch2]) );
       end;
    end; // of case

    if (iAlgo=17) or (iAlgo=18) or (iAlgo=19) or (iAlgo=23) then
    begin
      if (ComboLastLayer.ItemIndex = 0)
        then NN.AddLayer( TNNetSoftMax.Create() )
        else NN.AddLayer( TNNetSigmoid.Create() );
    end
    else if (ComboLastLayer.ItemIndex = 0) then
    begin
      NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
      NN.AddLayer( TNNetSoftMax.Create() );
    end
    else
    begin
      NN.AddLayer( TNNetFullConnectSigmoid.Create(NumClasses) );
    end;

    NN.SetLearningRate(CurrentLearningRate, fInertia);
  end;

  if NN.Layers[NN.GetLastLayerIdx()] is TNNetSoftMax then
  begin
    WriteLn('Neural network has Softmax.');
    bIsSoftmax := true;
  end
  else
  begin
    WriteLn('Softmax isn''t present.');
  end;

  FFinishedThread.ReSize(FThreadNum);
  FThreadNN := TNNetDataParallelism.Create(NN, FThreadNum);
  FAvgWeights := nil;
  FAvgWeight := NN.Clone();

  WriteLn('Setting L2 to:', fL2Decay:6:4, '  Learning rate:',CurrentLearningRate:6:4,'  Staircase ephocs:',iStaircaseEpochs);
  FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
  FThreadNN.SetBatchUpdate(true);

  if RadL2All.Checked
  then NN.SetL2Decay(fL2Decay)
  else NN.SetL2DecayToConvolutionalLayers(fL2Decay);
  NN.SetLearningRate(CurrentLearningRate, fInertia);

  // in batch update, threaded NN should not apply L2 (L2 is applied in the main thread).
  FThreadNN.SetL2Decay(0);

  //--------------------------------------------------------------------
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  vDisplay:= TNNetVolume.Create(NumClasses,1,1);
  firstNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx();

  if ( iAlgo = 6 ) or ( iAlgo = 7 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    thirdNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(secondNeuronalLayer + 1);
    CreateNeuronImages(thirdNeuronalLayer, 16, 16, FeaturesNum * 3);
  end
  else if ( iAlgo = 4 ) or ( iAlgo = 13 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(secondNeuronalLayer, 32, 8, FeaturesNum);
  end
  else if ( iAlgo = 5 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 32, 8, FeaturesNum * 2);
  end
  else if ( iAlgo = 9 ) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8, FeaturesNum);
  end
  else if ( iAlgo = 8 ) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8, FeaturesNum);
  end
  else if ( ( FeaturesNum > 64 ) or ( NN.Layers[firstNeuronalLayer].Neurons.Count > 64 ) ) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 16, 16, NN.Layers[firstNeuronalLayer].Neurons.Count);
  end
  else if ( NN.Layers[firstNeuronalLayer].Neurons.Count = 1 ) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 256, 1, NN.Layers[firstNeuronalLayer].Neurons.Count);
  end
  else
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8, NN.Layers[firstNeuronalLayer].Neurons.Count);
  end;

  globalImgInput.Copy(ImgVolumes[0]); // frog
  DisplayInputImage(color_encoding);

  NN.DebugWeights();
  WriteLn('Neural network has: ');
  NN.DebugStructure();
  LabCounts.Caption :=
    'Layers: '+IntToStr(NN.CountLayers())+
    '  Neurons: '+IntToStr(NN.CountNeurons())+
    '  Weights: '+IntToStr(NN.CountWeights());
  WriteLn('Computing...');

  globalStartTime := Now();
  while ( FRunning and (iMaxEpochs > iEpochCount) ) do
  begin
    if ChkRestartAtEveryEpoch.Checked then NN.InitWeights();
    globalErrorSum := 0;
    startTime := Now();
    CheckLearningRate();
    globalClassHits.Fill(0);
    for I := 1 to (ImgVolumes.Count div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;
      FFinishedThread.Fill(0);
      NN.ClearTime();
      ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
      //NN.DebugWeights();
      if ClipDelta > 0 then
      begin
        MaxDelta := NN.ForceMaxAbsoluteDelta(ClipDelta);
      end
      else
      begin
        MaxDelta := NN.NormalizeMaxAbsoluteDelta();
        if MaxDelta < 1 then
        begin
          WriteLn('Deltas have been multiplied by:', MaxDelta:8:4 );
        end;
      end;
      NN.UpdateWeights();
      NN.ComputeL2Decay();

      if (globalHit > 0) then
      begin
        CurrentRate := globalHit / (globalHit + globalMiss);
        CVSError := globalErrorSum / (globalHit + globalMiss);
        CVSLoss  := globalTotalLoss / (globalHit + globalMiss);
        CVSRate  := CurrentRate;
        CurrentAccuracy := (globalHit*100) div (globalHit+globalMiss);
        if (FStepSize < 100) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.99 + CurrentAccuracy*0.01;
        end
        else if (FStepSize < 500) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.9 + CurrentAccuracy*0.1;
        end
        else
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end;
      end;

      if ( (globalHit > 0) and (I mod 10 = 0) ) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;

        WriteLn
        (
          (globalHit + globalMiss)*I + iEpochCount*ImgVolumes.Count,
          ' Examples seen. Accuracy:', CVSRate:6:4,
          ' Error:', CVSError:10:5,
          ' Loss:', CVSLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Thread Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Thread Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Time:', totalTimeSeconds:6:2,'s'
        );
        if Not(bIsSoftmax) then PrintFrequencies(globalClassHits);
        LabClassRate.Caption := PadLeft(IntToStr(Round(AccuracyWithInertia))+'%',4) ;

        LabTime.Caption :=
          'Epoch time: ' + FloatToStrF( totalTimeSeconds*(50000/(FStepSize*10))/60,ffGeneral,1,4)+' minutes.' +
          ' 100 epochs: ' + FloatToStrF( 100*totalTimeSeconds*(50000/(FStepSize*10))/3600,ffGeneral,1,4)+' hours.';

        LabTotalTime.Caption :=
          'Epochs: '+IntToStr(iEpochCount)+
          '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.';

        startTime := Now();
        Application.ProcessMessages;
      end;
    end;

    if (FRunning) then
    begin
      if (iEpochCount mod 4 = 0) then
      begin
        if ( ( iAlgo = 6 ) or ( iAlgo = 7 ) ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, FeaturesNum*0, 16, color_encoding);
          ShowNeuronsIndependentChannels(secondNeuronalLayer, FeaturesNum*1, 16, color_encoding);
          ShowNeuronsIndependentChannels(thirdNeuronalLayer, FeaturesNum*2, 16, color_encoding);
        end
        else if ( iAlgo = 4 ) or ( iAlgo = 13 ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 32, color_encoding);
          if ( (StrToInt(EdM.Text) > 0) and (StrToInt(EdN.Text) > 0) ) then
          begin
            ShowNeuronsIndependentChannels(secondNeuronalLayer, StrToInt(EdM.Text), 32, color_encoding);
          end;
        end
        else if ( iAlgo = 5 )  then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 32, color_encoding);
          ShowNeuronsIndependentChannels(secondNeuronalLayer, FeaturesNum, 32, color_encoding);
        end
        else if ( iAlgo = 9 ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 32, color_encoding);
        end
        else if ( iAlgo = 8 ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 32, color_encoding);
        end
        else
        begin
          if (NN.Layers[firstNeuronalLayer].Neurons.Count = 1) then
          begin
            ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 128, color_encoding);
          end
          else if FeaturesNum>64 then
          begin
            ShowNeuronsIndependentChannels(firstNeuronalLayer, FeaturesNum*0, 16, color_encoding);
          end
          else
          begin
            ShowNeuronsIndependentChannels(firstNeuronalLayer, FeaturesNum*0, 32, color_encoding);
          end;
        end;
      end;
    end;

    Inc(iEpochCount);
    Inc(iEpochCountAfterLoading);

    if not Assigned(FAvgWeights) then
    begin
      FAvgWeight.CopyWeights(NN);
      if chkWeightAverage.Checked
      then FAvgWeights := TNNetDataParallelism.Create(NN, EpochsForAverage)
      else FAvgWeights := TNNetDataParallelism.Create(NN, 1)
    end;

    if (FRunning) then
    begin
      if chkWeightAverage.Checked then
      begin
        FAvgWeights.ReplaceAtIdxAndUpdateWeightAvg(iEpochCount mod EpochsForAverage, NN, FAvgWeight);
      end
      else
      begin
        FAvgWeight.CopyWeights(NN);
        //FAvgWeight.AddToExponentialWeightAverage(NN, 0.9);
        //FAvgWeight.AddToWeightAverage(NN, iEpochCountAfterLoading);
      end;

      ImgWorkingVolumes := ImgValidationVolumes;
      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;
      FConfusionMatrix.Fill(0);
      WriteLn('Starting Validation.');
      ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

      if globalHit + globalMiss > 0 then
      begin
        CVSValidationRate  := globalHit / (globalHit + globalMiss);
        CVSValidationLoss  := globalTotalLoss / (globalHit + globalMiss);
        CVSValidationError := globalErrorSum / (globalHit + globalMiss);
      end;

      if
        (CVSValidationRate > ValidationRecord) and
        Not(ChkRestartAtEveryEpoch.Checked) then
      begin
        ValidationRecord := CVSValidationRate;
        WriteLn('Saving Validation Record.');
        FAvgWeight.SaveToFile(fileName);
        //NN.CopyWeights(FAvgWeight);
      end;
      //WriteLn(globalHit ,' ', globalMiss);
      LabTestRate.Caption := PadLeft(IntToStr( Round(CVSValidationRate*100) )+'%',4) ;

      if (iEpochCount mod 4 = 0) then DisplayInputImage(color_encoding);

      if (iEpochCount mod FThreadNN.Count = 0) then
      begin
        Application.ProcessMessages;
        NN.DebugWeights();
        NN.DebugErrors();
        SaveScreenshot(fileNameImage);
        SaveNeuronsImage(fileNameNeuronsImage);
        Application.ProcessMessages;
      end;

      if ( (iEpochCount mod 10 = 0) and (iEpochCount > 0) ) then
      begin
        ImgWorkingVolumes := ImgTestVolumes;
        globalHit       := 0;
        globalMiss      := 0;
        globalTotalLoss := 0;
        globalErrorSum  := 0;
        FConfusionMatrix.Fill(0);
        WriteLn('Starting Testing.');
        ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

        if globalHit + globalMiss > 0 then
        begin
          CVSTestRate  := globalHit / (globalHit + globalMiss);
          CVSTestLoss  := globalTotalLoss / (globalHit + globalMiss);
          CVSTestError := globalErrorSum / (globalHit + globalMiss);
        end;

        WriteLn
        (
          CSVFile,
          iEpochCount,',',
          (AccuracyWithInertia/100):6:4,',',
          CVSLoss:6:4,',',
          CVSError:6:4,',',
          CVSValidationRate:6:4,',',
          CVSValidationLoss:6:4,',',
          CVSValidationError:6:4,',',
          CurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',',
          CVSTestRate:6:4,',',
          CVSTestLoss:6:4,',',
          CVSTestError:6:4
        );
      end
      else
      begin
        writeln
        (
          CSVFile,
          iEpochCount,',',
          (AccuracyWithInertia/100):6:4,',',
          CVSLoss:6:4,',',
          CVSError:6:4,',',
          CVSValidationRate:6:4,',',
          CVSValidationLoss:6:4,',',
          CVSValidationError:6:4,',',
          CurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60)
        );
      end;
      Application.ProcessMessages();
      ConfusionWriteCSV(CSVConfusion, FConfusionMatrix, 6);
      CloseFile(CSVFile);
      CloseFile(CSVConfusion);
      AssignFile(CSVFile, FileNameCSV);
      AssignFile(CSVConfusion, fileNameConfusion);
      Append(CSVFile);
      Append(CSVConfusion);

      if (globalHit > 0) then
      begin
        WriteLn
        (
          'Epochs: ',iEpochCount,
          ' Examples seen:', iEpochCount*ImgVolumes.Count,
          ' Accuracy:', CVSRate:6:4,' ', CVSValidationRate:6:4,
          ' Error: ', CVSError:10:5,' ',CVSTestError:10:5,
          ' Loss: ', CVSLoss:7:5,' ',CVSTestLoss:7:5,
          ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
        );
      end;
      Application.ProcessMessages();
    end;
  end;

  FreeNeuronImages();
  FAvgWeight.Free;
  if Assigned(FAvgWeights) then FAvgWeights.Free;
  FThreadNN.Free;
  CloseFile(CSVFile);
  CloseFile(CSVConfusion);
  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.EnableComponents(flag: boolean);
var
  i : Integer;
begin
  for i := 0 to ComponentCount-1 do
  begin
    if (Components[i] is TEdit) then
      TEdit(Components[i]).Enabled := flag;

    if (Components[i] is TComboBox) then
       TComboBox(Components[i]).Enabled := flag;

    if (Components[i] is TCheckBox) then
       TCheckBox(Components[i]).Enabled := flag;

    if (Components[i] is TRadioButton) then
       TRadioButton(Components[i]).Enabled := flag;
  end;

  Application.ProcessMessages;
end;

procedure TFormVisualLearning.SaveScreenshot(filename: string);
begin
  WriteLn(' Saving ',filename,'.');
  SaveHandleToBitmap(filename, Self.Handle);
end;

procedure TFormVisualLearning.SaveNeuronsImage(filename: string);
begin
  WriteLn(' Saving ',filename,'.');
  SaveHandleToBitmap(filename, GrBoxNeurons.Handle);
end;


procedure TFormVisualLearning.RunNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize, BlockSizeRest, CropSizeX, CropSizeY: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput, LocalClassHits: TNNetVolume;
  I, ImgIdx: integer;
  ImgShiftSize, ImgShiftCnt, ImgShiftCntD: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  LocalClassHits := TNNetVolume.Create(1, NumClasses, 2, 0);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := FStepSize div FThreadNum;
  BlockSizeRest := FStepSize mod FThreadNum;

  if (Index < BlockSizeRest) then Inc(BlockSize);

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(NN);
  LocalNN.ClearTime();
  LocalNN.ClearDeltas();
  LocalNN.EnableDropouts(true);
  for I := 1 to BlockSize do
  begin
    if not(FRunning) then Break;
    ImgIdx := Random(ImgVolumes.Count);

    //if Not( (ImgVolumes[ImgIdx].Tag = 0) or (Random(100) > 81) ) then continue;
    //if ImgVolumes[ImgIdx].Tag <> 0 then ImgVolumes[ImgIdx].Tag := 1; // this is for testing only

    if FImgCrop then
    begin
      ImgInput.CopyCropping(ImgVolumes[ImgIdx], random(8), random(8), 24, 24);
    end
    else
    begin
      if (FDataAugmentationType < 8) then
      begin
        CropSizeX := random(FMaxCrop + 1);
        CropSizeY := random(FMaxCrop + 1);
        //ImgInputCp.CopyResizing(ImgVolumes[ImgIdx], 40, 40);
        //ImgInput.CopyCropping(ImgInputCp, random(CropSizeX), random(CropSizeY),ImgVolumes[ImgIdx].SizeX, ImgVolumes[ImgIdx].SizeY);
        ImgInputCp.CopyCropping(ImgVolumes[ImgIdx], random(CropSizeX), random(CropSizeY),ImgVolumes[ImgIdx].SizeX-CropSizeX, ImgVolumes[ImgIdx].SizeY-CropSizeY);
        ImgInput.CopyResizing(ImgInputCp, ImgVolumes[ImgIdx].SizeX, ImgVolumes[ImgIdx].SizeY);
      end
      else
      begin
        ImgInput.Copy(ImgVolumes[ImgIdx]);
      end;
    end;

    // flip is always used in training
    if Random(1000) > 500 then
    begin
      ImgInput.FlipX();
    end;
    ImgInput.Tag := ImgVolumes[ImgIdx].Tag;

    if bDataAugmentation and (FDataAugmentationType < 8) then
    begin
      if ( (FDataAugmentationType = 1) or (FDataAugmentationType = 7) ) then
      begin
        // Modifies the image
        ImgInputCp.Copy(ImgInput);
        ImgShiftSize := Round ( Random(1280)*FNoiseLevel ) * ImgInput.Depth;
        if (ImgShiftSize > 0) then
        begin
          ImgInput.ShiftRight( ImgShiftSize ); //Shifts input image right and down

          ImgShiftCntD := 0;
          for ImgShiftCnt := ImgInput.Size - ImgShiftSize to ImgInput.Size - 1 do
          begin
            ImgInput.FData[ImgShiftCntD] := ImgInputCp.FData[ImgShiftCnt];
            Inc( ImgShiftCntD );
          end;
        end;
        //WriteLn('Testing ImgInputAdd:',ImgShiftSize,' : ', ( ( (Random(1024)-512)*FNoiseLevel) ) );
      end;

      if ( (FDataAugmentationType = 2) or (FDataAugmentationType = 4) ) then
      begin
        // Random "add"
        ImgInput.AddAtDepth(0, ( (Random(1024)-512)*FNoiseLevel) / 2560 );
      end;

      if ( (FDataAugmentationType = 3) or (FDataAugmentationType = 4) ) then
      begin
        ImgInput.AddSaltAndPepper( Round( ( (ImgInput.SizeX * ImgInput.SizeY) div 20)*FNoiseLevel ) );
      end;

      if ( (FDataAugmentationType = 6) or (FDataAugmentationType = 7) ) then
      begin
        if (ImgInput.Depth > 1) then
        begin
          if (Random(1000) > 750) then
          begin
            ImgInput.MakeGray(color_encoding);
          end;
        end;
      end;
    end;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );

    if bIsSoftmax
      then vOutput.SetClassForSoftMax( ImgInput.Tag )
      else vOutput.SetClass( ImgInput.Tag, +0.9, -0.1);

    LocalErrorSum += vOutput.SumDiff( pOutput );
    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(bIsSoftmax) then
    begin
      OutputToClassHits(LocalClassHits, pOutput, vOutput);
      OutputValue := Max(OutputValue, 0.001);
      LocalNN.Backpropagate(vOutput);
    end
    else
    begin
      LocalNN.Backpropagate(vOutput);
    end;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss += CurrentLoss;

    if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for

  if (FThreadNum - 1 = Index) then
  begin
    globalImgInput.Copy(ImgInput);
  end;

  if Index and 1 = 0 then
  begin
    if Index + 1 < FThreadNum then
    begin
      while FFinishedThread.FData[Index + 1] = 0 do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 1]);
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 1];
    end;
  end;
  FFinishedThread.FData[Index] += 1;
  if Index and 3 = 0 then
  begin
    if Index + 2 < FThreadNum then
    begin
      while FFinishedThread.FData[Index + 2] = 0 do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 2]);
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 2];
    end;
  end;
//  WriteLn('Index:',Index,' [',FFinishedThread.FData[Index],']');

  EnterCriticalSection(FCritSec);
  globalHit       += LocalHit;
  globalMiss      += LocalMiss;
  globalTotalLoss += LocalTotalLoss;
  globalErrorSum  += LocalErrorSum;
  globalClassHits.Add(LocalClassHits);

  NN.ForwardTime := NN.ForwardTime + LocalNN.ForwardTime;
  NN.BackwardTime := NN.BackwardTime + LocalNN.BackwardTime;
  {$IFDEF Debug}
  if Index and 3 = 0 then NN.SumDeltas(LocalNN);
  {$ELSE}
  if Index and 3 = 0 then NN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}
  LocalClassHits.Free;
  LeaveCriticalSection(FCritSec);
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.TestNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput, sumOutput, LocalFrequency: TNNetVolume;
  I, ImgIdx: integer;
  StartPos, FinishPos: integer;
  PredictedOutputValue, OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  PredictedClass, NPredictedClass: integer;
  cntYAY, cntBAD: integer;
begin
  cntYAY := 0;
  cntBAD := 0;
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  sumOutput := TNNetVolume.Create(NumClasses,1,1);
  LocalFrequency := TNNetVolume.Create();

  ClassHitsToFrequency(globalClassHits, LocalFrequency);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := (ImgWorkingVolumes.Count div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FAvgWeight);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if not(FRunning) then Break;
    sumOutput.Fill(0);
    ImgIdx := I;
    //if Not( (ImgWorkingVolumes[ImgIdx].Tag = 0) or (Random(100) > 81) ) then continue;
    //if ImgWorkingVolumes[ImgIdx].Tag <> 0 then ImgWorkingVolumes[ImgIdx].Tag := 1; // this is for testing only

    if FImgCrop then
    begin
      ImgInput.CopyCropping(ImgWorkingVolumes[ImgIdx], 4, 4, 24, 24);
    end
    else
    begin
      ImgInput.Copy(ImgWorkingVolumes[ImgIdx]);
    end;

    ImgInput.Tag := ImgWorkingVolumes[ImgIdx].Tag;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );
    if bMultipleSamplesAtValidation then
    begin
      sumOutput.Add( pOutput );

      ImgInput.FlipX();

      LocalNN.Compute( ImgInput );
      LocalNN.GetOutput( pOutput );
      sumOutput.Add( pOutput );

      if ImgInput.SizeX >= 32 then
      begin
        ImgInputCp.CopyCropping(ImgInput, FMaxCrop div 2, FMaxCrop div 2, ImgInput.SizeX - FMaxCrop, ImgInput.SizeY - FMaxCrop);
        ImgInput.CopyResizing(ImgInputCp, ImgInput.SizeX, ImgInput.SizeY);
        //ImgInputCp.CopyResizing(ImgVolumes[ImgIdx], 40, 40);
        //ImgInput.CopyCropping(ImgInputCp, 4, 4, ImgVolumes[ImgIdx].SizeX, ImgVolumes[ImgIdx].SizeY);

        LocalNN.Compute( ImgInput );
        LocalNN.GetOutput( pOutput );
        sumOutput.Add( pOutput );

        sumOutput.Divi(3);
      end
      else
      begin
        sumOutput.Divi(2);
      end;

      pOutput.Copy(sumOutput);
    end;

    vOutput.SetClassForSoftMax( ImgInput.Tag );
    LocalErrorSum += vOutput.SumDiff( pOutput );

    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(bIsSoftmax) then OutputValue += 0.5001;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss += CurrentLoss;

    FConfusionMatrix.Add(0, ImgInput.Tag, pOutput.GetClass(), 1);

    PredictedClass := pOutput.GetClass();
    PredictedOutputValue := pOutput.FData[PredictedClass];
(*
    // should use frequency
    if PredictedOutputValue > csThreshold then
    begin
      NPredictedClass := GetClassFromPredictionAndFrequency(pOutput, LocalFrequency);
      if (NPredictedClass<>-1) and (NPredictedClass <> PredictedClass) then
      begin
        Write('Changing predicted class from ',
        PredictedClass,'(',LocalFrequency[0,PredictedClass,0]:4:2,')',
        ' to ',
        NPredictedClass,'(',LocalFrequency[0,NPredictedClass,0]:4:2,'):');
        if NPredictedClass = ImgInput.Tag then
        begin
          Write('YAY');
          Inc(cntYAY);
        end
        else if PredictedClass = ImgInput.Tag then
        begin
          Write('BAD');
          Inc(cntBAD);
        end
        else Write('no change');
        WriteLn;
        (*if LocalFrequency[0,NPredictedClass,0] < LocalFrequency[0,PredictedClass,0] then
        begin
          WriteLn('oooppsss!!!');
          ReadLn;
        end;*)
        PredictedClass := NPredictedClass;
        PredictedOutputValue := pOutput.FData[PredictedClass];
      end;
    end;
*)
    //WriteLn( ImgInput.Tag, ' ', pOutput.GetClass() );
    if PredictedClass = ImgInput.Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for
  LocalNN.EnableDropouts(true);

  EnterCriticalSection(FCritSec);
  globalHit       += LocalHit;
  globalMiss      += LocalMiss;
  globalTotalLoss += LocalTotalLoss;
  globalErrorSum  += LocalErrorSum;
  LeaveCriticalSection(FCritSec);

  LocalFrequency.Free;
  sumOutput.Free;
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
  //WriteLn('Fixes: bad=',cntBAD,' yay=',cntYAY);
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.

