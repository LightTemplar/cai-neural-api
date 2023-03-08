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

unit utinyImageNet200;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets, neuraldatasetsv,
  neuralvolume, MTProcs, math;

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    CheckAutoNoisy: TCheckBox;
    CheckProportionalLearningRate: TCheckBox;
    CheckCenter: TCheckBox;
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
    EdImageSubFolder: TEdit;
    EdValidationFolder: TEdit;
    EdN: TEdit;
    EdM: TEdit;
    EdNoiseLevel: TEdit;
    EdLearnRateDecay: TEdit;
    EdPhysThreads: TEdit;
    EdTestFolder: TEdit;
    EdThreadCount: TEdit;
    EdL2Decay: TEdit;
    EdMinLearnRate: TEdit;
    EdMaxEpochs: TEdit;
    EdStaircaseEpochs: TEdit;
    EdStride: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    EdTrainFolder: TEdit;
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
    procedure ComboAlgoChange(Sender: TObject);
    procedure EdTrainFolderChange(Sender: TObject);
    procedure EdLearningRateChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure GroupBox1Click(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FThreadNum, FStepSize: integer;
    FThreadNN: TNNetDataParallelism;
    FNoiseLevel: TNeuralFloat;
    FDataAugmentationType: integer;
    FTrainingImageCount: integer;
    FValidationImageCount: integer;
    FTestImageCount: integer;
    FTrainingImageClassCount: integer;

    FTrainImages: TClassesAndElements;
    FTestImages: TClassesAndElements;
    FValidationImages: TClassesAndElements;
    FWorkingImages: TClassesAndElements;

    //ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    //ImgWorkingVolumes: TNNetVolumeList;
    globalImgInput: TNNetVolume;
    globalHit, globalMiss: integer;
    globalErrorSum: TNeuralFloat;
    globalTotalLoss: TNeuralFloat;
    color_encoding: integer;

    bIsSoftmax: boolean;
    bDataAugmentation: boolean;
    bLoadedFile: boolean;

    globalLoopCnt: integer;
    NN: TNNet;

    FCritSec: TRTLCriticalSection;

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

{ TFormVisualLearning }

const csLogEveryCicles: integer = 1;

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

procedure TFormVisualLearning.ComboAlgoChange(Sender: TObject);
begin
  if ComboAlgo.ItemIndex = 0 then
  begin
    EdMaxPool.Enabled := true;
    EdConvLayers.Enabled := true;
    EdFCLayers.Enabled := true;
  end
  else
  begin
    EdMaxPool.Enabled := false;
    EdConvLayers.Enabled := false;
    EdFCLayers.Enabled := false;
  end;
end;

procedure TFormVisualLearning.EdTrainFolderChange(Sender: TObject);
begin

end;

procedure TFormVisualLearning.EdLearningRateChange(Sender: TObject);
begin
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
  FTrainImages := TClassesAndElements.Create();
  FTestImages := TClassesAndElements.Create();
  FValidationImages := TClassesAndElements.Create();
  //EdThreadCount.Text := IntToStr(ProcThreadPool.MaxThreadCount);
  //EdPhysThreads.Text := IntToStr(ProcThreadPool.MaxThreadCount);
  globalImgInput := TNNetVolume.Create();
  InitCriticalSection(FCritSec);
  ComboAlgoChange(Sender);
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  DoneCriticalSection(FCritSec);
  globalImgInput.Free;
  FTrainImages.Free;
  FTestImages.Free;
end;

procedure TFormVisualLearning.GroupBox1Click(Sender: TObject);
begin

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
  begin
    PosTop  := 14;
    PosLeft := 22;

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
    end;

    GrBoxNeurons.Height := aImage[NeuronNum-1].Top  + filterSize + 24;
    GrBoxNeurons.Width  := aImage[NeuronNum-1].Left + filterSize + 10;

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
  I: integer;
  pOutput, vOutput: TNNetVolume;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  CurrentRate: double;
  CurrentLearningRate, fLearningRate, fInertia: single;
  MaxPool,Stride: integer;
  fDropout: single;
  FeaturesNum, FeatureSize: integer;
  InnerConvNum, InnerConvCnt: integer;
  InnerFCNum, InnerFCCnt: integer;
  fileName, fileNameBase, FileNameCSV, fileNameImage, fileNameNeuronsImage: string;
  firstNeuronalLayer, secondNeuronalLayer, thirdNeuronalLayer: integer;
  iMaxEpochs, iStaircaseEpochs: integer;
  iFCNeuronCount, iConvNeuronCount: integer;
  iLearningRateCounter: integer;
  fMinLearnRate, fNewLearningRate: single;
  fLearningRateDecay: single;
  bCenterDivide, bAutoNoisy: boolean;
  CVSError, CVSLoss, CVSRate,
  CVSTestError, CVSTestLoss, CVSTestRate: TNeuralFloat;
  CVSValidationError, CVSValidationLoss, CVSValidationRate: TNeuralFloat;
  CSVFile: TextFile;
  fL2Decay: TNeuralFloat;
  iAlgo: integer;
  iPadding: integer;
  iInnerConvFeatureSize: integer;
  iInputDepth: integer;
  InputLayer, Branch1, Branch2, Branch3, Start1, Start2, PlusOne, EnhancedInput: TNNetLayer;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  LocalPicture: TPicture;
begin
  color_encoding := ComboColor.ItemIndex;
  WriteLn('Searching for input images.');
  FTrainImages.Clear();
  FTestImages.Clear();
  FValidationImages.Clear();

  FTrainImages.LoadFoldersAsClassesProportional(EdTrainFolder.Text, EdImageSubFolder.Text, 0, 0.8);
  FTestImages.LoadFoldersAsClassesProportional(EdTestFolder.Text, EdImageSubFolder.Text, 0.8, 0.1);
  FValidationImages.LoadFoldersAsClassesProportional(EdTestFolder.Text, EdImageSubFolder.Text, 0.9, 0.1);

  WriteLn('Train classes: ', FTrainImages.Count);
  WriteLn('Validation classes: ', FValidationImages.Count);
  WriteLn('Test classes: ', FTestImages.Count);

  WriteLn('Loading test images.');
  FTestImages.LoadImages(color_encoding);
  FTestImageCount := FTestImages.CountElements();

  WriteLn(FTestImageCount,' test images loaded. Loading validation images.');
  FValidationImages.LoadImages(color_encoding);
  FValidationImageCount := FValidationImages.CountElements();
  WriteLn(FValidationImageCount, ' validation images loaded. Loading training images.');
  FTrainImages.LoadImages(color_encoding);
  FTrainingImageCount := FTrainImages.CountElements();

  FTrainingImageClassCount := FTrainImages.Count;

  WriteLn('Training Classes:', FTrainImages.Count, ' Test Classes:', FTrainingImageClassCount, ' Validation Classes:', FValidationImages.Count );
  WriteLn('Training Images:', FTrainingImageCount, ' Test Images:', FTestImageCount, ' Validation Images:', FValidationImageCount );

  LocalPicture := TPicture.Create;
  LocalPicture.LoadFromFile( FTrainImages.GetFileName(0,0) );
  LoadPictureIntoVolume(LocalPicture, globalImgInput);

  AccuracyWithInertia := 10;
  FThreadNum := StrToInt(EdThreadCount.Text);
  FDataAugmentationType := ComboAugmentation.ItemIndex;
  FNoiseLevel := StrToFloat(EdNoiseLevel.Text);
  ProcThreadPool.MaxThreadCount := StrToInt(EdPhysThreads.Text);

  FStepSize := StrToInt(EdBatchSize.Text);

  globalLoopCnt := 0;
  bLoadedFile := false;
  bAutoNoisy  := CheckAutoNoisy.Checked;
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

  if color_encoding = csEncodeGray then
  begin
    iInputDepth := 1;
  end
  else
  begin
    iInputDepth := 3;
  end;

  iAlgo := ComboAlgo.ItemIndex;
  iPadding := StrToInt(EdConvPadding.Text);
  iInnerConvFeatureSize := StrToInt(EdInnerConvFeatureSize.Text);

  WriteLn('Number of threads:', FThreadNum);
  WriteLn('Algorithm:',iAlgo,' Color Encoding:', color_encoding,' Input Channels:', iInputDepth, ' Step Size:', FStepSize);

  if iAlgo = 0 then
  begin
    fileNameBase :=
      'autosave-'+
      EdLearningRate.Text+'-'+
      EdLearnRateDecay.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdDropout.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdStride.Text+'-'+
      EdMaxPool.Text+'-'+
      EdInnerConvFeatureSize.Text+'-'+
      EdConvPadding.Text+'-'+
      EdConvLayers.Text+'-'+
      EdInnerConvNeuronCount.Text+'-'+
      EdFCLayers.Text+'-'+
      EdFCNeuronCount.Text+'-'+
      BoolToStr(CheckCenter.Checked,'T','F')+'-'+
      BoolToStr(CheckAutoNoisy.Checked,'T','F')+'-'+
      EdBatchSize.Text+'-'+
      ComboAugmentation.Text+'-'+
      EdNoiseLevel.Text+'-'+
      ComboColor.Text+'-'+
      ComboLastLayer.Text;
  end
  else
  begin
    fileNameBase :=
      'autosave-algo'+
      IntToStr(iAlgo)+'-'+
      EdLearningRate.Text+'-'+
      EdLearnRateDecay.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdDropout.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdStride.Text+'-'+
      EdMaxPool.Text+'-'+
      EdInnerConvFeatureSize.Text+'-'+
      EdConvPadding.Text+'-'+
      BoolToStr(CheckCenter.Checked,'T','F')+'-'+
      BoolToStr(CheckAutoNoisy.Checked,'T','F')+'-'+
      EdThreadCount.Text+'-'+
      EdBatchSize.Text+'-'+
      ComboAugmentation.Text+'-'+
      EdNoiseLevel.Text+'-'+
      ComboColor.Text+'-'+
      ComboLastLayer.Text;
  end;

  if ( (iAlgo=4) or (iAlgo=5) ) then
  begin
    fileNameBase += '-M'+EdM.Text+'N'+EdN.Text;
  end;

  WriteLn('File name is: ',fileNameBase);

  FileNameCSV := fileNameBase + '.csv';

  FileName := fileNameBase + '.nn';

  fileNameImage := fileNameBase + '.bmp';

  fileNameNeuronsImage := fileNameBase + '-neurons.bmp';

  AssignFile(CSVFile, FileNameCSV);

  FeaturesNum := StrToInt(EdFeatures.Text);
  FeatureSize := StrToInt(EdFeatureSize.Text);
  fLearningRate := StrToFloat(EdLearningRate.Text);
  fLearningRateDecay := 1 - StrToFloat(EdLearnRateDecay.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fLearningRate;
  fDropout := StrToFloat(EdDropout.Text);
  MaxPool := StrToInt(EdMaxPool.Text);
  Stride := StrToInt(EdStride.Text);
  iMaxEpochs := StrToInt(EdMaxEpochs.Text);
  iStaircaseEpochs := StrToInt(EdStaircaseEpochs.Text);
  iFCNeuronCount := StrToInt(EdFCNeuronCount.Text);
  iConvNeuronCount := StrToInt(EdInnerConvNeuronCount.Text);
  iLearningRateCounter := 0;
  fMinLearnRate := StrToFloat(EdMinLearnRate.Text);
  fL2Decay := StrToFloat(EdL2Decay.Text);

  if (FStepSize > 1) and (fL2Decay>0) then
  begin
    fL2Decay := (1 - power( 1 - fL2Decay, FStepSize) );
    WriteLn('Actual L2:', fL2Decay:6:4);
  end;

  LabComputedL2.Caption := 'Computed L2: '+FloatToStrF(fL2Decay,ffGeneral,1,6)+' times learning rate.';

  bDataAugmentation := (ComboAugmentation.ItemIndex > 0);
  bCenterDivide := CheckCenter.Checked;

  if bAutoNoisy then
  begin
    bDataAugmentation := false;
  end;

  writeln('Creating Neural Network...');
  NN := TNNet.Create();

  if ( FileExists(fileName) and FileExists(FileNameCSV) ) then
  begin
    writeln('Loading neural network from file: ',fileName);
    Append(CSVFile);
    NN.LoadFromFile(fileName);
    // as this is a loaded NN, we'll start measuring and not learning
    NN.SetLearningRate(0, fInertia);
    if CheckProportionalLearningRate.Checked then bLoadedFile := true;
  end
  else
  begin
    ReWrite(CSVFile);
    writeln(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');

    InputLayer := NN.AddLayer( TNNetInput.Create(globalImgInput.SizeX,globalImgInput.SizeY,iInputDepth) );

    if ( (iAlgo<>8) and (iAlgo<>6) and (iAlgo<>5) and (iAlgo<>4) and (iAlgo<>3) ) then
    begin
      NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,FeatureSize,0,Stride) );
    end;

      case iAlgo of
      0:
      begin

        if MaxPool>0 then
        begin
          NN.AddLayer( TNNetMaxPool.Create(MaxPool) );
          if bCenterDivide then NN.AddLayer( TNNetLayerStdNormalization.Create() );
        end;

        InnerConvNum := StrToInt(EdConvLayers.Text);
        InnerFCNum := StrToInt(EdFCLayers.Text);

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) );
          end;
          //NN.AddLayer( TNNetLocalResponseNorm2D.Create(11) );
          //NN.AddLayer( TNNetLocalResponseNormDepth.Create(11) );
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(iFCNeuronCount) );
            if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
          end;
        end;

      end;
      1:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      2:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      // https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
      3: // TENSOR FLOW INSPIRED
      begin
        NN.AddLayer( TNNetConvolutionReLU.Create(64, 5, 2, 1) );
        NN.AddLayer( TNNetMaxPool.Create(3) );
        NN.AddLayer( TNNetLocalResponseNormDepth.Create(11) );
        NN.AddLayer( TNNetConvolutionReLU.Create(64, 5, 2, 1) );
        NN.AddLayer( TNNetMaxPool.Create(2) );
        NN.AddLayer( TNNetLocalResponseNormDepth.Create(11) );
        NN.AddLayer( TNNetFullConnectReLU.Create(384) );
        NN.AddLayer( TNNetFullConnectReLU.Create(192) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      4:
      begin
        // Branch1
        if StrToInt(EdM.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
          NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdM.Text), FeatureSize, 0, 0) );
          Branch1 := NN.AddLayer( TNNetMaxPool.Create(MaxPool) );
        end;

        // Branch2
        if StrToInt(EdN.Text) > 0 then
        begin
          NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
          NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdN.Text), FeatureSize, 0, 0) );
          Branch2 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));
        end;

        if ( (StrToInt(EdM.Text) > 0) and (StrToInt(EdN.Text) > 0) ) then
        begin
          NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
        end;

        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      5:
      begin
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        Start1 := NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdM.Text), FeatureSize, 0, 0) );

        NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
        Start2 := NN.AddLayer( TNNetConvolutionReLU.Create(StrToInt(EdN.Text), FeatureSize, 0, 0) );

        // Branch1
        NN.AddLayerAfter( TNNetMaxPool.Create(MaxPool), Start1 );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount div 2, iInnerConvFeatureSize, iPadding, 0) );
        Branch1 := NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount div 2, iInnerConvFeatureSize, iPadding, 0) );

        // Branch2
        NN.AddLayerAfter(TNNetMaxPool.Create(MaxPool), Start2 );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount div 2, iInnerConvFeatureSize, iPadding, 0) );
        Branch2 := NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount div 2, iInnerConvFeatureSize, iPadding, 0) );

        // Concats both branches
        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      6:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch1 := NN.AddLayer(TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0));

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch2 := NN.AddLayer(TNNetConvolutionReLU.Create(22, iInnerConvFeatureSize, iPadding, 0));

        // Branch3
        NN.AddLayerAfter( TNNetSplitChannels.Create([2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0) );
        NN.AddLayer( TNNetMaxPool.Create(4) );
        Branch3 := NN.AddLayer( TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2, Branch3]) );
        NN.AddLayer( TNNetLayerStdNormalization.Create() );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      7:
      begin
        // Branch1 (default is 5x5 features)
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch1 := NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize, 0, 0));

        // Branch2 (default is 3x3 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize-2, 0, 0),InputLayer);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize-2, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch2 := NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize-2, 0, 0));

        // Branch3 (default is 7x7 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize+2, 0, 0),InputLayer);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize+2, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch3 := NN.AddLayer(TNNetConvolutionReLU.Create(iConvNeuronCount, FeatureSize+2, 0, 0));

        // Concats both branches so the NN has only one end.
        NN.AddLayer(TNNetConcat.Create([Branch1,Branch2,Branch3]));

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      8:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(64, FeatureSize, 0, 0) );
        Branch1 := NN.AddLayer( TNNetMaxPool.Create(MaxPool) );

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(64, FeatureSize, 0, 0) );
        Branch2 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));

        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
    end;

    if (ComboLastLayer.ItemIndex = 0) then
    begin
      NN.AddLayer( TNNetFullConnectLinear.Create(FTrainingImageClassCount) );
      NN.AddLayer( TNNetSoftMax.Create() );
    end
    else
    begin
      NN.AddLayer( TNNetFullConnectReLU.Create(FTrainingImageClassCount) );
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

  FThreadNN := TNNetDataParallelism.Create(NN, FThreadNum);

  WriteLn('Setting L2 to:', fL2Decay:6:4, '  Learning rate:',CurrentLearningRate:6:4,'  Staircase ephocs:',iStaircaseEpochs);
  FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
  FThreadNN.SetBatchUpdate(true);

  if RadL2All.Checked
  then NN.SetL2Decay(fL2Decay)
  else NN.SetL2DecayToConvolutionalLayers(fL2Decay);
  NN.SetLearningRate(CurrentLearningRate, fInertia);

  // in batch update, threaded NN should not apply L2.
  FThreadNN.SetL2DecayToConvolutionalLayers(0);

  //--------------------------------------------------------------------
  pOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);
  vOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);
  vDisplay:= TNNetVolume.Create(FTrainingImageClassCount,1,1);
  firstNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx();

  if ( iAlgo = 6 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    thirdNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(secondNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 16, 16, FeaturesNum*3);
  end
  else if ( ( iAlgo = 4 ) or (iAlgo = 5) ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 32, 8, 64);
  end
  else if (iAlgo = 8) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 16, 16, 64);
  end
  else if (iAlgo = 3) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8, 64);
  end
  else if (FeaturesNum>64) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 16, 16, FeaturesNum);
  end
  else
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8, FeaturesNum);
  end;

  globalImgInput.RgbImgToNeuronalInput(color_encoding);
  DisplayInputImage(color_encoding);
  Application.ProcessMessages;

  NN.DebugWeights();
  WriteLn('Neural network has: ');
  NN.DebugStructure();
  LabCounts.Caption :=
    'Layers: '+IntToStr(NN.CountLayers())+
    '  Neurons: '+IntToStr(NN.CountNeurons())+
    '  Weights: '+IntToStr(NN.CountWeights());
  WriteLn('Computing...');

  globalStartTime := Now();
  while ( FRunning and (iMaxEpochs > globalLoopCnt) ) do
  begin
    globalErrorSum := 0;
    startTime := Now();

    for I := 1 to (FTrainingImageCount div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;

      NN.ClearTime();
      ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
      NN.UpdateWeights();
      NN.ComputeL2Decay();

      if (globalHit > 0) then
      begin
        CurrentRate := globalHit / (globalHit + globalMiss);
        CVSError := globalErrorSum / (globalHit + globalMiss);
        CVSLoss  := (globalTotalLoss / (globalHit + globalMiss) );
        CVSRate  := CurrentRate;

        if ( bAutoNoisy and (CVSValidationRate > 0) )then
        begin
          // is the NN 10% overfit?
          if (CVSRate > CVSValidationRate + 0.1) then
          begin
            bDataAugmentation := true;
            //Writeln('Auto augmentation is on -> ', CVSRate:4:2,' : ',CVSValidationRate:4:2);
          end
          else
          begin
            bDataAugmentation := false;
          end;
        end;

        if (FStepSize < 500) then
        begin
          CurrentAccuracy := (globalHit*100) div (globalHit+globalMiss);
          AccuracyWithInertia := AccuracyWithInertia*0.9 + CurrentAccuracy*0.1;
        end
        else
        begin
          CurrentAccuracy := (globalHit*100) div (globalHit+globalMiss);
          AccuracyWithInertia := CurrentAccuracy;
        end;
      end;

      if ( (globalHit > 0) and (I mod csLogEveryCicles = 0) ) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;

        WriteLn
        (
          (globalHit + globalMiss)*I + globalLoopCnt*FTrainingImageCount,
          ' Examples seen. Accuracy:', CVSRate:6:4,
          ' Error:', CVSError:10:5,
          ' Loss:', CVSLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Thread Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Thread Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Time:', totalTimeSeconds:6:2,'s'
        );

        LabClassRate.Caption := PadLeft(IntToStr(Round(AccuracyWithInertia))+'%',4) ;

        LabTime.Caption :=
          'Epoch time: ' + FloatToStrF( totalTimeSeconds*(FTrainingImageCount/(FStepSize*csLogEveryCicles))/60,ffGeneral,1,4)+' minutes.' +
          ' 200 epochs: ' + FloatToStrF( 200*totalTimeSeconds*(FTrainingImageCount/(FStepSize*csLogEveryCicles))/3600,ffGeneral,1,4)+' hours.';

        LabTotalTime.Caption :=
          'Epochs: '+IntToStr(globalLoopCnt)+
          '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.';

        startTime := Now();
        Application.ProcessMessages;
      end;
    end;

    if (FRunning) then
    begin
      if (globalLoopCnt mod ProcThreadPool.MaxThreadCount = 0) then
      begin

        if ( iAlgo = 6 ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, FeaturesNum*0, 16, color_encoding);
          ShowNeuronsIndependentChannels(secondNeuronalLayer, FeaturesNum*1, 16, color_encoding);
          ShowNeuronsIndependentChannels(thirdNeuronalLayer, FeaturesNum*2, 16, color_encoding);
        end
        else if ( ( iAlgo = 4 ) or ( iAlgo = 5 ) ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 32, color_encoding);
          if ( (StrToInt(EdM.Text) > 0) and (StrToInt(EdN.Text) > 0) ) then
          begin
            ShowNeuronsIndependentChannels(secondNeuronalLayer, StrToInt(EdM.Text), 32, color_encoding);
          end;
        end
        else if ( iAlgo = 8 ) then
        begin
          ShowNeuronsIndependentChannels(firstNeuronalLayer, 0, 16, color_encoding);
          ShowNeuronsIndependentChannels(secondNeuronalLayer, 64, 16, color_encoding);
        end
        else
        begin
          if FeaturesNum>64 then
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

    Inc(globalLoopCnt);
    Inc(iLearningRateCounter);

    if (FRunning) then
    begin
      FWorkingImages := FValidationImages;

      WriteLn('Starting Validation.');

      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;

      ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

      CVSValidationRate  := globalHit / FValidationImageCount;
      CVSValidationLoss  := globalTotalLoss / FValidationImageCount;
      CVSValidationError := globalErrorSum / FValidationImageCount;

      LabTestRate.Caption := PadLeft(IntToStr( Round(CVSValidationRate*100) )+'%',4) ;

      if (globalLoopCnt mod ProcThreadPool.MaxThreadCount = 0) then
      begin
        DisplayInputImage(color_encoding);
        Application.ProcessMessages;
        NN.DebugWeights();
        NN.DebugErrors();
        NN.SaveToFile(fileName);
        SaveScreenshot(fileNameImage);
        SaveNeuronsImage(fileNameNeuronsImage);
        Application.ProcessMessages;
      end;

      if ( (globalLoopCnt mod 10 = 0) and (globalLoopCnt > 0) ) then
      begin
        FWorkingImages := FTestImages;
        globalHit       := 0;
        globalMiss      := 0;
        globalTotalLoss := 0;
        globalErrorSum  := 0;

        WriteLn('Starting Testing.');
        ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

        CVSTestRate  := globalHit / FTestImageCount;
        CVSTestLoss  := globalTotalLoss / FTestImageCount;
        CVSTestError := globalErrorSum / FTestImageCount;

        writeln
        (
          CSVFile,
          globalLoopCnt,',',
          CVSRate:6:4,',',
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
          globalLoopCnt,',',
          CVSRate:6:4,',',
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

      if CheckProportionalLearningRate.Checked then
      begin
        fNewLearningRate := (CurrentLearningRate * power(fLearningRateDecay,iStaircaseEpochs));
        if ( ( fNewLearningRate >= fMinLearnRate ) and (iLearningRateCounter>=iStaircaseEpochs) ) then
        begin
          iLearningRateCounter := 0;
          CurrentLearningRate := fNewLearningRate;
          FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
          NN.SetLearningRate(CurrentLearningRate, fInertia);
          NN.ClearInertia();
          WriteLn
          (
            'Learning dropped to:',CurrentLearningRate:7:5
          );
        end;
      end
      else
      begin
        if (CurrentLearningRate * fLearningRateDecay > fMinLearnRate) then
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

      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

      if (globalHit > 0) then
      begin
        WriteLn
        (
          'Epochs: ',globalLoopCnt,
          ' Examples seen:', globalLoopCnt * FTrainingImageCount,
          ' Accuracy:', CVSRate:6:4,' ', CVSValidationRate:6:4,
          ' Error: ', CVSError:10:5,' ',CVSTestError:10:5,
          ' Loss: ', CVSLoss:7:5,' ',CVSTestLoss:7:5,
          ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
        );

      end;
      Application.ProcessMessages();
    end;
  end;

  FTrainImages.Clear();
  FTestImages.Clear();
  FValidationImages.Clear();

  FreeNeuronImages();

  FThreadNN.Free;
  CloseFile(CSVFile);
  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  LocalPicture.Free;
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

  if ( (ComboAlgo.ItemIndex <> 0) and flag) then
  begin
    EdMaxPool.Enabled := false;
    EdConvLayers.Enabled := false;
    EdFCLayers.Enabled := false;
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
  ImgInput, ImgInputCp2: TNNetVolume;
  pOutput, vOutput: TNNetVolume;
  I: integer;
  ImgShiftSize, ImgShiftCnt, ImgShiftCntD: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  ClassId, ImageId: integer;
  LocalPicture: TPicture;
  SourceVolume: TNNetVolume;
begin
  LocalPicture := TPicture.Create;
  ImgInput := TNNetVolume.Create();
  ImgInputCp2 := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);
  vOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);

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
  for I := 1 to BlockSize do
  begin
    if not(FRunning) then Break;

    ClassId := FTrainImages.GetRandomClassId();
    ImageId := FTrainImages.List[ClassId].GetRandomIndex();

    SourceVolume := FTrainImages.List[ClassId].List[ImageId];

    if SourceVolume.Size < 2 then
    begin
      (*if I=1 then*) LocalPicture.LoadFromFile( FTrainImages.GetFileName(ClassId, ImageId) );
      (*if I=1 then*) LoadPictureIntoVolume(LocalPicture, SourceVolume);
      SourceVolume.Tag := ClassId;
      SourceVolume.RgbImgToNeuronalInput(color_encoding);
    end;

    //WriteLn(ClassId,':',ImageId,' ',ImgInputCp.SizeX,':',ImgInputCp.SizeY);

    CropSizeX := random(9);
    CropSizeY := random(9);

    ImgInputCp2.CopyCropping(SourceVolume, random(CropSizeX), random(CropSizeY),SourceVolume.SizeX-CropSizeX, SourceVolume.SizeY-CropSizeY);
    ImgInput.CopyResizing(ImgInputCp2, SourceVolume.SizeX, SourceVolume.SizeY);
    ImgInput.Tag := ClassId;

    // flip is always used in training
    if Random(1000) > 500 then
    begin
      ImgInput.FlipX();
    end;

    (*
    // Random "add" is always used in training
    ImgInput.Add( ( (Random(1024)-512)*FNoiseLevel) / 2560 );
    *)


    if bDataAugmentation then
    begin
      if ( (FDataAugmentationType = 1) or (FDataAugmentationType = 7) ) then
      begin
        // Modifies the image
        ImgInputCp2.Copy(ImgInput);
        ImgShiftSize := Round ( Random(1280)*FNoiseLevel ) * ImgInput.Depth;
        if (ImgShiftSize > 0) then
        begin
          ImgInput.ShiftRight( ImgShiftSize ); //Shifts input image right and down

          ImgShiftCntD := 0;
          for ImgShiftCnt := ImgInput.Size - ImgShiftSize to ImgInput.Size - 1 do
          begin
            ImgInput.FData[ImgShiftCntD] := ImgInputCp2.FData[ImgShiftCnt];
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
    LocalNN.GetOutput(pOutput);

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( ImgInput.Tag );
    end
    else
    begin
      vOutput.SetClassForReLU( ImgInput.Tag );
    end;

    LocalErrorSum += vOutput.SumDiff(pOutput);

    if (bIsSoftmax) then
    begin
        OutputValue := pOutput.FData[ ImgInput.Tag ];
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
    end;

    if ( (globalLoopCnt>0) or not(bLoadedFile) ) then
    begin
      // we don't backpropagate when we are too sure.
      // if ( (CurrentLoss>=0.1) or not(bIsSoftmax) ) then
      begin
        LocalNN.Backpropagate(vOutput);
      end;
    end;

    if pOutput.GetClass() = ImgInput.Tag then
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

  EnterCriticalSection(FCritSec);
  globalHit       += LocalHit;
  globalMiss      += LocalMiss;
  globalTotalLoss += LocalTotalLoss;
  globalErrorSum  += LocalErrorSum;

  NN.ForwardTime := NN.ForwardTime + LocalNN.ForwardTime;
  NN.BackwardTime := NN.BackwardTime + LocalNN.BackwardTime;

  LeaveCriticalSection(FCritSec);
  {$IFDEF Debug}
  NN.SumDeltas(LocalNN);
  {$ELSE}
  NN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}

  LocalNN.ClearDeltas();

  ImgInputCp2.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
  LocalPicture.Free;
end;

procedure TFormVisualLearning.TestNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize: integer;
  LocalNN: TNNet;
  pOutput, vOutput: TNNetVolume;
  I, ImgIdx: integer;
  StartPos, FinishPos: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  LocalWorkingCount: integer;
  ClassId, ImageId: integer;
  LocalPicture: TPicture;
  SourceVolume: TNNetVolume;
begin
  LocalPicture := TPicture.Create;
  pOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);
  vOutput := TNNetVolume.Create(FTrainingImageClassCount,1,1);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;
  LocalWorkingCount := FWorkingImages.CountElements();

  BlockSize := (LocalWorkingCount div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(NN);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if not(FRunning) then Break;
    ImgIdx := I;

    ClassId := ImgIdx mod FWorkingImages.Count;
    ImageId := ImgIdx div FWorkingImages.Count;

    SourceVolume := FTrainImages.List[ClassId].List[ImageId];

    if SourceVolume.Size < 2 then
    begin
      (*if I=1 then*) LocalPicture.LoadFromFile( FTrainImages.GetFileName(ClassId, ImageId) );
      (*if I=1 then*) LoadPictureIntoVolume(LocalPicture, SourceVolume);
      SourceVolume.Tag := ClassId;
      SourceVolume.RgbImgToNeuronalInput(color_encoding);
    end;

    LocalNN.Compute( SourceVolume );
    LocalNN.GetOutput( pOutput );

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( SourceVolume.Tag );
    end
    else
    begin
      vOutput.SetClassForReLU( SourceVolume.Tag );
    end;

    LocalErrorSum += vOutput.SumDiff(pOutput);

    if (bIsSoftmax) then
    begin
        OutputValue := pOutput.FData[ SourceVolume.Tag ];
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
    end;

    if pOutput.GetClass() = SourceVolume.Tag then
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

  LocalPicture.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.

