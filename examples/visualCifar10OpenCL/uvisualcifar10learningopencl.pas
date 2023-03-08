unit uvisualcifar10learningopencl;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Menus, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume, MTProcs, math, neuralopencl, neuralopenclv;

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ButProbe: TButton;
    ChkRunOnCPU: TCheckBox;
    ChkCrop: TCheckBox;
    ComboAlgo: TComboBox;
    ComboColor: TComboBox;
    ComboDevType: TComboBox;
    ComboLastLayer: TComboBox;
    ComboPlatform: TComboBox;
    EdClipDelta: TEdit;
    EdInertia: TEdit;
    EdFeatures: TEdit;
    EdFeatureSize: TEdit;
    EdFCLayers: TEdit;
    EdConvLayers: TEdit;
    EdConvPadding: TEdit;
    EdInnerConvFeatureSize: TEdit;
    EdFCNeuronCount: TEdit;
    EdBatchSize: TEdit;
    EdInnerConvNeuronCount: TEdit;
    EdAutosaveName: TEdit;
    EdLearnRateDecay: TEdit;
    EdPhysThreads: TEdit;
    EdThreadCount: TEdit;
    EdL2Decay: TEdit;
    EdMinLearnRate: TEdit;
    EdMaxEpochs: TEdit;
    EdStaircaseEpochs: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    ImgSample: TImage;
    LabAlgo: TLabel;
    LabClipDelta: TLabel;
    Label1: TLabel;
    LabColor: TLabel;
    LabCounts: TLabel;
    LabBatchSize: TLabel;
    LabComputedL2: TLabel;
    LabConvNeurons: TLabel;
    LabAutosaveName: TLabel;
    LabelDevType: TLabel;
    LabPlatform: TLabel;
    LabStaircase: TLabel;
    LabFCNCount: TLabel;
    LabInnerConvFeatureSize: TLabel;
    LabPadding: TLabel;
    LabLearnDecay: TLabel;
    LabPhysThreads: TLabel;
    LabTotalTime: TLabel;
    LabL2Decay: TLabel;
    LabTime: TLabel;
    LabTestRate: TLabel;
    LabConv: TLabel;
    LabClassRate: TLabel;
    LabLastLayer: TLabel;
    LabMinLearning: TLabel;
    LabMaxEpochs: TLabel;
    LabMaxPool: TLabel;
    LabHiddenNum: TLabel;
    LabFeatureSize: TLabel;
    LabNFeatures: TLabel;
    LabInertia: TLabel;
    LabLearning: TLabel;
    procedure ButLearnClick(Sender: TObject);
    procedure ButProbeClick(Sender: TObject);
    procedure ComboAlgoChange(Sender: TObject);
    procedure ComboDevTypeChange(Sender: TObject);
    procedure ComboPlatformChange(Sender: TObject);
    procedure EdLearningRateChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure GroupBox1Click(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FImgCrop: boolean;
    FThreadNum, FStepSize: integer;
    FThreadNN: TNNetDataParallelism;
    ImgVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    ImgWorkingVolumes: TNNetVolumeList;
    globalImgInput: TNNetVolume;
    FFinishedThread: TNNetVolume;
    globalHit, globalMiss: integer;
    globalErrorSum: TNeuralFloat;
    globalTotalLoss: TNeuralFloat;
    color_encoding: integer;

    bIsSoftmax: boolean;

    iEpochCount: integer;
    NN: THistoricalNets;

    FCritSec: TRTLCriticalSection;

    FOpenCL: TEasyOpenCLCL;

    procedure LoadCifar10binFilesIfRequired();
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
    procedure SaveScreenshot(filename: string);
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
  MaxCrop = 2;
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

procedure TFormVisualLearning.ButProbeClick(Sender: TObject);
begin
  FOpenCL.printDevicesInfo();
  if not FOpenCL.ProbeClick(ComboPlatform, ComboDevType) then
  begin
    ChkRunOnCPU.Checked := true;
  end;
  ButLearn.Enabled := True;
end;

procedure TFormVisualLearning.ComboAlgoChange(Sender: TObject);
begin
end;

procedure TFormVisualLearning.ComboDevTypeChange(Sender: TObject);
begin
  FOpenCL.ComboDevTypeChange(ComboDevType);
end;

procedure TFormVisualLearning.ComboPlatformChange(Sender: TObject);
begin
  FOpenCL.ComboPlatformChange(ComboPlatform, ComboDevType);
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
  ImgVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  FFinishedThread := TNNetVolume.Create();
  EdThreadCount.Text := IntToStr(Min(32,ProcThreadPool.MaxThreadCount * 2));
  EdPhysThreads.Text := IntToStr(Min(32,ProcThreadPool.MaxThreadCount * 2));
  globalImgInput := TNNetVolume.Create();
  InitCriticalSection(FCritSec);
  FOpenCL := TEasyOpenCLCL.Create();
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  FFinishedThread.Free;
  FOpenCL.Free;
  DoneCriticalSection(FCritSec);
  globalImgInput.Free;
  ImgValidationVolumes.Free;
  ImgTestVolumes.Free;
  ImgVolumes.Free;
end;

procedure TFormVisualLearning.GroupBox1Click(Sender: TObject);
begin

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

procedure TFormVisualLearning.Learn(Sender: TObject);
var
  aImage: array of TImage;
  vDisplay: TNNetVolume;

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

  procedure CreateNeuronImages(firstNeuronalLayer, filterSize, imagesPerRow: integer);
  var
    NeuronCount: integer;
  begin
    SetLength(aImage, NN.Layers[firstNeuronalLayer].Neurons.Count * 3);

    for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count * 3 - 1 do
    begin
      aImage[NeuronCount] := TImage.Create(FormVisualLearning);
      aImage[NeuronCount].Parent := FormVisualLearning;
      aImage[NeuronCount].Width  := NN.Layers[firstNeuronalLayer].Neurons[0].Weights.SizeX;
      aImage[NeuronCount].Height := NN.Layers[firstNeuronalLayer].Neurons[0].Weights.SizeY;
      aImage[NeuronCount].Top    := (NeuronCount div imagesPerRow) * (filterSize+4) + 212;
      aImage[NeuronCount].Left   := (NeuronCount mod imagesPerRow) * (filterSize+4) + 16;
      aImage[NeuronCount].Stretch:=true;
    end;
  end;

  procedure FreeNeuronImages();
  var
    NeuronCount: integer;
  begin
    for NeuronCount := Low(aImage) to High(aImage) do
    begin
      aImage[NeuronCount].Free;
    end;
    SetLength(aImage, 0);
  end;

var
  I: integer;
  pOutput, vOutput: TNNetVolume;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  CurrentRate: double;
  CurrentLearningRate, fInitialLearningRate, fInertia: single;
  MaxPool,Stride: integer;
  FeaturesNum, FeatureSize: integer;
  InnerConvNum, InnerConvCnt: integer;
  InnerFCNum, InnerFCCnt: integer;
  fileName, fileNameBase, FileNameCSV, fileNameImage: string;
  firstNeuronalLayer, secondNeuronalLayer, thirdNeuronalLayer: integer;
  iMaxEpochs, iStaircaseEpochs: integer;
  iFCNeuronCount, iConvNeuronCount: integer;
  iLearningRateCounter: integer;
  fMinLearnRate, fNewLearningRate: single;
  fLearningRateDecay: single;
  CVSError, CVSLoss, CVSRate,
  CVSTestError, CVSTestLoss, CVSTestRate: TNeuralFloat;
  CVSValidationError, CVSValidationLoss, CVSValidationRate: TNeuralFloat;
  CSVFile: TextFile;
  fL2Decay: TNeuralFloat;
  iAlgo: integer;
  iPadding: integer;
  iInnerConvFeatureSize: integer;
  iInputDepth: integer;
  InputLayer, Branch1, Branch2, Branch3, PlusOne, EnhancedInput: TNNetLayer;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  MaxDelta, ClipDelta: TNeuralFloat;

  procedure CheckLearningRate();
  var
    iStairCount: integer;
  begin
    iStairCount := iEpochCount div iStaircaseEpochs;
    fNewLearningRate := (fInitialLearningRate * power(fLearningRateDecay,iStaircaseEpochs*iStairCount));
    if ( ( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate < CurrentLearningRate) ) then
    begin
        CurrentLearningRate := fNewLearningRate;
        FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        NN.ClearInertia();
        WriteLn
        (
          'Learning dropped to:',CurrentLearningRate:7:5
        );
    end;
  end;

begin
  AccuracyWithInertia := 10;
  FThreadNum := StrToInt(EdThreadCount.Text);
  ProcThreadPool.MaxThreadCount := StrToInt(EdPhysThreads.Text);

  FStepSize := StrToInt(EdBatchSize.Text);
  ClipDelta := StrToFloat(EdClipDelta.Text);

  iEpochCount := 0;
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
  iPadding := StrToInt(EdConvPadding.Text);
  iInnerConvFeatureSize := StrToInt(EdInnerConvFeatureSize.Text);

  WriteLn('Number of threads:', FThreadNum);
  WriteLn('Algorithm:',iAlgo,' Color Encoding:', color_encoding,' Input Channels:', iInputDepth, ' Step Size:', FStepSize);

  if iAlgo = 0 then
  begin
    fileNameBase :=
      EdAutosaveName.Text+'-'+
      EdLearningRate.Text+'-'+
      EdLearnRateDecay.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdMaxPool.Text+'-'+
      EdInnerConvFeatureSize.Text+'-'+
      EdConvPadding.Text+'-'+
      EdConvLayers.Text+'-'+
      EdInnerConvNeuronCount.Text+'-'+
      EdFCLayers.Text+'-'+
      EdFCNeuronCount.Text+'-'+
      BoolToStr(ChkCrop.Checked,'T','F')+'-'+
      EdThreadCount.Text+'-'+
      EdBatchSize.Text+'-'+
      ComboColor.Text+'-'+
      ComboLastLayer.Text;
  end
  else
  begin
    fileNameBase :=
      EdAutosaveName.Text+'-algo'+
      IntToStr(iAlgo)+'-'+
      EdLearningRate.Text+'-'+
      EdLearnRateDecay.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdMaxPool.Text+'-'+
      EdInnerConvFeatureSize.Text+'-'+
      EdConvPadding.Text+'-'+
      BoolToStr(ChkCrop.Checked,'T','F')+'-'+
      EdThreadCount.Text+'-'+
      EdBatchSize.Text+'-'+
      ComboColor.Text+'-'+
      ComboLastLayer.Text;
  end;

  WriteLn('File name is: ',fileNameBase);

  FileNameCSV := fileNameBase + '.csv';

  FileName := fileNameBase + '.nn';

  fileNameImage := fileNameBase + '.bmp';

  AssignFile(CSVFile, FileNameCSV);

  LoadCifar10binFilesIfRequired();

  loadCifar10Dataset(ImgVolumes, 1, 0, color_encoding);
  loadCifar10Dataset(ImgVolumes, 2, 10000, color_encoding);
  loadCifar10Dataset(ImgVolumes, 3, 20000, color_encoding);
  loadCifar10Dataset(ImgVolumes, 4, 30000, color_encoding);
  loadCifar10Dataset(ImgValidationVolumes, 5, 0, color_encoding);
  loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0, color_encoding);

  WriteLn('Training Images:', ImgVolumes.Count, ' Test Images:', ImgTestVolumes.Count, ' Validation Images:', ImgValidationVolumes.Count);

  FeaturesNum := StrToInt(EdFeatures.Text);
  FeatureSize := StrToInt(EdFeatureSize.Text);
  FImgCrop := ChkCrop.Checked;
  fInitialLearningRate := StrToFloat(EdLearningRate.Text);
  fLearningRateDecay := 1 - StrToFloat(EdLearnRateDecay.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fInitialLearningRate;
  MaxPool := StrToInt(EdMaxPool.Text);
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

  writeln('Creating Neural Network...');
  NN := THistoricalNets.Create();

  if ( FileExists(fileName) and FileExists(FileNameCSV) ) then
  begin
    writeln('Loading neural network from file: ',fileName);
    Reset(CSVFile);
    while not Eof(CSVFile) do
    begin
      ReadLn(CSVFile);
      Inc(iEpochCount);
    end;
    if (iEpochCount > 0) then iEpochCount := iEpochCount - 1;
    CloseFile(CSVFile);
    WriteLn('Precomputed epochs: ', iEpochCount);
    Append(CSVFile);
    NN.LoadFromFile(fileName);
    // as this is a loaded NN, we'll start measuring and not learning
    NN.SetLearningRate(0, fInertia);
  end
  else
  begin
    ReWrite(CSVFile);
    writeln(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');

    if (FImgCrop) then
    begin
      InputLayer := NN.AddLayer( TNNetInput.Create(24,24,iInputDepth) );
    end
    else
    begin
      InputLayer := NN.AddLayer( TNNetInput.Create(32,32,iInputDepth) );
    end;

    if ( (iAlgo<>9) and (iAlgo<>8) and (iAlgo<>6) and (iAlgo<>5) and (iAlgo<>4) and (iAlgo<>3) ) then
    begin
      NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,FeatureSize,(FeatureSize-1) div 2, 1) );
    end;

    InnerConvNum := StrToInt(EdConvLayers.Text);
    InnerFCNum   := StrToInt(EdFCLayers.Text);

      case iAlgo of
      0:
      begin
        if (InnerConvNum >= 2) then
        begin
          for InnerConvCnt := 2 to InnerConvNum do
          begin
            if InnerConvCnt = InnerConvNum
            then NN.AddLayer( TNNetConvolutionLinear.Create(FeaturesNum, 3, 1, 1) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum, 3, 1, 1) );
          end;
        end;

        if MaxPool > 0 then
        begin
          NN.AddLayer( TNNetMaxPool.Create(MaxPool) );
        end;

        if (InnerConvNum > 0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            if InnerConvCnt = InnerConvNum
              then NN.AddLayer( TNNetConvolutionLinear.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) )
              else NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) );
          end;
          if MaxPool > 0 then
          begin
            NN.AddLayer( TNNetMaxPool.Create(MaxPool) );
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
      1:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
      end;
      2:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
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
      end;
      4:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(22, FeatureSize, (FeatureSize-1) div 2, 1) );

        if (InnerConvNum >= 2) then
        begin
          for InnerConvCnt := 2 to InnerConvNum do
          begin
            if InnerConvCnt = InnerConvNum
            then NN.AddLayer( TNNetConvolutionLinear.Create(22, 3, 1, 1) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(22, 3, 1, 1) );
          end;
        end;

        Branch1 := NN.AddLayer( TNNetMaxPool.Create(MaxPool) );

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(42, FeatureSize, (FeatureSize-1) div 2, 1) );

        if (InnerConvNum >= 2) then
        begin
          for InnerConvCnt := 2 to InnerConvNum do
          begin
            if InnerConvCnt = InnerConvNum
            then NN.AddLayer( TNNetConvolutionLinear.Create(42, 3, 1, 1) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(42, 3, 1, 1) );
          end;
        end;

        Branch2 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));

        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );

        for InnerConvCnt := 1 to InnerConvNum do
        begin
          if InnerConvCnt = InnerConvNum
            then NN.AddLayer( TNNetConvolutionLinear.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) )
            else NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount,iInnerConvFeatureSize,iPadding,0) );
        end;

        if MaxPool > 0 then
        begin
          NN.AddLayer( TNNetMaxPool.Create(MaxPool) );
        end;

        if (InnerFCNum > 0) then
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
        NN.AddLayer( TNNetConvolutionReLU.Create(42, FeatureSize, 0, 0) );
        Branch1 := NN.AddLayer( TNNetMaxPool.Create(MaxPool) );

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1,2]), InputLayer);
        NN.AddLayer( TNNetConvolutionReLU.Create(22, FeatureSize, 0, 0) );
        Branch2 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));

        NN.AddLayer( TNNetDeepConcat.Create([Branch1, Branch2]) );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );
        NN.AddLayer( TNNetConvolutionReLU.Create(iConvNeuronCount, iInnerConvFeatureSize, iPadding, 0) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
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

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
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

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
      end;
      9:
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
    end;

    if (ComboLastLayer.ItemIndex = 0) then
    begin
      NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
      NN.AddLayer( TNNetSoftMax.Create() );
    end
    else
    begin
      NN.AddLayer( TNNetFullConnectReLU.Create(NumClasses) );
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

  WriteLn('Setting L2 to:', fL2Decay:6:4, '  Learning rate:',CurrentLearningRate:6:4,'  Staircase ephocs:',iStaircaseEpochs);
  FThreadNN.SetLearningRate(CurrentLearningRate, fInertia);
  FThreadNN.SetBatchUpdate(true);

  NN.SetL2Decay(fL2Decay);
  NN.SetLearningRate(CurrentLearningRate, fInertia);

  // in batch update, threaded NN should not apply L2.
  FThreadNN.SetL2Decay(0);

  {$IFDEF OpenCL}
  if not ChkRunOnCPU.Checked then
  begin
    FThreadNN.EnableOpenCL(FOpenCL.CurrentPlatform, FOpenCL.CurrentDevice);
  end;
  {$ENDIF}

  //--------------------------------------------------------------------
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  vDisplay:= TNNetVolume.Create(NumClasses,1,1);
  firstNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx();

  if ( iAlgo = 6 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    thirdNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(secondNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 16, 16);
  end
  else if ( ( iAlgo = 4 ) or (iAlgo = 5) ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 32, 8);
  end
  else if (iAlgo = 8) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 16, 16);
  end
  else if (FeaturesNum>64) then
  begin
    CreateNeuronImages(firstNeuronalLayer, 16, 16);
  end
  else
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8);
  end;

  //globalImgInput.Copy(ImgVolumes[0]);
  pOutput.CopyCropping(ImgVolumes[0], random(8), random(8), 24, 24);
  globalImgInput.CopyResizing(pOutput, 32, 32);
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
  Application.ProcessMessages;
  while ( FRunning and (iMaxEpochs > iEpochCount) ) do
  begin
    globalErrorSum := 0;
    CheckLearningRate();
    startTime := Now();

    for I := 1 to (ImgVolumes.Count div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;

      FFinishedThread.Fill(0);
      NN.ClearTime();
      ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
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
        CVSError := globalErrorSum / (globalHit+globalMiss);
        CVSLoss  := (globalTotalLoss / (globalHit+globalMiss) );
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

        LabClassRate.Caption := PadLeft(IntToStr(Round(AccuracyWithInertia))+'%',4) ;

        LabTime.Caption :=
          'Epoch time: ' + FloatToStrF( totalTimeSeconds*(50000/(FStepSize*10))/60,ffGeneral,1,4)+' minutes.' +
          ' 200 epochs: ' + FloatToStrF( 200*totalTimeSeconds*(50000/(FStepSize*10))/3600,ffGeneral,1,4)+' hours.';

        LabTotalTime.Caption :=
          'Epochs: '+IntToStr(iEpochCount)+
          '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.';

        startTime := Now();
        Application.ProcessMessages;
      end;
    end;

    if (FRunning) then
    begin
      if (iEpochCount mod FThreadNum = 0) then
      begin
        if ( iAlgo = 6 ) then
        begin
          ShowNeurons(firstNeuronalLayer, FeaturesNum*0, 16, color_encoding);
          ShowNeurons(secondNeuronalLayer, FeaturesNum*1, 16, color_encoding);
          ShowNeurons(thirdNeuronalLayer, FeaturesNum*2, 16, color_encoding);
        end
        else if ( iAlgo = 4 ) then
        begin
          ShowNeurons(firstNeuronalLayer, 0, 32, color_encoding);
          ShowNeurons(secondNeuronalLayer, 22, 32, color_encoding);
        end
        else if ( iAlgo = 5 ) then
        begin
          ShowNeurons(firstNeuronalLayer, 0, 32, color_encoding);
          ShowNeurons(secondNeuronalLayer, 42, 32, color_encoding);
        end
        else if ( iAlgo = 8 ) then
        begin
          ShowNeurons(firstNeuronalLayer, 0, 16, color_encoding);
          ShowNeurons(secondNeuronalLayer, 64, 16, color_encoding);
        end
        else
        begin
          if FeaturesNum>64 then
          begin
            ShowNeurons(firstNeuronalLayer, FeaturesNum*0, 16, color_encoding);
          end
          else
          begin
            ShowNeurons(firstNeuronalLayer, FeaturesNum*0, 32, color_encoding);
          end;
        end;
      end;
    end;

    Inc(iEpochCount);
    Inc(iLearningRateCounter);

    if (FRunning) then
    begin
      ImgWorkingVolumes := ImgValidationVolumes;

      WriteLn('Starting Validation.');

      globalHit       := 0;
      globalMiss      := 0;
      globalTotalLoss := 0;
      globalErrorSum  := 0;

      ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

      CVSValidationRate  := globalHit / ImgWorkingVolumes.Count;
      CVSValidationLoss  := globalTotalLoss / ImgWorkingVolumes.Count;
      CVSValidationError := globalErrorSum / ImgWorkingVolumes.Count;

      LabTestRate.Caption := PadLeft(IntToStr( Round(CVSValidationRate*100) )+'%',4) ;

      if (iEpochCount mod FThreadNum = 0) then
      begin
        DisplayInputImage(color_encoding);
        Application.ProcessMessages;
        NN.DebugWeights();
        NN.DebugErrors();
        NN.SaveToFile(fileName);
        SaveScreenshot(fileNameImage);
        Application.ProcessMessages;
      end;

      if ( (iEpochCount mod 10 = 0) and (iEpochCount > 0) ) then
      begin
        ImgWorkingVolumes := ImgTestVolumes;
        globalHit       := 0;
        globalMiss      := 0;
        globalTotalLoss := 0;
        globalErrorSum  := 0;

        WriteLn('Starting Testing.');
        ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);

        CVSTestRate  := globalHit / ImgWorkingVolumes.Count;
        CVSTestLoss  := globalTotalLoss / ImgWorkingVolumes.Count;
        CVSTestError := globalErrorSum / ImgWorkingVolumes.Count;

        writeln
        (
          CSVFile,
          iEpochCount,',',
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
          iEpochCount,',',
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

      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

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

  FThreadNN.Free;
  CloseFile(CSVFile);
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
var
  MyBitmap: TBitmap;
  MyDC: HDC;
begin
  MyDC := GetDC(Self.Handle);
  MyBitmap := TBitmap.Create;
  try
    MyBitmap.LoadFromDevice(MyDC);
    Write(' Saving ',filename,':');
    MyBitmap.SaveToFile(filename);
    WriteLn('OK');
  finally
    ReleaseDC(Self.Handle, MyDC);
    FreeAndNil(MyBitmap);
  end;
end;

procedure TFormVisualLearning.RunNNThread(Index: PtrInt; Data: Pointer;
  Item: TMultiThreadProcItem);
var
  BlockSize, BlockSizeRest, CropSizeX, CropSizeY: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput: TNNetVolume;
  I, ImgIdx: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);

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
  for I := 1 to BlockSize do
  begin
    if not(FRunning) then Break;
    ImgIdx := Random(ImgVolumes.Count);

    if FImgCrop then
    begin
      ImgInput.CopyCropping(ImgVolumes[ImgIdx], random(8), random(8), 24, 24);
    end
    else
    begin
      CropSizeX := random(MaxCrop + 1);
      CropSizeY := random(MaxCrop + 1);

      ImgInputCp.CopyCropping(ImgVolumes[ImgIdx], random(CropSizeX), random(CropSizeY),ImgVolumes[ImgIdx].SizeX-CropSizeX, ImgVolumes[ImgIdx].SizeY-CropSizeY);
      ImgInput.CopyResizing(ImgInputCp, ImgVolumes[ImgIdx].SizeX, ImgVolumes[ImgIdx].SizeY);
    end;

    // flip is always used in training
    if Random(1000) > 500 then
    begin
      ImgInput.FlipX();
    end;

    // Makes 25% of images gray.
    if (ImgInput.Depth > 1) then
    begin
      if (Random(1000) > 750) then
      begin
        ImgInput.MakeGray(color_encoding);
      end;
    end;

    ImgInput.Tag := ImgVolumes[ImgIdx].Tag;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );

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

    LocalNN.Backpropagate(vOutput);

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
    end;
  end;
  FFinishedThread.FData[Index] := 1;
  if Index and 3 = 0 then
  begin
    if Index + 2 < FThreadNum then
    begin
      while FFinishedThread.FData[Index + 2] = 0 do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 2]);
    end;
  end;

  EnterCriticalSection(FCritSec);
  globalHit       += LocalHit;
  globalMiss      += LocalMiss;
  globalTotalLoss += LocalTotalLoss;
  globalErrorSum  += LocalErrorSum;

  NN.ForwardTime := NN.ForwardTime + LocalNN.ForwardTime;
  NN.BackwardTime := NN.BackwardTime + LocalNN.BackwardTime;
  {$IFDEF Debug}
  if Index and 3 = 0 then NN.SumDeltas(LocalNN);
  {$ELSE}
  if Index and 3 = 0 then NN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}
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
  pOutput, vOutput: TNNetVolume;
  I, ImgIdx: integer;
  StartPos, FinishPos: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := (ImgWorkingVolumes.Count div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(NN);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if not(FRunning) then Break;
    ImgIdx := I;

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

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( ImgInput.Tag );
    end
    else
    begin
      vOutput.SetClassForReLU( ImgInput.Tag );
    end;

    LocalErrorSum += vOutput.SumDiff(pOutput);

    // Runs on the flipped image.
    ImgInput.FlipX();
    LocalNN.Compute( ImgInput );
    LocalNN.AddOutput( pOutput );

    if ImgInput.SizeX >= 32 then
    begin
      ImgInputCp.CopyCropping(ImgInput, MaxCrop div 2, MaxCrop div 2, ImgInput.SizeX - MaxCrop, ImgInput.SizeY - MaxCrop);
      ImgInput.CopyResizing(ImgInputCp, ImgInput.SizeX, ImgInput.SizeY);

      LocalNN.Compute( ImgInput );
      LocalNN.AddOutput( pOutput );

      pOutput.Divi(3);
    end
    else
    begin
      pOutput.Divi(2);
    end;

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

    if pOutput.GetClass() = ImgInput.Tag then
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

  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.

