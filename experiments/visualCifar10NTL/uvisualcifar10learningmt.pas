unit uvisualcifar10learningMT;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  {$IFDEF NeuralThreadLib}
  ntl,
  {$ENDIF}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, uconvolutionneuralnetwork, uvolumelcl, ucifar10,
  uvolume;

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
    EdAutoTrigger: TEdit;
    EdNoiseLevel: TEdit;
    EdLearnRateDecay: TEdit;
    EdThreadCount: TEdit;
    EdL2Decay: TEdit;
    EdMinLearnRate: TEdit;
    EdMaxEpochs: TEdit;
    EdStride: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    ImgSample: TImage;
    LabAlgo: TLabel;
    Label1: TLabel;
    LabDataAugType: TLabel;
    LabColor: TLabel;
    LabCounts: TLabel;
    LabAutoTrigger: TLabel;
    LabInnerConvFeatureSize: TLabel;
    LabPadding: TLabel;
    LabNoiseLevel: TLabel;
    LabLearnDecay: TLabel;
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
    procedure EdLearningRateChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure GroupBox1Click(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FThreadNum, FStepSize: integer;
    FNoiseLevel: TNeuralFloat;
    FDataAugmentationType: integer;
    ImgVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    ImgWorkingVolumes: TNNetVolumeList;
    globalImgInput: TNNetVolume;
    Hit, Miss, Loss, ErrorSum: TNNetVolume;

    bIsSoftmax: boolean;
    bDataAugmentation: boolean;
    bLoadedFile: boolean;

    TotalLoss: single;
    globalLoopCnt: integer;
    NN: TNNet;

    procedure LoadCifar10binFilesIfRequired();
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
    procedure SaveScreenshot(filename: string);
  public
    { public declarations }
    procedure RunNNThread();

    procedure TestNNThread();

    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType;

const NumClasses = 10;
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
  globalImgInput := TNNetVolume.Create();
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
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

  procedure ShowNeurons(firstNeuronalLayer, startImage, filterSize: integer);
  var
    NeuronCount: integer;
    MaxW, MinW: TNeuralFloat;
  begin
    for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do
    begin
      MaxW := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMax();
      MinW := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMin();
      vDisplay.Copy(NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights);
      vDisplay.Mul(256/(MaxW-MinW));
      vDisplay.Add(128);
      LoadVolumeIntoTImage(vDisplay, aImage[NeuronCount + startImage]);
      aImage[NeuronCount + startImage].Width := filterSize;
      aImage[NeuronCount + startImage].Height := filterSize;
    end;
    Application.ProcessMessages;
  end;

  procedure DisplayInputImage(color_encoding: integer);
  begin
        vDisplay.Copy(globalImgInput);
        if (color_encoding = csEncodeRGB) then
        begin
          vDisplay.Mul(64);
          vDisplay.Add(128);
        end
        else if (color_encoding = csEncodeHSV) then
        begin
          vDisplay.Add(2);
          vDisplay.Mul(0.25);
          vDisplay.HsvToRgb();
        end
        else if (color_encoding = csEncodeHSL) then
        begin
          vDisplay.Add(2);
          vDisplay.Mul(0.25);
          vDisplay.HslToRgb();
        end
        else if (color_encoding = csEncodeLAB) then
        begin
          vDisplay.Mul(10);
          vDisplay.LabToRgb();
        end
        else
        begin
          WriteLn('Bad color encoding:', color_encoding);
        end;

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
  CurrentLearningRate, fLearningRate, fInertia: single;
  MaxPool,Stride: integer;
  fDropout: single;
  FeaturesNum, FeatureSize: integer;
  InnerConvNum, InnerConvCnt: integer;
  InnerFCNum, InnerFCCnt: integer;
  fileName, fileNameBase, FileNameCSV, fileNameImage: string;
  firstNeuronalLayer, secondNeuronalLayer, thirdNeuronalLayer: integer;
  iMaxEpochs: integer;
  color_encoding: integer;
  fMinLearnRate, fNewLearningRate: single;
  fLearningRateDecay: single;
  bCenterDivide, bAutoNoisy: boolean;
  CVSError, CVSLoss, CVSRate,
  CVSTestError, CVSTestLoss, CVSTestRate: TNeuralFloat;
  CVSValidationError, CVSValidationLoss, CVSValidationRate: TNeuralFloat;
  CSVFile: TextFile;
  fL2Decay: TNeuralFloat;
  fAutoTrigger: TNeuralFloat;
  iAlgo: integer;
  iPadding: integer;
  iInnerConvFeatureSize: integer;
  InputLayer, Branch1, Branch2, Branch3: TNNetLayer;
  LocalErrorSum: TNeuralFloat;
  globalHit, globalMiss: integer;
begin
  FThreadNum := StrToInt(EdThreadCount.Text);
  {$IFDEF NeuralThreadLib}NeuronThreadListCreate(FThreadNum);{$ENDIF}
  FDataAugmentationType := ComboAugmentation.ItemIndex;
  FNoiseLevel := StrToFloat(EdNoiseLevel.Text);
  FStepSize := 1000;
  Hit := TNNetVolume.Create(1);
  Miss := TNNetVolume.Create(1);
  Loss := TNNetVolume.Create(1);
  ErrorSum := TNNetVolume.Create(1);
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

  color_encoding := ComboColor.ItemIndex;

  iAlgo := ComboAlgo.ItemIndex;
  iPadding := StrToInt(EdConvPadding.Text);
  iInnerConvFeatureSize := StrToInt(EdInnerConvFeatureSize.Text);

  WriteLn('Number of threads:', FThreadNum);
  WriteLn('Algorithm:',iAlgo,' Color Encoding:', color_encoding,' Step Size:', FStepSize);

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
      EdFCLayers.Text+'-'+
      BoolToStr(CheckCenter.Checked,'T','F')+'-'+
      BoolToStr(CheckAutoNoisy.Checked,'T','F')+'-'+
      EdThreadCount.Text+'-'+
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
      ComboAugmentation.Text+'-'+
      EdNoiseLevel.Text+'-'+
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
  fLearningRate := StrToFloat(EdLearningRate.Text);
  fLearningRateDecay := 1 - StrToFloat(EdLearnRateDecay.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fLearningRate;
  fDropout := StrToFloat(EdDropout.Text);
  MaxPool := StrToInt(EdMaxPool.Text);
  Stride := StrToInt(EdStride.Text);
  iMaxEpochs := StrToInt(EdMaxEpochs.Text);
  fMinLearnRate := StrToFloat(EdMinLearnRate.Text);
  fL2Decay := StrToFloat(EdL2Decay.Text);
  bDataAugmentation := (ComboAugmentation.ItemIndex > 0);
  fAutoTrigger := StrToFloat(EdAutoTrigger.Text);
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

    InputLayer := NN.AddLayer( TNNetInput.Create(32,32,3) );

    if ( (iAlgo<>6) and (iAlgo<>4) ) then
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
            NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,iInnerConvFeatureSize,iPadding,0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(32) );
            if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
          end;
        end;

      end;
      1:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer( TNNetLayerStdNormalization.Create() );
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer( TNNetLayerStdNormalization.Create() );
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
        //ideal input:
        //  NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        //NN.AddLayer(TNNetLayerStdNormalization.Create());
        NN.AddLayer(TNNetConvolutionReLU.Create(64, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        //NN.AddLayer(TNNetLayerStdNormalization.Create());
        NN.AddLayer(TNNetFullConnectReLU.Create(384));
        NN.AddLayer(TNNetFullConnectReLU.Create(192));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      4:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0,1]), InputLayer);
        NN.AddLayer(TNNetConvolutionReLU.Create(42, FeatureSize, 0, 0));
        Branch1 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));
        if bCenterDivide then Branch1 := NN.AddLayer( TNNetLayerStdNormalization.Create() );

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([2]), InputLayer);
        NN.AddLayer(TNNetConvolutionReLU.Create(22, FeatureSize, 0, 0));
        Branch2 := NN.AddLayer(TNNetMaxPool.Create(MaxPool));
        if bCenterDivide then Branch2 := NN.AddLayer( TNNetLayerStdNormalization.Create() );

        NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2]));
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, iInnerConvFeatureSize, iPadding, 0));

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      5:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        if bCenterDivide then NN.AddLayer( TNNetLayerStdNormalization.Create() );
        NN.AddLayer(TNNetConvolutionReLU.Create(128, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLocalConnectReLU.Create(256, iInnerConvFeatureSize, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(256));
        NN.AddLayer(TNNetFullConnectReLU.Create(128));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      6:
      begin
        // Branch1
        NN.AddLayerAfter( TNNetSplitChannels.Create([0]), InputLayer);
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(4));
        if bCenterDivide then NN.AddLayer( TNNetLayerStdNormalization.Create() );
        Branch1 := NN.AddLayer(TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0));

        // Branch2
        NN.AddLayerAfter( TNNetSplitChannels.Create([1]), InputLayer);
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(4));
        if bCenterDivide then NN.AddLayer( TNNetLayerStdNormalization.Create() );
        Branch2 := NN.AddLayer(TNNetConvolutionReLU.Create(22, iInnerConvFeatureSize, iPadding, 0));

        // Branch3
        NN.AddLayerAfter( TNNetSplitChannels.Create([2]), InputLayer);
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(4));
        if bCenterDivide then NN.AddLayer( TNNetLayerStdNormalization.Create() );
        Branch3 := NN.AddLayer(TNNetConvolutionReLU.Create(21, iInnerConvFeatureSize, iPadding, 0));

        NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2, Branch3]));
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, iInnerConvFeatureSize, iPadding, 0));

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      7:
      begin
        // Branch1 (default is 5x5 features)
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch1 := NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize, 0, 0));

        // Branch2 (default is 3x3 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize-2, 0, 0),InputLayer);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize-2, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch2 := NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize-2, 0, 0));

        // Branch3 (default is 7x7 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize+2, 0, 0),InputLayer);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize+2, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        Branch3 := NN.AddLayer(TNNetConvolutionReLU.Create(FeaturesNum, FeatureSize+2, 0, 0));

        // Concats both branches so the NN has only one end.
        NN.AddLayer(TNNetConcat.Create([Branch1,Branch2,Branch3]));

        NN.AddLayer( TNNetFullConnectReLU.Create(32) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
    end;

    NN.AddLayer( TNNetFullConnectReLU.Create(NumClasses) );
    if (ComboLastLayer.ItemIndex = 0) then
    begin
      NN.AddLayer( TNNetSoftMax.Create() );
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


  WriteLn('Setting L2 to:', fL2Decay:6:4);
  NN.SetLearningRate(CurrentLearningRate, fInertia);
  if RadL2All.Checked
  then NN.SetL2Decay(fL2Decay)
  else NN.SetL2DecayToConvolutionalLayers(fL2Decay);

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
  else if ( iAlgo = 4 ) then
  begin
    secondNeuronalLayer := NN.GetFirstImageNeuronalLayerIdx(firstNeuronalLayer + 1);
    CreateNeuronImages(firstNeuronalLayer, 32, 8);
  end
  else
  begin
    CreateNeuronImages(firstNeuronalLayer, 32, 8);
  end;

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
    LocalErrorSum := 0;
    startTime := Now();

    for I := 1 to (ImgVolumes.Count div FStepSize) {div 10} do
    begin
      Hit.Fill(0);
      Miss.Fill(0);
      Loss.Fill(0);
      ErrorSum.Fill(0);

      RunNNThread();

      globalHit := Round(Hit.GetSum());
      globalMiss := Round(Miss.GetSum());
      TotalLoss := Loss.GetSum();
      LocalErrorSum := ErrorSum.GetSum();

      if (globalHit > 0) then
      begin
        CurrentRate := globalHit / (globalHit + globalMiss);
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;

        CVSError := LocalErrorSum / (globalHit+globalMiss);
        CVSLoss  := (TotalLoss / (globalHit+globalMiss) );
        CVSRate  := CurrentRate;

        WriteLn
        (
          (globalHit + globalMiss)*I + globalLoopCnt*ImgVolumes.Count,
          ' Examples seen. Accuracy:', CVSRate:6:4,
          ' Error:', CVSError:10:5,
          ' Loss:', CVSLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Thread Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Thread Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Time:', totalTimeSeconds:6:2,'s'
        );
        NN.ClearTime();

        LabClassRate.Caption := PadLeft(IntToStr( (globalHit*100) div (globalHit+globalMiss) )+'%',4) ;

        LabTime.Caption :=
          'Epoch time: ' + FloatToStrF(Round(totalTimeSeconds*(50000/FStepSize)/60),ffGeneral,4,2)+' minutes.' +
          ' 200 epochs: ' + FloatToStrF(Round(200*totalTimeSeconds*(50000/FStepSize)/3600),ffGeneral,4,2)+' hours.';

        LabTotalTime.Caption :=
          'Epochs: '+IntToStr(globalLoopCnt)+
          '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.';

        if ( bAutoNoisy and (CVSValidationRate > 0) )then
        begin
          // is the NN 10% overfit?
          if (CVSRate > CVSValidationRate + fAutoTrigger) then
          begin
            bDataAugmentation := true;
          end
          else
          begin
            bDataAugmentation := false;
          end;
        end;

        startTime := Now();
        Application.ProcessMessages;
      end;
    end;

    if (FRunning) then
    begin
      if (globalLoopCnt mod FThreadNum = 0) then
      begin

        if ( iAlgo = 6 ) then
        begin
          ShowNeurons(firstNeuronalLayer, FeaturesNum*0, 16);
          ShowNeurons(secondNeuronalLayer, FeaturesNum*1, 16);
          ShowNeurons(thirdNeuronalLayer, FeaturesNum*2, 16);
        end
        else if ( iAlgo = 4 ) then
        begin
          ShowNeurons(firstNeuronalLayer, 0, 32);
          ShowNeurons(secondNeuronalLayer, 42, 32);
        end
        else
        begin
          ShowNeurons(firstNeuronalLayer, FeaturesNum*0, 32);
        end;

      end;
    end;

    Inc(globalLoopCnt);

    if (FRunning) then
    begin
      Hit.Fill(0);
      Miss.Fill(0);
      Loss.Fill(0);
      ErrorSum.Fill(0);

      ImgWorkingVolumes := ImgValidationVolumes;

      WriteLn('Starting Validation.');
      TestNNThread();

      CVSValidationRate  := Hit.GetSum() / ImgWorkingVolumes.Count;
      CVSValidationLoss  := Loss.GetSum() / ImgWorkingVolumes.Count;
      CVSValidationError := ErrorSum.GetSum() / ImgWorkingVolumes.Count;

      LabTestRate.Caption := PadLeft(IntToStr( Round(CVSValidationRate*100) )+'%',4) ;

      if (globalLoopCnt mod FThreadNum = 0) then
      begin
        DisplayInputImage(color_encoding);
        Application.ProcessMessages;
        NN.DebugWeights();
        NN.DebugErrors();
        NN.SaveToFile(fileName);
        SaveScreenshot(fileNameImage);
        Application.ProcessMessages;
      end;

      if ( (globalLoopCnt mod 10 = 0) and (globalLoopCnt > 0) ) then
      begin
        Hit.Fill(0);
        Miss.Fill(0);
        Loss.Fill(0);
        ErrorSum.Fill(0);

        ImgWorkingVolumes := ImgTestVolumes;

        WriteLn('Starting Testing.');
        TestNNThread();

        CVSTestRate  := Hit.GetSum() / ImgWorkingVolumes.Count;
        CVSTestLoss  := Loss.GetSum() / ImgWorkingVolumes.Count;
        CVSTestError := ErrorSum.GetSum() / ImgWorkingVolumes.Count;

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

      if (CurrentLearningRate * fLearningRateDecay > fMinLearnRate) then
      begin
        CurrentLearningRate *= fLearningRateDecay;
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        WriteLn
        (
          'Learning rate set to:',CurrentLearningRate:9:7
        );
      end;

      if CheckProportionalLearningRate.Checked then
      begin
        fNewLearningRate := (fLearningRate * (1 - CVSValidationRate));
        if (( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate < CurrentLearningRate)) then
        begin
          CurrentLearningRate := fNewLearningRate;
          bLoadedFile := false;
          NN.SetLearningRate(CurrentLearningRate, fInertia);
          WriteLn
          (
            'Learning rate proportionally set to:',CurrentLearningRate:7:5
          );
        end
        else
        begin
            //WriteLn('Current learning rate is: [',CurrentLearningRate:7:5,']');
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
          ' Examples seen:', globalLoopCnt*ImgVolumes.Count,
          ' Accuracy:', CVSRate:6:4,' ', CVSValidationRate:6:4,
          ' Error: ', CVSError:10:5,' ',CVSValidationError:10:5,
          ' Loss: ', CVSLoss:7:5,' ',CVSValidationLoss:7:5,
          ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
        );

      end;
      Application.ProcessMessages();
    end;
    NN.ClearTime();
  end;

  FreeNeuronImages();
  {$IFDEF NeuralThreadLib}NeuronThreadListFree();{$ENDIF}

  Hit.Free;
  Miss.Free;
  Loss.Free;
  ErrorSum.Free;
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

  if ( (ComboAlgo.ItemIndex <> 0) and flag) then
  begin
    EdMaxPool.Enabled := false;
    EdConvLayers.Enabled := false;
    EdFCLayers.Enabled := false;
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

procedure TFormVisualLearning.RunNNThread();
var
  BlockSize, BlockSizeRest: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput: TNNetVolume;
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

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := FStepSize;
  BlockSizeRest := FStepSize;

  LocalNN := NN;
  for I := 1 to BlockSize do
  begin
    if not(FRunning) then Break;
    ImgIdx := Random(ImgVolumes.Count);

    ImgInput.Copy(ImgVolumes[ImgIdx]);
    ImgInput.Tag := ImgVolumes[ImgIdx].Tag;

    if bDataAugmentation then
    begin
      if ( (FDataAugmentationType = 1) or (FDataAugmentationType = 3) ) then
      begin
        // Modifies the image
        ImgInputCp.Copy(ImgInput);
        ImgShiftSize := Random(128) * 3;
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

        ImgInput.Add( ( (Random(100)-50)*FNoiseLevel) / 256 );
      end;

      if ( (FDataAugmentationType = 2) or (FDataAugmentationType = 3) ) then
      begin
        ImgInput.AddGaussianNoise(FNoiseLevel);
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

    if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;

  end; // of for

  globalImgInput.Copy(ImgInput);

  Hit.FData[0] := LocalHit;
  Miss.FData[0] := LocalMiss;
  Loss.FData[0] := LocalTotalLoss;
  ErrorSum.FData[0] := LocalErrorSum;

  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearning.TestNNThread();
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

  BlockSize := ImgWorkingVolumes.Count;
  StartPos  := 0;
  FinishPos := BlockSize - 1;

  LocalNN := NN;
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if not(FRunning) then Break;
    ImgIdx := I;

    ImgInput.Copy(ImgWorkingVolumes[ImgIdx]);
    ImgInput.Tag := ImgWorkingVolumes[ImgIdx].Tag;

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

  Hit.FData[0] := LocalHit;
  Miss.FData[0] := LocalMiss;
  Loss.FData[0] := LocalTotalLoss;
  ErrorSum.FData[0] := LocalErrorSum;

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

