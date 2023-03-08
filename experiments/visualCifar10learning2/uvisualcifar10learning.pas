unit uvisualcifar10learning;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume;

type

  { TFormVisualLearning }

  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    CheckAddNoisy: TCheckBox;
    CheckAutoNoisy: TCheckBox;
    CheckCenter: TCheckBox;
    ComboAlgo: TComboBox;
    ComboLastLayer: TComboBox;
    EdInertia: TEdit;
    EdFeatures: TEdit;
    EdFeatureSize: TEdit;
    EdFCLayers: TEdit;
    EdConvLayers: TEdit;
    EdDropout: TEdit;
    EdL2Decay: TEdit;
    EdMinLearnRate: TEdit;
    EdMaxEpochs: TEdit;
    EdStride: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    ImgSample: TImage;
    LabAlgo: TLabel;
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
    procedure EdLearningRateChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
  public
    { public declarations }
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils;

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
end;

procedure TFormVisualLearning.Learn(Sender: TObject);
var
  NN: TNNet;
  I: integer;
  ImgVolumes, ImgTestVolumes: TNNetVolumeList;
  ImgShiftSize, ImgShiftCnt, ImgShiftCntD: integer;
  ImgInput, ImgInputCp: TNNetVolume;
  Volume: TNNetVolume;
  pOutput, vOutput, vDisplay: TNNetVolume;
  hit, miss: integer;
  NumClasses: integer;
  ErrorSum, LastError: TNeuralFloat;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  globalLoopCnt: integer;
  CurrentRate: double;
  CurrentLearningRate, fLearningRate, fInertia: single;
  MaxPool,Stride: integer;
  fDropout: single;
  aImage: array of TImage;
  NeuronCount: integer;
  ImgIdx: integer;
  FeaturesNum, FeatureSize: integer;
  InnerConvNum, InnerConvCnt: integer;
  InnerFCNum, InnerFCCnt: integer;
  MaxW, MinW: TNeuralFloat;
  fileName, fileNameBase, FileNameCSV: string;
  firstNeuronalLayer: integer;
  bLoadedFile: boolean;
  iMaxEpochs: integer;
  fMinLearnRate, fNewLearningRate: single;
  bIsSoftmax: boolean;
  bAddNoisy: boolean;
  bCenterDivide: boolean;
  CurrentLoss, TotalLoss: single;
  CVSError, oldCVSLoss, CVSLoss, CVSRate,
  CVSTestError, CVSTestLoss, CVSTestRate: TNeuralFloat;
  CSVFile: TextFile;
  OutputValue: TNeuralFloat;
  fL2Decay: TNeuralFloat;
  iAlgo: integer;
  InputLayer, Branch1, Branch2, Branch3: TNNetLayer;
begin
  writeln('Creating Neural Network...');
  ImgVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  NumClasses := 10;
  globalLoopCnt := 0;
  bLoadedFile := false;
  bIsSoftmax  := false;
  CVSError    := 0;
  CVSLoss     := 0;
  CurrentLoss := 0;
  oldCVSLoss  := 100;
  CVSRate     := 0;
  CurrentRate := 0;
  CVSTestRate := 1;
  CVSTestLoss := 0;
  CVSTestError:= 0;

  iAlgo := ComboAlgo.ItemIndex;

  if iAlgo = 0 then
  begin
    fileNameBase :=
      'autosave-'+
      EdLearningRate.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdDropout.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdStride.Text+'-'+
      EdMaxPool.Text+'-'+
      EdConvLayers.Text+'-'+
      EdFCLayers.Text+'-'+
      BoolToStr(CheckCenter.Checked,'T','F')+'-'+
      ComboLastLayer.Text;
  end
  else
  begin
    fileNameBase :=
      'autosave-algo'+
      IntToStr(iAlgo)+'-'+
      EdLearningRate.Text+'-'+
      EdInertia.Text+'-'+
      EdL2Decay.Text+'-'+
      EdDropout.Text+'-'+
      EdFeatures.Text+'-'+
      EdFeatureSize.Text+'-'+
      EdStride.Text+'-'+
      BoolToStr(CheckCenter.Checked,'T','F')+'-'+
      ComboLastLayer.Text;
  end;

  WriteLn('File name is: ',fileNameBase);

  FileNameCSV := fileNameBase + '.csv';

  FileName := fileNameBase + '.nn';

  AssignFile(CSVFile, FileNameCSV);

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I := 0 to 49999 do
  begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;

  for I := 0 to 9999 do
  begin
    Volume := TNNetVolume.Create();
    ImgTestVolumes.Add(Volume);
  end;

  loadCifar10Dataset(ImgVolumes, 1, 0);
  loadCifar10Dataset(ImgVolumes, 2, 10000);
  loadCifar10Dataset(ImgVolumes, 3, 20000);
  loadCifar10Dataset(ImgVolumes, 4, 30000);
  loadCifar10Dataset(ImgVolumes, 5, 40000);
  loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0);
  //--------------------------------------------------------------------
  FeaturesNum := StrToInt(EdFeatures.Text);
  FeatureSize := StrToInt(EdFeatureSize.Text);
  fLearningRate := StrToFloat(EdLearningRate.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fLearningRate;
  fDropout := StrToFloat(EdDropout.Text);
  MaxPool := StrToInt(EdMaxPool.Text);
  Stride := StrToInt(EdStride.Text);
  iMaxEpochs := StrToInt(EdMaxEpochs.Text);
  fMinLearnRate := StrToFloat(EdMinLearnRate.Text);
  fL2Decay := StrToFloat(EdL2Decay.Text);
  bAddNoisy := CheckAddNoisy.Checked;
  bCenterDivide := CheckCenter.Checked;

  if CheckAutoNoisy.Checked then
  begin
    bAddNoisy := false;
    CheckAddNoisy.Checked := false;
  end;

  NN := TNNet.Create();

  if ( FileExists(fileName) and FileExists(FileNameCSV) ) then
  begin
    writeln('Loading neural network from file: ',fileName);
    Append(CSVFile);
    NN.LoadFromFile(fileName);
    // as this is a loaded NN, we'll start measuring and not learning
    NN.SetLearningRate(0, fInertia);
    bLoadedFile := true;
  end
  else
  begin
    ReWrite(CSVFile);
    writeln(CSVFile, 'epoch,training rate,training loss,training error,testing rate,testing loss,testing error,learning rate');

    InputLayer := NN.AddLayer( TNNetInput.Create(32,32,3) );

    if bCenterDivide then
    begin
      InputLayer := NN.AddLayer( TNNetLayerStdNormalization.Create() );
    end;

    NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,FeatureSize,0,Stride) );

      case iAlgo of
      0:
      begin

        if MaxPool>0 then NN.AddLayer( TNNetMaxPool.Create(MaxPool) );

        InnerConvNum := StrToInt(EdConvLayers.Text);
        InnerFCNum := StrToInt(EdFCLayers.Text);

        if (InnerConvNum>0) then
        begin
          for InnerConvCnt := 1 to InnerConvNum do
          begin
            NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,3,1,0) );
          end;
        end;

        if (InnerFCNum>0) then
        begin
          for InnerFCCnt := 1 to InnerFCNum do
          begin
            NN.AddLayer( TNNetFullConnectReLU.Create(32) );
          end;
          if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
        end;

      end;
      1:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      2:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      // https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
      3: // TENSOR FLOW INSPIRED
      begin
        //ideal input:
        //  NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        //  NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLayerStdNormalization.Create());
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLayerStdNormalization.Create());
        NN.AddLayer(TNNetFullConnectReLU.Create(384));
        NN.AddLayer(TNNetFullConnectReLU.Create(192));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      4:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      5:
      begin
        //ideal input: NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLocalConnectReLU.Create(256, 5, 0, 0));
        NN.AddLayer(TNNetFullConnectReLU.Create(256));
        NN.AddLayer(TNNetFullConnectReLU.Create(128));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
      end;
      6:
      begin
        //ideal input: Reshape, Identity, LocalConnect testing
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLocalConnectReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetReshape.Create(128,1,1));
        NN.AddLayer(TNNetIdentity.Create());
        NN.AddLayer(TNNetFullConnectReLU.Create(64));
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

        // This algorith requires special inertia
        //fInertia := Max(fInertia, 0.999);
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
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(128));
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

  WriteLn('Setting L2 to:', fL2Decay:6:4);

  if RadL2All.Checked
  then NN.SetL2Decay(fL2Decay)
  else NN.SetL2DecayToConvolutionalLayers(fL2Decay);

  if NN.Layers[NN.GetLastLayerIdx()] is TNNetSoftMax then
  begin
    WriteLn('Neural network has Softmax.');
    bIsSoftmax := true;
  end
  else
  begin
    WriteLn('Softmax isn''t present.');
  end;

  //--------------------------------------------------------------------
  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  vDisplay:= TNNetVolume.Create(NumClasses,1,1);
  firstNeuronalLayer := NN.GetFirstNeuronalLayerIdx();

  SetLength(aImage, NN.Layers[firstNeuronalLayer].Neurons.Count);

  for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do
  begin
    aImage[NeuronCount] := TImage.Create(FormVisualLearning);
    aImage[NeuronCount].Parent := FormVisualLearning;
    aImage[NeuronCount].Width  := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeX;
    aImage[NeuronCount].Height := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.SizeY;
    aImage[NeuronCount].Top    := (NeuronCount div 8) * 36 + 180;
    aImage[NeuronCount].Left   := (NeuronCount mod 8) * 36 + 32;
    aImage[NeuronCount].Stretch:=true;
  end;

  NN.DebugWeights();
  WriteLn('Neural network has: ');
  NN.DebugStructure();
  WriteLn('Computing...');

  globalStartTime := Now();
  while ( FRunning and (iMaxEpochs > globalLoopCnt div 5) ) do
  begin
    hit  := 0;
    miss := 0;
    ErrorSum := 0;
    LastError := 0;
    startTime := Now();
    TotalLoss := 0;

    for I := 0 to ImgVolumes.Count - 1 do
    begin
      if not(FRunning) then Break;
      ImgIdx := Random(ImgVolumes.Count);
      //-- CAREFUL
      ImgInput.Copy(ImgVolumes[ImgIdx]);
      ImgInput.Tag := ImgVolumes[ImgIdx].Tag;

      if bAddNoisy then
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
          ImgInput.Add( (Random(10)-5)/256 );
        end;
      end;

      NN.Compute( ImgInput );
      NN.GetOutput(pOutput);

      if (bIsSoftmax) then
      begin
        vOutput.SetClassForSoftMax( ImgInput.Tag );
      end
      else
      begin
        vOutput.SetClassForReLU( ImgInput.Tag );
      end;

      ErrorSum += vOutput.SumDiff(pOutput);

      if (bIsSoftmax) then
      begin
        OutputValue := pOutput.FData[ ImgInput.Tag ];
        if (OutputValue > 0) then
        begin
          CurrentLoss := -Ln(OutputValue);
        end
        else
        begin
          WriteLn('Error: invalid output value',OutputValue);
          CurrentLoss := 1;
        end;
        TotalLoss += CurrentLoss;
      end;

      if ( (globalLoopCnt>0) or not(bLoadedFile) ) then
      begin
        // we don't backpropagate when we are too sure.
        // if ( (CurrentLoss>=0.1) or not(bIsSoftmax) ) then
        begin
          NN.Backpropagate(vOutput);
        end;
      end;

      if I mod 1000 = 0 then
      begin
        vDisplay.Copy(ImgInput);
        vDisplay.Mul(64);
        vDisplay.Add(128);

        LoadVolumeIntoTImage(vDisplay, ImgSample);
        ImgSample.Width := 64;
        ImgSample.Height := 64;

        for NeuronCount := 0 to NN.Layers[firstNeuronalLayer].Neurons.Count - 1 do
        begin
          MaxW := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMax();
          MinW := NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights.GetMin();
          vDisplay.Copy(NN.Layers[firstNeuronalLayer].Neurons[NeuronCount].Weights);
          vDisplay.Mul(256/(MaxW-MinW));
          vDisplay.Add(128);
          LoadVolumeIntoTImage(vDisplay, aImage[NeuronCount]);
          aImage[NeuronCount].Width := 32;
          aImage[NeuronCount].Height := 32;
        end;
      end;

      Application.ProcessMessages();

      if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then
      begin
        Inc(Hit);
      end
      else
      begin
        Inc(Miss);
      end;

      if (Hit > 0) then CurrentRate := Hit / (Hit + Miss);

      if (Hit>0) and (I>0) and (I mod 1000 = 0) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;

        CVSError := (ErrorSum-LastError);
        CVSLoss  := (TotalLoss/(Hit+Miss));
        CVSRate  := CurrentRate;

        WriteLn
        (
          I, ' Examples seen. Accuracy:', CVSRate:6:4,
          ' Error:', CVSError:10:5,
          ' Loss:', CVSLoss:7:5,
          ' Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Time:', totalTimeSeconds:6:2,'s'
        );
        NN.ClearTime();

        LabClassRate.Caption := PadLeft(IntToStr( (Hit*100) div (Hit+Miss) )+'%',4) ;

        LabTime.Caption :=
          'Epoch time: ' + FloatToStrF(Round(totalTimeSeconds*50/60),ffGeneral,4,2)+' minutes.' +
          ' 200 epochs: ' + FloatToStrF(Round(200*totalTimeSeconds*50/3600),ffGeneral,4,2)+' hours.';

        LabTotalTime.Caption :=
          'Epochs: '+IntToStr(globalLoopCnt)+
          '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.';

        fNewLearningRate := (fLearningRate * (1 - CurrentRate));
        if (( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate < CurrentLearningRate)) then
        begin
          CurrentLearningRate := fNewLearningRate;
          bLoadedFile := false;
          NN.SetLearningRate(CurrentLearningRate, fInertia);
          WriteLn
          (
            'Learning rate set to:',CurrentLearningRate:7:5
          );
        end
        else
        begin
          //WriteLn('Current learning rate is: [',CurrentLearningRate:7:5,']');
        end;

        if CheckAutoNoisy.Checked then
        begin
          // is the NN 10% overfit?
          if (CVSRate > CVSTestRate + 0.1) then
          begin
            bAddNoisy := true;
            CheckAddNoisy.Checked := true;
          end
          else
          begin
            bAddNoisy := false;
            CheckAddNoisy.Checked := false;
          end;
        end;

        startTime := Now();
        LastError := ErrorSum;
        Application.ProcessMessages;

        Hit := 0;
        Miss := 0;
        TotalLoss := 0;
      end;
    end;

    Inc(globalLoopCnt);

    if (FRunning) then
    begin
      NN.EnableDropouts(false);
      TestBatch(NN, ImgTestVolumes, 0, CVSTestRate, CVSTestLoss, CVSTestError);
      NN.EnableDropouts(true);

      writeln
      (
        CSVFile,
        globalLoopCnt,',',
        CVSRate:6:4,',',
        CVSLoss:7:5,',',
        CVSError:10:5,',',
        CVSTestRate:6:4,',',
        CVSTestLoss:7:5,',',
        CVSTestError:10:5,',',
        CurrentLearningRate:9:7
      );

      // has it got worse? If yes, reduce learning rate.
      if
      (
        (oldCVSLoss <= CVSLoss) and
        (CurrentLearningRate * 0.99 > fMinLearnRate)
      ) then
      begin
        CurrentLearningRate *= 0.99;
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        WriteLn
        (
          'Loss got worse. Learning rate set to:',CurrentLearningRate:9:7
        );
      end;
      oldCVSLoss := CVSLoss;

      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

      LabTestRate.Caption := PadLeft(IntToStr( Round(CVSTestRate*100) )+'%',4) ;

      if (Hit > 0) then
      begin
        WriteLn
        (
          'Epochs: ',globalLoopCnt,
          ' Examples seen:', globalLoopCnt*ImgVolumes.Count,
          ' Accuracy:', CVSRate : 6: 4,' ', CVSTestRate:6:4,
          ' Error: ', CVSError: 10: 5,' ',CVSTestError:10:5,
          ' Loss: ', CVSLoss:7:5,' ',CVSTestLoss:7:5,
          ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
        );

        NN.DebugWeights();
        NN.DebugErrors();
        NN.SaveToFile(fileName);
      end;
    end;
    NN.ClearTime();
  end;

  for NeuronCount := Low(aImage) to High(aImage) do
  begin
    aImage[NeuronCount].Free;
  end;

  ImgInputCp.Free;
  ImgInput.Free;
  CloseFile(CSVFile);
  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  ImgTestVolumes.Free;
  ImgVolumes.Free;
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

end.

