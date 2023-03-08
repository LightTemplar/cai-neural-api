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
    EdInertia: TEdit;
    EdFeatures: TEdit;
    EdFeatureSize: TEdit;
    EdFCLayers: TEdit;
    EdConvLayers: TEdit;
    EdDropout: TEdit;
    EdStride: TEdit;
    EdMaxPool: TEdit;
    EdLearningRate: TEdit;
    ImgSample: TImage;
    LabConv: TLabel;
    Labdropout: TLabel;
    LabStride: TLabel;
    LabMaxPool: TLabel;
    LabHiddenNum: TLabel;
    LabFeatureSize: TLabel;
    LabNFeatures: TLabel;
    LabInertia: TLabel;
    LabLearning: TLabel;
    procedure ButLearnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    procedure Learn(Sender: TObject);
  public
    { public declarations }
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;

  if (FRunning) then
  begin
    FRunning := false;
    ButLearn.Caption := 'Restart';
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    Learn(Sender);
  end;

end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  FRunning := false;
end;

procedure TFormVisualLearning.Learn(Sender: TObject);
var
  NN: TNNet;
  I: integer;
  ImgVolumes: TNNetVolumeList;
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
  fileName: string;
  firstNeuronalLayer: integer;
begin
  writeln('Creating Neural Network...');
  ImgVolumes := TNNetVolumeList.Create();
  NumClasses := 10;
  globalLoopCnt := 0;

  fileName :=
    'autosave-'+
    EdLearningRate.Text+'-'+
    EdInertia.Text+'-'+
    EdDropout.Text+'-'+
    EdFeatures.Text+'-'+
    EdFeatureSize.Text+'-'+
    EdStride.Text+'-'+
    EdMaxPool.Text+'-'+
    EdConvLayers.Text+'-'+
    EdFCLayers.Text+
    '.nn';

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I := 0 to 9999 do
  begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;

  //--------------------------------------------------------------------
  FeaturesNum := StrToInt(EdFeatures.Text);
  FeatureSize := StrToInt(EdFeatureSize.Text);
  fLearningRate := StrToFloat(EdLearningRate.Text);
  fInertia := StrToFloat(EdInertia.Text);
  CurrentLearningRate := fLearningRate;
  fDropout := StrToFloat(EdDropout.Text);
  MaxPool := StrToInt(EdMaxPool.Text);
  Stride := StrToInt(EdStride.Text);

  NN := TNNet.Create();

  if FileExists(fileName) then
  begin
    writeln('Loading neural network from file: ',fileName);
    NN.LoadFromFile(fileName);
    // as this is a loaded NN, we'll start measuring and not learning
    NN.SetLearningRate(0, fInertia);
  end
  else
  begin
    NN.AddLayer( TNNetInput.Create(32,32,3) );

    if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

    NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,FeatureSize,Stride,0) );

    if MaxPool>0 then NN.AddLayer( TNNetMaxPool.Create(MaxPool) );

    if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );

    InnerConvNum := StrToInt(EdConvLayers.Text);
    InnerFCNum := StrToInt(EdFCLayers.Text);

    if (InnerConvNum>0) then
    begin
      for InnerConvCnt := 1 to InnerConvNum do
      begin
        NN.AddLayer( TNNetConvolutionReLU.Create(FeaturesNum,3,1,0) );
        if fDropout > 0 then NN.AddLayer( TNNetDropout.Create(fDropout) );
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

    NN.AddLayer( TNNetFullConnectReLU.Create(NumClasses) );
    NN.SetLearningRate(CurrentLearningRate, fInertia);
  end;

  //--------------------------------------------------------------------

  NN.SetLearningRate(CurrentLearningRate, fInertia);
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
    aImage[NeuronCount].Top    := (NeuronCount div 12) * 36 + 120;
    aImage[NeuronCount].Left   := (NeuronCount mod 12) * 36 + 32;
    aImage[NeuronCount].Stretch:=true;
  end;

  //NN.LoadFromFile('autosave.cai');
  NN.DebugWeights();
  WriteLn('Neural network has: ');
  NN.DebugStructure();
  WriteLn('Computing...');

  globalStartTime := Now();
  while FRunning do
  begin
    hit  := 0;
    miss := 0;
    ErrorSum := 0;
    LastError := 0;
    startTime := Now();
    loadCifar10Dataset(ImgVolumes, globalLoopCnt mod 5 + 1);
    for I := 0 to ImgVolumes.Count - 1 do
    begin
      if not(FRunning) then Break;
      ImgIdx := Random(ImgVolumes.Count);
      //-- CAREFUL
      NN.Compute(ImgVolumes[ImgIdx]);
      NN.GetOutput(pOutput);

      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag ); // ReLU - no softmax

      ErrorSum += vOutput.SumDiff(pOutput);

      if (globalLoopCnt>0) then
      begin
        NN.Backpropagate(vOutput);
      end;

      if I mod 1000 = 0 then
      begin
        vDisplay.Copy(ImgVolumes[ImgIdx]);
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
        WriteLn
        (
          I, ' Accuracy:', Hit/(Hit+Miss):6:4,
          ' Error:', (ErrorSum-LastError):10:5,
          ' Time:', totalTimeSeconds:6:2,'s',
          ' Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s'
        );
        NN.ClearTime();

        startTime := Now();
        LastError := ErrorSum;
        Application.ProcessMessages;
      end;
    end;

    Inc(globalLoopCnt);
    if (Hit > 0) then
    begin
      WriteLn
      (
        '---- ', globalLoopCnt*10000,
        ' Examples seen. Training accuracy:', CurrentRate : 6: 4,
        ' Error: ', ErrorSum: 10: 5,
        ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
      );

      if (fLearningRate * (1 - CurrentRate)) < CurrentLearningRate then
      begin
        CurrentLearningRate := fLearningRate * (1 - CurrentRate);
        NN.SetLearningRate(CurrentLearningRate, fInertia);
        WriteLn('Learning rate set to: [',CurrentLearningRate:7:5,']');
      end
      else
      begin
        WriteLn('Current learning rate is: [',CurrentLearningRate:7:5,']');
      end;

      NN.DebugWeights();
      NN.DebugErrors();
      NN.SaveToFile(fileName);
    end;
    NN.ClearTime();
  end;

  for NeuronCount := Low(aImage) to High(aImage) do
  begin
    aImage[NeuronCount].Free;
  end;

  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  ImgVolumes.Free;
end;


end.

