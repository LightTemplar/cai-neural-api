unit uvisualcifar10test;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuraldatasetsv, neuralvolumev, neuraldatasets,
  neuralvolume;

type

  { TFormVisualLearning }

  TFormVisualLearning = class(TForm)
    ButTest: TButton;
    EdTestBinFile: TEdit;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabTestFile: TLabel;
    OpenDialogNN: TOpenDialog;
    procedure ButTestClick(Sender: TObject);
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

procedure TFormVisualLearning.ButTestClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;

  if (FRunning) then
  begin
    FRunning := false;
    ButTest.Caption := 'Retest';
    LabClassRate.Caption := '0%';
  end
  else
  begin
    if (OpenDialogNN.Execute()) then
    begin
      if FileExists(OpenDialogNN.FileName) then
      begin
        FRunning := true;
        ButTest.Caption := 'Stop';
        Learn(Sender);
        FRunning := false;
        ButTest.Caption := 'Retest';
      end;
    end;
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
  aImage: array of TImage;
  NeuronCount: integer;
  ImgIdx: integer;
  MaxW, MinW: TNeuralFloat;
  fileName: string;
  firstNeuronalLayer: integer;
begin
  writeln('Creating Neural Network...');
  ImgVolumes := TNNetVolumeList.Create();
  NumClasses := 10;

  fileName := OpenDialogNN.FileName;

  //--------------------------------------------------------------------
  // creates required volumes to store images
  for I := 0 to 9999 do
  begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  //--------------------------------------------------------------------

  NN := TNNet.Create();

  writeln('Loading neural network from file: ',fileName);
  NN.LoadFromFile(fileName);
  NN.EnableDropouts(false);
  firstNeuronalLayer := NN.GetFirstNeuronalLayerIdx();

  pOutput := TNNetVolume.Create(NumClasses,1,1);
  vOutput := TNNetVolume.Create(NumClasses,1,1);
  vDisplay:= TNNetVolume.Create(NumClasses,1,1);

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

  NN.DebugWeights();
  WriteLn('Neural network has: ');
  WriteLn(' Layers: ',  NN.CountLayers()  );
  WriteLn(' Neurons:',  NN.CountNeurons() );
  WriteLn(' Weights:' , NN.CountWeights() );
  WriteLn('Computing...');

  begin
    hit  := 0;
    miss := 0;
    ErrorSum := 0;
    LastError := 0;
    startTime := Now();
    loadCifar10Dataset(ImgVolumes, EdTestBinFile.Text);
    for I := 0 to ImgVolumes.Count - 1 do
    begin
      if not(FRunning) then Break;
      ImgIdx := Random(ImgVolumes.Count);
      //-- CAREFUL
      NN.Compute(ImgVolumes[ImgIdx]);
      NN.GetOutput(pOutput);

      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag ); // ReLU - no softmax
      ErrorSum += vOutput.SumDiff(pOutput);

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

        Application.ProcessMessages();
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

      if (Hit>0) and (I>0) and ((I+1) mod 1000 = 0) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;
        WriteLn
        (
          I+1, ' Accuracy:', Hit/(Hit+Miss):6:4,
          ' Error:', (ErrorSum-LastError):10:5,
          ' Time:', totalTimeSeconds:6:2,'s',
          ' Forward:', (NN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward:', (NN.BackwardTime * 24 * 60 * 60):6:2,'s'
        );
        NN.ClearTime();

        LabClassRate.Caption := IntToStr( (Hit*100) div (Hit+Miss) )+'%';

        startTime := Now();
        LastError := ErrorSum;
        Application.ProcessMessages;
      end;

    end;

    NN.DebugWeights();
  end;

  for NeuronCount := Low(aImage) to High(aImage) do
  begin
    aImage[NeuronCount].Free;
  end;

  LabClassRate.Caption := '0%';

  vDisplay.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  ImgVolumes.Free;
end;


end.

