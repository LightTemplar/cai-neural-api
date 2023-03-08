unit uincreaseresolutionlearn;
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuraldatasets, neuraldatasetsv, fpimage, IntfGraphics, lcltype, Buttons,
  fgl, neuralnetwork, neuralvolume;

type
  { TForm1 }
  TForm1 = class(TForm)
    BitBtn7: TBitBtn;
    Image1: TImage;
    ImageHE: TImage;
    ImageVE: TImage;
    ImageGray: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    procedure BitBtn7Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    FWantToQuit: boolean;

    function CheckBinFiles():boolean;
  end;

  { TEvolutionary }


var
  Form1: TForm1;

implementation
{$R *.lfm}

uses uresizeutil;

{ TForm1 }

procedure TForm1.BitBtn7Click(Sender: TObject);
var
  // Aux Neural Network used only to Maxpool
  NNMaxPool: TNNet;
  // Main Neural Network
  NN: TNNet;
  Img: TTinyImage;
  cifarFile: TTInyImageFile;
  I: integer;
  ImgVolumes: TNNetVolumeList;
  Volume: TNNetVolume;
  pOutput, vOutput, vDisplay: TNNetVolume;
  ErrorSum: TNeuralFloat;
  TB: TNNetLayer;
begin
  BitBtn7.Enabled := false;

  if not(CheckBinFiles()) then
  begin
    BitBtn7.Enabled := true;
    exit;
  end;

  writeln('Creating Neural Network...');

  NNMaxPool := TNNet.Create();
  NNMaxPool.AddLayer( TNNetInput.Create(32,32,3) );
  NNMaxPool.AddLayer( TNNetMaxPool.Create(2) );

  if (FileExists('increase-resolution-autosave.nn')) then
  begin
    NN := TNNet.Create();
    WriteLn('Loading previously trained network.');
    NN.LoadFromFile('increase-resolution-autosave.nn');
  end
  else
  begin
    NN := CreateResizingNN(16,16);
  end;

  TB := NN.Layers[1];
  pOutput := TNNetVolume.Create(1,1,1);
  vOutput := TNNetVolume.Create(1,1,1);
  vDisplay:= TNNetVolume.Create(1,1,1);

  writeln('Loading Images...');
  AssignFile(cifarFile, 'data_batch_1.bin');
  Reset(cifarFile);
  I := 1;

  Label1.Caption := 'Loading training set.';

  ImgVolumes := TNNetVolumeList.Create();
  while not EOF(cifarFile) do
  begin
    Read(cifarFile, Img);
    Volume := TNNetVolume.Create();
    LoadTinyImageIntoNNetVolume(Img, Volume);
    Volume.Divi(128);

    ImgVolumes.Add(Volume);

    if (I mod 100 = 0) then
    begin
      LoadTinyImageIntoTImage(Img, Image1);
      LoadTinyImageIntoTImage(Img, ImageGray);
      LoadTinyImageIntoTImage(Img, ImageHE);
      Image1.Width := 128;
      Image1.Height := 128;
      ImageGray.Width := 128;
      ImageGray.Height := 128;
      ImageHE.Width := 128;
      ImageHE.Height := 128;
      ImageVE.Width := 128;
      ImageVE.Height := 128;
      Application.ProcessMessages;
    end;
    inc(I);
  end;
  CloseFile(cifarFile);

  NN.SetLearningRate(0.0001,0.9);
  NN.DebugWeights();
  WriteLn('Neural network has: ');
  WriteLn(' Layers: ', NN.CountLayers()  );
  WriteLn(' Neurons:', NN.CountNeurons() );
  WriteLn(' Weights:' ,NN.CountWeights() );
  WriteLn('Computing...');

  while not (FWantToQuit) do
  begin
    ErrorSum := 0;
    for I := 0 to ImgVolumes.Count - 1 do
    begin
      // Creates the input image.
      NNMaxPool.Compute(ImgVolumes[I]);
      NNMaxPool.GetOutput(pOutput);

      // Runs the main neural network.
      NN.Compute(pOutput);
      NN.GetOutput(pOutput);

      vOutput.Copy(ImgVolumes[I]);

      ErrorSum += vOutput.SumDiff(pOutput);
      NN.Backpropagate(vOutput);

      if ( Random(25) = 0 ) then
      begin
        Label1.Caption := 'Learning: '+ csTinyImageLabel[ ImgVolumes[I].Tag ];

        vDisplay.Copy(pOutput);
        vDisplay.Mul(128);
        LoadNNetVolumeIntoTinyImage(vDisplay, Img);
        LoadTinyImageIntoTImage(Img, ImageGray);
        NN.DebugWeights();
        NN.DebugErrors();

        vDisplay.Copy(TB.Output);
        vDisplay.Mul(128);
        LoadNNetVolumeIntoTinyImage(vDisplay, Img);
        LoadTinyImageIntoTImage(Img, ImageHE);
      end;
      Application.ProcessMessages();

      if (FWantToQuit) then break;

    end; // of imgvolumes for

    NN.SaveToFile('increase-resolution-autosave.nn');
    NN.DebugWeights();
    NN.DebugErrors();
    NN.ClearTime();
  end;

  vDisplay.Free;
  NNMaxPool.Free;
  NN.Free;
  vOutput.Free;
  pOutput.Free;
  ImgVolumes.Free;
  BitBtn7.Enabled := true;
end;

procedure TForm1.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  FWantToQuit := true;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  FWantToQuit := false;
  CheckBinFiles();
end;

function TForm1.CheckBinFiles: boolean;
begin
  Result := true;
  if not (FileExists('data_batch_1.bin')) then
  begin
    Result := false;
    ShowMessage('CIFAR-10 files have not been found.' + Chr(13) +
      'Please download from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz');
  end;
end;

end.

