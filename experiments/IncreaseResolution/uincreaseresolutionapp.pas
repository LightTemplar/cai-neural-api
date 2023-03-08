unit uincreaseresolutionapp;
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuraldatasets, neuraldatasetsv, fpimage, IntfGraphics, lcltype, Buttons,
  neuralnetwork, neuralvolume, neuralvolumev;

type
  { TForm1 }
  TForm1 = class(TForm)
    BitBtn13: TBitBtn;
    CheckPause: TCheckBox;
    Image1: TImage;
    ImageHE: TImage;
    ImageVE: TImage;
    ImageGray: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Label5: TLabel;
    procedure BitBtn13Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    FWantQuit: boolean;
    function CheckBinFiles: boolean;
  public

  end;

  { TEvolutionary }


var
  Form1: TForm1;

implementation
uses uresizeutil, Math;

{$R *.lfm}

{ TForm1 }

procedure FixBorders(V:TNNetVolume);
var
  I, J: integer;
  XMax, YMax: integer;
begin
  XMax := V.SizeX - 1;
  YMax := V.SizeY - 1;

  for I := 0 to XMax do
  begin
    V[I,0,0] := Max(V[I,0,0],V[I,1,0]);
    V[I,0,1] := Max(V[I,0,1],V[I,1,1]);
    V[I,0,2] := Max(V[I,0,2],V[I,1,2]);

    V[I,YMax,0] := Max(V[I,YMax,0],V[I,YMax-1,0]);
    V[I,YMax,1] := Max(V[I,YMax,1],V[I,YMax-1,1]);
    V[I,YMax,2] := Max(V[I,YMax,2],V[I,YMax-1,2]);
  end;

  for J := 0 to YMax do
  begin
    V[0,J,0] := Max(V[0,J,0],V[1,J,0]);
    V[0,J,1] := Max(V[0,J,1],V[1,J,1]);
    V[0,J,2] := Max(V[0,J,2],V[1,J,2]);

    V[XMax,J,0] := Max(V[XMax,J,0],V[XMax-1,J,0]);
    V[XMax,J,1] := Max(V[XMax,J,1],V[XMax-1,J,1]);
    V[XMax,J,2] := Max(V[XMax,J,2],V[XMax-1,J,2]);
  end;

end;

procedure TForm1.BitBtn13Click(Sender: TObject);
var
  NN2, NN3, NN4: TNNet;
  Img: TTinyImage;
  cifarFile: TTInyImageFile;
  I, K: integer;
  ImgVolumes: TNNetVolumeList;
  Volume: TNNetVolume;
  vInput, vOutput, vBigOutput, vDisplay: TNNetVolume;
begin
  if not(CheckBinFiles()) then
  begin
    exit;
  end;

  Randomize;
  BitBtn13.Enabled := false;
  writeln('Creating Neural Network...');
  ImgVolumes := TNNetVolumeList.Create();

  NN2 := CreateResizingNN(32,32);
  NN2.LoadDataFromFile('increase-resolution-autosave.nn');

  NN3 := CreateResizingNN(64,64);
  NN3.LoadDataFromFile('increase-resolution-autosave.nn');

  NN4 := CreateResizingNN(128,128);
  NN4.LoadDataFromFile('increase-resolution-autosave.nn');

  writeln('Loading Images...');
  AssignFile(cifarFile, 'data_batch_1.bin');
  Reset(cifarFile);
  I := 1;

  while not EOF(cifarFile) do
  begin
    Read(cifarFile, Img);
    Label1.Caption := csTinyImageLabel[Img.bLabel];
    Volume := TNNetVolume.Create();
    LoadTinyImageIntoNNetVolume(Img, Volume);
    Volume.Divi(128);
    ImgVolumes.Add(Volume);

    if (I mod 100 = 0) then
    begin
      LoadTinyImageIntoTImage(Img, Image1);
      LoadTinyImageIntoTImage(Img, ImageGray);
      LoadTinyImageIntoTImage(Img, ImageHE);
      LoadTinyImageIntoTImage(Img, ImageVE);
      Image1.Width := 128;
      Image1.Height := 128;
      ImageGray.Width := 128;
      ImageGray.Height := 128;
      ImageHE.Width := 128;
      ImageHE.Height := 128;
      ImageVE.Width := 256;
      ImageVE.Height := 256;
      Application.ProcessMessages;
    end;
    inc(I);
  end;
  CloseFile(cifarFile);

  vInput     := TNNetVolume.Create(32,32,3);
  vOutput    := TNNetVolume.Create(64,64,3);
  vDisplay   := TNNetVolume.Create();
  vBigOutput := TNNetVolume.Create(64,64,3);

  WriteLn('128x128 -> 256x256 Neural network has: ');
  WriteLn(' Layers: ', NN4.CountLayers()  );
  WriteLn(' Neurons:', NN4.CountNeurons() );
  WriteLn(' Weights:' ,NN4.CountWeights() );
  WriteLn('Computing...');

  while not (FWantQuit) do
  begin
    I := random(10000);
    vInput.Copy(ImgVolumes[I]);

    Label1.Caption := csTinyImageLabel[ ImgVolumes[I].Tag ];

    vDisplay.Copy(vInput);
    vDisplay.Mul(128);
    LoadNNetVolumeIntoTinyImage(vDisplay, Img);
    LoadTinyImageIntoTImage(Img, Image1);

    NN2.Compute(vInput); NN2.GetOutput(vBigOutput);

    vDisplay.Copy(vBigOutput);
    vDisplay.Mul(128);
    LoadVolumeIntoTImage(vDisplay, ImageGray);
    ImageGray.Width := 128;
    ImageGray.Height := 128;

    NN3.Compute(vBigOutput);  NN3.GetOutput(vBigOutput);

    vDisplay.Copy(vBigOutput);
    vDisplay.Mul(128);
    LoadVolumeIntoTImage(vDisplay, ImageHE);
    ImageHE.Width := 128;
    ImageHE.Height := 128;

    NN4.Compute(vBigOutput);  NN4.GetOutput(vBigOutput);

    vDisplay.Copy(vBigOutput);
    vDisplay.Mul(128);
    LoadVolumeIntoTImage(vDisplay, ImageVE);
    ImageVE.Width := 256;
    ImageVE.Height := 256;

    for K := 1 to 50 do
    begin
      Application.ProcessMessages();
      Sleep(100);
    end;

    while CheckPause.Checked do
    begin
      Application.ProcessMessages();
      Sleep(100);
    end;
  end;

  NN2.Free;
  NN3.Free;
  NN4.Free;
  vBigOutput.Free;
  vDisplay.Free;
  vInput.Free;
  vOutput.Free;
  ImgVolumes.Free;
  BitBtn13.Enabled := true;
end;

procedure TForm1.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  FWantQuit := true;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  FWantQuit := false;
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

  if not (FileExists('increase-resolution-autosave.nn')) then
  begin
    Result := false;
    ShowMessage('Please run IncreaseResolutionLearning first.' + Chr(13) + Chr(13) +
      'The learning executable will create a neural network file (increase-resolution-autosave.nn) used by this application.');
  end;
end;

end.

