// This is under development - DO NOT USE IT.
{
unit
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

unit uviewinnerpatterns;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume, MTProcs, math, neuralfit;

type
  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    GrBoxNeurons: TGroupBox;
    GrBoxNeurons2: TGroupBox;
    GrBoxNeurons3: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    procedure ButLearnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure OnEpoch(Sender: TObject);
    procedure OnStep(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    globalImgInput: TNNetVolume;
    ImgTrainingVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    aImage, aImage2, aImage3: array of TImage;
    Layer2, Layer3: TNNetNeuronList;

    iEpochCount, iEpochCountAfterLoading: integer;

    // TODO: Give better names to these properties
    FNN: TNNet;
    vDisplay: TNNetVolume;

    FFit: TNeuralImageFit;
    FFirstNeuronalLayerIdx: integer;
    FFilterSize: integer;
    FImagesPerRow: integer;
    FNeuronNum, FNeuronNum2, FNeuronNum3: integer;
    procedure RebuildNeuronList;
    procedure Learn(Sender: TObject);
    procedure EnableComponents(flag: boolean);
    procedure SaveScreenshot(filename: string);
    procedure SaveNeuronsImage(filename: string);
    (*procedure ShowNeurons(pNeuronList: TNNetNeuronList;
      var pImage: TImageDynArr;
      startImage, filterSize, color_encoding: integer;
      ScalePerImage: boolean);*)
  public
    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType, neuraldatasetsv;

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
end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  if FFit.Running then
  begin
    WriteLn('Requesting to stop threads.');
    FFit.ShouldQuit := true;
  end;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  ImgValidationVolumes := nil;
  ImgTestVolumes := nil;
  ImgTrainingVolumes := nil;
  FFit := TNeuralImageFit.Create();
  FRunning := false;
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  if FFit.Running then
  begin
    WriteLn('Waiting to stop threads.');
    FFit.WaitUntilFinished;
  end;
  FFit.Free;
  FRunning := false;
  if Assigned(ImgValidationVolumes) then ImgValidationVolumes.Free;
  if Assigned(ImgTestVolumes) then ImgTestVolumes.Free;
  if Assigned(ImgTrainingVolumes) then ImgTrainingVolumes.Free;
  ImgValidationVolumes := nil;
  ImgTestVolumes := nil;
  ImgTrainingVolumes := nil;
end;

procedure TFormVisualLearning.OnEpoch(Sender: TObject);
var
  ScalePerImage: boolean;
begin
  RebuildNeuronList;
  ScalePerImage := true;
  ShowNeurons(FNN.Layers[FFirstNeuronalLayerIdx].Neurons, aImage, {startImage=}0, {filterSize=}32, csEncodeRGB, ScalePerImage);
  ShowNeurons(Layer2, aImage2, {startImage=}0, {filterSize=}FFilterSize, csEncodeRGB, ScalePerImage);
  ShowNeurons(Layer3, aImage3, {startImage=}0, {filterSize=}FFilterSize, csEncodeRGB, ScalePerImage);

  Application.ProcessMessages;
  SaveScreenshot('autosave.bmp');
end;

procedure TFormVisualLearning.OnStep(Sender: TObject);
begin
  LabClassRate.Caption := PadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+'%',4);
  Application.ProcessMessages;
end;

procedure TFormVisualLearning.RebuildNeuronList;
var
  ReLU: boolean;
  Threshold: TNeuralFloat;
begin
  ReLU := false;
  Threshold := 0.5;

  RebuildNeuronListOnPreviousPatterns
  (
    {CalculatedLayer=} Layer2,
    {CurrentLayer=}FNN.Layers[3].Neurons,
    {PrevLayer=}FNN.Layers[1].Neurons,
    {PrevStride=}5,
    {ReLU=}ReLU,
    {Threshold=}Threshold
  );

  RebuildNeuronListOnPreviousPatterns
  (
    {CalculatedLayer=} Layer3,
    {CurrentLayer=}FNN.Layers[5].Neurons,
    {PrevLayer=}Layer2,
    {PrevStride=}15 {3*5},
    {ReLU=}ReLU,
    {Threshold=}Threshold
  );
end;

procedure TFormVisualLearning.Learn(Sender: TObject);

  procedure DisplayInputImage(color_encoding: integer);
  begin
    vDisplay.Copy(globalImgInput);

    vDisplay.NeuronalInputToRgbImg(color_encoding);

    LoadVolumeIntoTImage(vDisplay, ImgSample);
    ImgSample.Width := 64;
    ImgSample.Height := 64;
  end;

var
  aLabelX, aLabelY: array of TLabel;
  aLabelX2, aLabelY2: array of TLabel;
  aLabelX3, aLabelY3: array of TLabel;

begin
  if not(Assigned(ImgTrainingVolumes)) then
  begin
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);
  end;
  // This experiment doesn't require validation nor testing.
  ImgValidationVolumes.Clear();
  ImgTestVolumes.Clear();

  vDisplay := TNNetVolume.Create();
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  WriteLn('Creating Neural Network...');
  FNN := TNNet.Create();

  FNN.AddLayer([
     TNNetInput.Create(32,32,3),
     TNNetConvolutionReLU.Create({NumFeatures=}128, {pFeatureSize=}5, {pInputPadding=}2, {pStride=}5),
     TNNetMulLearning.Create(5),
     TNNetConvolutionReLU.Create({NumFeatures=} 64, {pFeatureSize=}3, {pInputPadding=}1, {pStride=}1),
     TNNetMulLearning.Create(2),
     TNNetConvolutionReLU.Create({NumFeatures=}128, {pFeatureSize=}1, {pInputPadding=}0, {pStride=}1),
     TNNetMulLearning.Create(2),
     TNNetFullConnectReLU.Create(64),
     TNNetFullConnectReLU.Create(64),
     TNNetFullConnectLinear.Create(10),
     //TNNetPointwiseConvLinear.Create(10),
     //TNNetAvgChannel.Create(),
     TNNetSoftMax.Create()
  ]);

  WriteLn('Initializing weights...');
  FNN.Layers[1].InitBasicPatterns();
  FNN.Layers[3].Neurons.InitForDebug();
  FNN.Layers[5].Neurons.InitForDebug();

  FNN.DebugStructure();

  FFirstNeuronalLayerIdx := 1;
  FFilterSize := 48;
  FImagesPerRow := 6;
  FNeuronNum  := FNN.Layers[1].Neurons.Count;
  FNeuronNum2 := FNN.Layers[3].Neurons.Count;
  FNeuronNum3 := FNN.Layers[5].Neurons.Count;

  Layer2 := TNNetNeuronList.CreateWithElements(FNeuronNum2);
  Layer3 := TNNetNeuronList.CreateWithElements(FNeuronNum3);

  RebuildNeuronList;

  CreateNeuronImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {firstNeuronalLayer=}FNN.Layers[FFirstNeuronalLayerIdx].Neurons,
    {filterSize=}32,
    {imagesPerRow=}8,
    {NeuronNum=}FNeuronNum
  );

  CreateNeuronImages
  (
    GrBoxNeurons2,
    aImage2, aLabelX2, aLabelY2,
    {firstNeuronalLayer=}Layer2,
    {filterSize=}FFilterSize,
    {imagesPerRow=}FImagesPerRow,
    {NeuronNum=}FNeuronNum2
  );

  CreateNeuronImages
  (
    GrBoxNeurons3,
    aImage3, aLabelX3, aLabelY3,
    {firstNeuronalLayer=}Layer3,
    {filterSize=}FFilterSize,
    {imagesPerRow=}FImagesPerRow * 2,
    {NeuronNum=}FNeuronNum3
  );

  FormVisualLearning.Width := GrBoxNeurons3.Left + GrBoxNeurons3.Width + 10;
  FormVisualLearning.Height := GrBoxNeurons3.Top + GrBoxNeurons3.Height + 10;
  Application.ProcessMessages;

  Self.OnEpoch(Sender);

  FFit.OnAfterEpoch := @Self.OnEpoch;
  FFit.OnAfterStep := @Self.OnStep;
  FFit.L2Decay := 0.0001;
  FFit.InitialLearningRate := 0.002;
  FFit.MultipleSamplesAtValidation := false;
  //FFit.MaxThreadNum := 1;
  FFit.HasImgCrop := false;
  FFit.Fit(FNN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 10, {BatchSize=}64, {Epochs=}350);

  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  FreeNeuronImages(aImage2, aLabelX2, aLabelY2);
  FreeNeuronImages(aImage3, aLabelX3, aLabelY3);

  Layer2.Free;
  vDisplay.Free;
  FNN.Free;
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

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.

