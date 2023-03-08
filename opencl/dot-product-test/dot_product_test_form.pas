unit dot_product_test_form;
{
 Lazarus / Free Pascal / OpenCL Example
 This code implements an experiment:
 A vectors (A1, A2, A3, ...) are operated with a number of
 B vectors (B1, B2, B3, ...) via dot product. There is a resulting vector
 R with all dot products A1.B1 .. A1.B2 .. A2.B1 .. AN.BN .

 This program compares the speed of OpenCL against TNNetVolume (AVX).

 This prototype is part of the Conscious Artificial Intelligence project
 located at:
 https://sourceforge.net/projects/cai/
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  Menus, neuralopencl, neuralopenclv, neuralvolume, neuralnetwork;

type

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    ButProbe: TButton;
    ComboKernel: TComboBox;
    ComboPlatform: TComboBox;
    ComboDevType: TComboBox;
    Ed3: TEdit;
    LabTiledResult: TLabel;
    LabTiled: TLabel;
    LabInterleavedResult: TLabel;
    LabInterleaved: TLabel;
    LabVolumeResult: TLabel;
    LabVolumeTime: TLabel;
    LabText2: TLabel;
    LabText1: TLabel;
    LabPlatform: TLabel;
    LabelDevType: TLabel;
    LabKernel: TLabel;
    LabResult: TLabel;
    LabTotal: TLabel;
    RB6: TRadioButton;
    RB5: TRadioButton;
    RB4: TRadioButton;
    RB3: TRadioButton;
    RB7: TRadioButton;
    RBBillion: TRadioButton;
    RBTrillion: TRadioButton;
    procedure ButProbeClick(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure ComboDevTypeChange(Sender: TObject);
    procedure ComboKernelChange(Sender: TObject);
    procedure ComboPlatformChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure RBBillionChange(Sender: TObject);
  private
    { private declarations }
    FOpenCL: TEasyOpenCLCL;
    VA_NUM, VB_NUM: integer;
    VECTOR_SIZE: integer;
    LOOP_COUNT: integer;
    procedure testOpenCL;
    procedure testVolume;
    procedure testTiledVolume(TILE_SIZE: integer);
    procedure testInterleavedVolume;
    procedure testInterleavingSpeed;
  public
    { public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.lfm}

procedure TForm1.testOpenCL;
var
  startTime, finishTime, totalTimeSeconds: double;
  A,B,R: TNNetVolume;
  DotProductObj: TDotProductCL;
  i: integer;
  InterleavedA: TNNetVolume;
begin
  A := TNNetVolume.Create(VA_NUM, 1, VECTOR_SIZE);
  B := TNNetVolume.Create(VB_NUM, 1, VECTOR_SIZE);
  R := TNNetVolume.Create(VA_NUM * VB_NUM);
  InterleavedA := TNNetVolume.Create(VECTOR_SIZE,1,VA_NUM);

  A.Fill(1);
  B.Fill(2);
  InterleavedA.InterleaveWithXFrom(A,VECTOR_SIZE);

  DotProductObj := TDotProductCL.Create(FOpenCL.CurrentPlatform, FOpenCL.CurrentDevice);
  if (ComboKernel.Text = 'cai_dot_product') or (ComboKernel.Text = 'cai_dot_product2') or (ComboKernel.Text = 'cai_dot_product_simple')
    then DotProductObj.PrepareForCompute(InterleavedA, B, VECTOR_SIZE, ComboKernel.Text)
    else DotProductObj.PrepareForCompute(InterleavedA, B, VECTOR_SIZE, ComboKernel.Text, 16, 16);
  DotProductObj.HideMessages();

  startTime := now();
  Write('OpenCL input:'); A.PrintDebug(); WriteLn;
  for i := 1 to LOOP_COUNT do
  begin
    //InterleavedA.InterleaveWithXFrom(A,VECTOR_SIZE);
    DotProductObj.Compute(InterleavedA, B, 0);
    DotProductObj.FinishAndLoadResult(R);
    //R.DotProducts(VA_NUM, VB_NUM, VECTOR_SIZE, A, B);
    //WriteLn(i:3,' Total OpenCL Sum:', R.GetSum():10:5);
  end;
  finishTime := now();

  Write('OpenCL interleaved input:'); InterleavedA.PrintDebug(); WriteLn;
  //InterleavedA.Print();
  //R.Print();

  totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

  writeln('Total OpenCL run time:', (totalTimeSeconds): 10: 6, ' seconds.');
  Write('OpenCL '); R.PrintDebug(); WriteLn;

  LabResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';
  Application.ProcessMessages();

  // Shutdown and cleanup
  InterleavedA.Free;
  A.Free;
  B.Free;
  R.Free;
  DotProductObj.Free;
end;

procedure TForm1.testVolume;
var
  A,B: TNNetVolumeList;
  R: TNNetVolume;
  CountA, CountB: integer;
  startTime, finishTime, totalTimeSeconds: double;
  i: integer;
begin
  A := TNNetVolumeList.Create();
  B := TNNetVolumeList.Create();
  R := TNNetVolume.Create(VA_NUM, VB_NUM, 1);

  A.AddVolumes(VA_NUM,VECTOR_SIZE,1,1,1);
  B.AddVolumes(VB_NUM,VECTOR_SIZE,1,1,1);

  A.Fill(1);
  B.Fill(2);

  startTime := now();
  for I := 1 to LOOP_COUNT do
  begin
    for CountB := 0 to VB_NUM - 1 do
    begin
      for CountA := 0 to VA_NUM - 1 do
      begin
        R[CountA, CountB, 0] := A[CountA].DotProduct(B[CountB]);
      end;
    end;
  end;

  finishTime := now();
  Write('VOLUME:'); R.PrintDebug(); WriteLn;

  totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

  writeln('Total VOLUME run time:', (totalTimeSeconds): 10: 6, ' seconds.');
  LabVolumeResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';
  Application.ProcessMessages();
  R.Free;
  B.Free;
  A.Free;
end;

procedure TForm1.testTiledVolume(TILE_SIZE: integer);
var
  A,B: TNNetVolumeList;
  R: TNNetVolume;
  CountA, CountB: integer;
  startTime, finishTime, totalTimeSeconds: double;
  i: integer;
  VA_TILE_NUM, VB_TILE_NUM: integer;
  VA_TILE_CNT, VB_TILE_CNT: integer;
  VA_TILE_START, VB_TILE_START: integer;
  VA_TILE_FINISH, VB_TILE_FINISH: integer;
begin
  A := TNNetVolumeList.Create();
  B := TNNetVolumeList.Create();
  R := TNNetVolume.Create(VA_NUM, VB_NUM, 1);

  A.AddVolumes(VA_NUM,VECTOR_SIZE,1,1,1);
  B.AddVolumes(VB_NUM,VECTOR_SIZE,1,1,1);

  A.Fill(1);
  B.Fill(2);

  startTime := now();
  VA_TILE_NUM := VA_NUM  div TILE_SIZE;
  VB_TILE_NUM := VB_NUM  div TILE_SIZE;

  for I := 1 to LOOP_COUNT do
  begin
    for VA_TILE_CNT := 0 to VA_TILE_NUM - 1 do
    begin
      VA_TILE_START := VA_TILE_CNT * TILE_SIZE;
      VA_TILE_FINISH := VA_TILE_START + TILE_SIZE - 1;
      for VB_TILE_CNT := 0 to VB_TILE_NUM - 1 do
      begin
        VB_TILE_START := VB_TILE_CNT * TILE_SIZE;
        VB_TILE_FINISH := VB_TILE_START + TILE_SIZE - 1;
        for CountB := VB_TILE_START to VB_TILE_FINISH do
        begin
          for CountA := VA_TILE_START to VA_TILE_FINISH do
          begin
            R[CountA, CountB, 0] := A[CountA].DotProduct(B[CountB]);
          end;
        end;
      end;
    end;
  end;

  finishTime := now();
  Write('TILED VOLUME:'); R.PrintDebug(); WriteLn;

  totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

  writeln('Total TILED ',IntToStr(TILE_SIZE),'x',IntToStr(TILE_SIZE),' VOLUME run time:', (totalTimeSeconds): 10: 6, ' seconds.');
  LabTiledResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';
  Application.ProcessMessages();
  R.Free;
  B.Free;
  A.Free;
end;

procedure TForm1.testInterleavedVolume;
var
  A,B,R: TNNetVolumeList;
  InterleavedA, ConcatedA: TNNetVolume;
  CountB: integer;
  startTime, finishTime, totalTimeSeconds: double;
  i: integer;
begin
  A := TNNetVolumeList.Create();
  B := TNNetVolumeList.Create();
  R := TNNetVolumeList.Create();

  InterleavedA := TNNetVolume.Create(VA_NUM * VECTOR_SIZE);
  ConcatedA := TNNetVolume.Create(VA_NUM * VECTOR_SIZE);

  A.AddVolumes(VA_NUM,VECTOR_SIZE,1,1,0);
  B.AddVolumes(VB_NUM,VECTOR_SIZE,1,1,0);
  R.AddVolumes(VB_NUM,VA_NUM,1,1,0);

  A.Fill(1);
  B.Fill(2);
  A.ConcatInto(ConcatedA);

  startTime := now();
  for I := 1 to LOOP_COUNT do
  begin
    InterleavedA.InterleaveWithXFrom(ConcatedA,VECTOR_SIZE);
    for CountB := 0 to VB_NUM - 1 do
    begin
      R[CountB].InterleavedDotProduct(InterleavedA, B[CountB]);
    end;
    //WriteLn(i:3,' Total VOLUME Sum:', R.GetSum():10:5);
  end;
  finishTime := now();

  WriteLn('INTERLEAVED VOLUME Size:', R.GetTotalSize(), ' Avg:', R.GetAvg():10:5);

  totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

  writeln('Total INTERLEAVED VOLUME run time:', (totalTimeSeconds): 10: 6, ' seconds.');
  LabInterleavedResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';
  Application.ProcessMessages();
  ConcatedA.Free;
  InterleavedA.Free;
  R.Free;
  B.Free;
  A.Free;
end;

procedure TForm1.testInterleavingSpeed;
var
  A: TNNetVolumeList;
  InterleavedA, ConcatedA: TNNetVolume;
  startTime, finishTime, totalTimeSeconds: double;
  i: integer;
begin
  A := TNNetVolumeList.Create();

  InterleavedA := TNNetVolume.Create(VA_NUM * VECTOR_SIZE);
  ConcatedA := TNNetVolume.Create(VA_NUM * VECTOR_SIZE);

  A.AddVolumes(VA_NUM,VECTOR_SIZE,1,1,0);

  A.Fill(1);
  A.ConcatInto(ConcatedA);

  startTime := now();
  for I := 1 to LOOP_COUNT*VB_NUM div 10 do
  begin
    InterleavedA.InterleaveWithXFrom(ConcatedA,VECTOR_SIZE);
  end;
  finishTime := now();

  totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

  writeln('Total INTERLEAVING TEST run time:', (totalTimeSeconds): 10: 6, ' seconds.');
  Application.ProcessMessages();
  ConcatedA.Free;
  InterleavedA.Free;
  A.Free;
end;

{ TForm1 }
procedure TForm1.Button1Click(Sender: TObject);
begin
  Button1.Enabled := false;
  LabResult.Caption := 'RUNNING - Please Wait';
  LabVolumeResult.Caption := 'RUNNING - Please Wait';
  LabInterleavedResult.Caption := 'RUNNING - Please Wait';
  LabTiledResult.Caption := 'RUNNING - Please Wait';

  Application.ProcessMessages();

  if RBBillion.Checked then
  begin
    LOOP_COUNT := 1000;
    VB_NUM := 32*32;
    VA_NUM := 64;
    VECTOR_SIZE := 3*3*64;
  end
  else if RBTrillion.Checked then
  begin
    LOOP_COUNT := 10;
    VB_NUM := 32*32*100; {100 batch}
    VA_NUM := 64;
    VECTOR_SIZE := 3*3*64;
  end
  else if RB3.Checked then
  begin
    LOOP_COUNT := 100;
    VB_NUM := StrToInt(Ed3.Text);
    VA_NUM := VB_NUM;
    VECTOR_SIZE := VB_NUM;
  end else if RB4.Checked then
  begin
    LOOP_COUNT := 100;
    VB_NUM := 16*16;
    VA_NUM := 64;
    VECTOR_SIZE := 3*3*64;
  end else if RB5.Checked then
  begin
    LOOP_COUNT := 100;
    VB_NUM := 8*8;
    VA_NUM := 64;
    VECTOR_SIZE := 3*3*64;
  end else if RB6.Checked then
  begin
    LOOP_COUNT := 1000;
    VB_NUM := 1;
    VA_NUM := 392;
    VECTOR_SIZE := 1024;
  end else if RB7.Checked then
  begin
    LOOP_COUNT := 1000;
    VB_NUM := 64;
    VA_NUM := 32*32;
    VECTOR_SIZE := 3*3*64;
  end;

  testOpenCL();
  testVolume();
  testInterleavedVolume();
  testTiledVolume(4);
  testTiledVolume(8);
  testTiledVolume(16);
  testTiledVolume(32);
  testTiledVolume(64);
  TestConvolutionOpenCL(FOpenCL.CurrentPlatform, FOpenCL.CurrentDevice);
  //testInterleavingSpeed();

  Button1.Enabled := true;
end;

procedure TForm1.ComboDevTypeChange(Sender: TObject);
begin
  FOpenCL.ComboDevTypeChange(ComboDevType);
end;

procedure TForm1.ComboKernelChange(Sender: TObject);
begin

end;

procedure TForm1.ComboPlatformChange(Sender: TObject);
begin
  FOpenCL.ComboPlatformChange(ComboPlatform, ComboDevType);
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  FOpenCL := TEasyOpenCLCL.Create();
end;

procedure TForm1.FormDestroy(Sender: TObject);
begin
  FOpenCL.Free;
end;

procedure TForm1.RBBillionChange(Sender: TObject);
begin

end;

procedure TForm1.ButProbeClick(Sender: TObject);
begin
  FOpenCL.printDevicesInfo();
  if FOpenCL.ProbeClick(ComboPlatform, ComboDevType) then
  begin
    Button1.Enabled := True;
  end;
end;


end.
