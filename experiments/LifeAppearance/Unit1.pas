unit Unit1;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, Buttons;

type
  TVideoBuffer = array[0..639, 0..479] of byte;

  TForm1 = class(TForm)
    PB: TPaintBox;
    Timer1: TTimer;
    procedure StBtnClick(Sender: TObject);
    procedure ClBtnClick(Sender: TObject);
    procedure ReStBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure CheckBox1Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure Timer1Timer(Sender: TObject);
  private
    WantQuit: boolean;
    procedure InitPb;
    procedure RepaintAll;
    procedure RePaintAdd;
  public
    { Public declarations }
  end;

const
  TcorA: array[0..3] of Tcolor =
    (clBlack, clAqua, clGreen, clYellow);

var
  Form1: TForm1;

implementation

{$R *.lfm}
var
  VBI, VBF: TVideoBuffer; { video buffer Initial/Final}

function visit(x, y: integer): byte;
var
  co: byte;
begin
  co := 0;
  Inc(co, VBI[x - 1, y - 1]);
  Inc(co, VBI[x, y - 1]);
  Inc(co, VBI[x + 1, y - 1]);

  Inc(co, VBI[x - 1, y]);
  Inc(co, VBI[x + 1, y]);

  Inc(co, VBI[x - 1, y + 1]);
  Inc(co, VBI[x, y + 1]);
  Inc(co, VBI[x + 1, y + 1]);
  visit := co;
end;

procedure AddRan;
var
  I, I2: word;
begin
  for I := 0 to 180 do
    for I2 := 0 to 180 do
      if (visit(I + 300, I2 + 200) = 0) and (VBI[I + 300, I + 200] = 0) and
        (random(61) > ln(I + I2 + 2) * 10) then
      begin
        VBI[I + 300, I2 + 200] := round(random(3) + 1);
      end;
end;

procedure CalculateStep;
{ calculate the life and dead cels }
var
  I, I2: word;
  V, R: byte;
begin
  for I := 1 to 639 do
    for I2 := 1 to 479 do
    begin
      R := VBI[I, I2];
      V := visit(I, I2);
      if R > 0 then {alive}
      begin
        if not ({(R+4=V) or (R+2=V) or (R+3=V)}
          (R + 5 > V) and (V > R + 1)) then
          VBF[I, I2] := 0;
      end
      else
      begin
        if (V = 7) or (V = 5) or (V = 6{9}) then
          VBF[I, I2] := (V mod 3) + 1;
      end;  { external if }

    end;  { of for }
  Application.ProcessMessages;
end; { of procedure }

procedure TFORM1.RePaintAdd;
{ display only the modified cels }
var
  I, I2: word;
begin
  for I := 0 to 639 do
    for I2 := 0 to 479 do
      if VBI[I, I2] <> VBF[I, I2] then
      begin
        VBI[I, I2] := VBF[I, I2];
        PB.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2]];
      end;
  Application.ProcessMessages;
end; { of procedure }

procedure TForm1.RepaintAll;
{ display all cels }
var
  I, I2: word;
begin
  VBI := VBF;
  for I := 0 to 639 do
    for I2 := 0 to 479 do
      PB.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2]];
end;

procedure TForm1.InitPb;
{ put the starting cels }
var
  I, I2: word;
  R: byte;
begin
  randomize;
  for I := 0 to 639 do
    for I2 := 0 to 479 do
      if random(10) > 8 then
      begin
        R := round(random(3) + 1);
        PB.Canvas.Pixels[I, I2] := TcorA[R];
        VBI[I, I2] := R;
      end
      else
      begin
        PB.Canvas.Pixels[I, I2] := TcorA[0];
        VBI[I, I2] := 0;
      end;
end;


procedure TForm1.StBtnClick(Sender: TObject);
var
  I: integer;
begin
  InitPb;
  for I := 1 to 1 do
  begin
    CalculateStep;
    RePaintAdd;
  end;
  ReStBtnClick(Sender);
end;

procedure TForm1.ClBtnClick(Sender: TObject);
begin
  Close;
end;

procedure TForm1.ReStBtnClick(Sender: TObject);
var
  I: longint;
begin
  I := 0;
  while not (WantQuit) do
  begin
    I := (I + 1) mod 10;
    CalculateStep;
    RePaintAdd;
    if I = 0 then
      AddRan;
    Application.ProcessMessages;
  end;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  WantQuit := False;
end;

procedure TForm1.CheckBox1Click(Sender: TObject);
begin
  //WantQuit:=CheckBox1.Checked;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  WantQuit := True;
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
  Timer1.Enabled := False;
  StBtnClick(Sender);
end;

end.
