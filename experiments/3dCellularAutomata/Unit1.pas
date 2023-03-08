unit Unit1;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, Buttons;

const
  csMaxX = 100;
  csMaxY = 100;
  csMaxZ = 5;

type
  TVideoBuffer = array[0..csMaxX, 0..csMaxY, 0..csMaxZ] of byte;
  {TCor=(clBlack,clAqua,clGreen,clYellow); }
  TForm1 = class(TForm)
    Image1: TImage;
    Image2: TImage;
    Image0: TImage;
    Image3: TImage;
    Image4: TImage;
    Image5: TImage;
    Timer1: TTimer;
    procedure StBtnClick(Sender: TObject);
    procedure ClBtnClick(Sender: TObject);
    procedure ReStBtnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure Timer1Timer(Sender: TObject);
  private
    procedure InitPb;
    procedure RepaintAll;
    procedure RePaintAdd;
    { Private declarations }
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

function visit(x, y, z: integer): byte;
var
  co: byte;
  cx, cy, cz: integer;
begin
  co := 0;
  for cx := 0 to 2 do
    for cy := 0 to 2 do
      for cz := 0 to 2 do
        Inc(co, VBI[(csMaxX + x + cx) mod (csMaxX + 1),
          (csMaxY + y + cy) mod (csMaxY + 1),
          (csMaxZ + z + cz) mod (csMaxZ + 1)]);
  Dec(co, VBI[x, y, z]);

{inc(co,VBI[x,y,(csMaxZ+z+2) mod (csMaxZ+1)]);
inc(co,VBI[x,y,(csMaxZ+z) mod (csMaxZ+1)]);

inc(co,VBI[x, (csMaxY+y+2) mod (csMaxY+1),z ]);
inc(co,VBI[x, (csMaxY+y) mod (csMaxY+1),z ]);

inc(co,VBI[ (csMaxX+x+2) mod (csMaxX+1) , y, z]);
inc(co,VBI[ (csMaxX+x) mod (csMaxX+1) , y, z]);}

  visit := co;
end;

procedure AddRan;
var
  I, I2, I3: word;
begin
  for I := 0 to 30 do
    for I2 := 0 to 30 do
      for I3 := 1 to 1 do
        if (visit(I + 100, I2 + 50, I3) = 0) and (VBI[I + 100, I + 50, I3] = 0) and
          (random(61) > ln(I + I2 + 2) * 10) then
        begin
          VBI[I + 100, I2 + 50, I3] := round(random(3) + 1);
        end;
end;

procedure CalculateStep;
{ calculate the life and dead cels }
var
  I, I2, I3: word;
  V: byte;
begin
  for I := 0 to csMaxX do
    for I2 := 0 to csMaxY do
      for I3 := 0 to csMaxZ do
      begin
        V := visit(I, I2, I3);
        VBF[I, I2, I3] := V mod 3;
      end;  { of for }
  Application.ProcessMessages;
end; { of procedure }

procedure TFORM1.RePaintAdd;
{ display only the modified cels }
var
  I, I2: word;
begin
  for I := 0 to csMaxX do
    for I2 := 0 to csMaxY do
    begin
      if (VBI[I, I2, 0] <> VBF[I, I2, 0]) then
        Image0.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 0]];

      if (VBI[I, I2, 1] <> VBF[I, I2, 1]) then
        Image1.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 1]];

      if (VBI[I, I2, 2] <> VBF[I, I2, 2]) then
        Image2.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 2]];

      if (VBI[I, I2, 3] <> VBF[I, I2, 3]) then
        Image3.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 3]];

      if (VBI[I, I2, 4] <> VBF[I, I2, 4]) then
        Image4.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 4]];

      if (VBI[I, I2, 5] <> VBF[I, I2, 5]) then
        Image5.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 5]];
    end;
  VBI := VBF;
  Application.ProcessMessages;
end; { of procedure }

procedure TForm1.RepaintAll;
{ display all cels }
var
  I, I2: word;
begin
  VBI := VBF;
  for I := 0 to csMaxX do
    for I2 := 0 to csMaxY do
    begin
      Image0.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 0]];
      Image1.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 1]];
      Image2.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 2]];
      Image3.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 3]];
      Image4.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 4]];
      Image5.Canvas.Pixels[I, I2] := TcorA[VBF[I, I2, 5]];
    end;
end;

procedure TForm1.InitPb;
{ put the starting cels }
var
  I, I2: word;
  R: byte;
begin
  randomize;

  for I := 0 to csMaxX do
    for I2 := 0 to csMaxY do
      if random(10) > 9 then
      begin
        R := round(random(3) + 1);
        Image1.Canvas.Pixels[I, I2] := TcorA[R];
        VBI[I, I2, 0] := R;
      end
      else
      begin
        Image0.Canvas.Pixels[I, I2] := TcorA[0];
        Image1.Canvas.Pixels[I, I2] := TcorA[0];
        Image2.Canvas.Pixels[I, I2] := TcorA[0];
        Image3.Canvas.Pixels[I, I2] := TcorA[0];
        Image4.Canvas.Pixels[I, I2] := TcorA[0];
        Image5.Canvas.Pixels[I, I2] := TcorA[0];
        VBI[I, I2, 0] := 0;
      end;
end;


procedure TForm1.StBtnClick(Sender: TObject);
begin
  InitPb;
  VBI[50, 50, 2] := 1;
end;

procedure TForm1.ClBtnClick(Sender: TObject);
begin
  Close;
end;

procedure TForm1.ReStBtnClick(Sender: TObject);
var
  I: integer;
begin
  for I := 1 to 500 do
  begin
    CalculateStep;
    RePaintAdd;
    Application.ProcessMessages;
  end;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  Timer1.Enabled := False;
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
  CalculateStep;
  RePaintAdd;
end;

end.
