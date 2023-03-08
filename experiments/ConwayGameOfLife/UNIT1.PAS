unit Unit1;

interface

uses
  Windows, Messages, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, Buttons;

type
  TVideoBuffer=array[0..639,0..479] of byte;
  {TCor=(clBlack,clAqua,clGreen,clYellow); }
  TForm1 = class(TForm)
    PB: TPaintBox;
    StBtn: TBitBtn;
    CheckBox1: TCheckBox;
    ClBtn: TBitBtn;
    VIVLAB: TLabel;
    Timer1: TTimer;
    procedure StBtnClick(Sender: TObject);
    procedure ClBtnClick(Sender: TObject);
    procedure ReStBtnClick(Sender: TObject);
    procedure PBPaint(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
     { Private declarations }
     InTask:boolean;
     procedure InitPb;
     procedure RepaintAll;
     procedure RePaintAdd;
     public
    { Public declarations }
  end;

const TcorA : array[0..6] of Tcolor =
      (clBlack,clNavy,clGreen,clYellow,clPurple,clAqua,clolive);

var
  Form1: TForm1;

implementation
{$R *.DFM}
var VBI,VBF:TVideoBuffer; { video buffer Initial/Final}

function visit(x,y:integer):byte;
var co:byte;
begin
co:=0;
inc(co,VBI[x-1,y-1]);
inc(co,VBI[x  ,y-1]);
inc(co,VBI[x+1,y-1]);

inc(co,VBI[x-1,y]);
inc(co,VBI[x+1,y]);

inc(co,VBI[x-1,y+1]);
inc(co,VBI[x  ,y+1]);
inc(co,VBI[x+1,y+1]);
visit:=co;
end;

procedure AddRan;
var I,I2:word;
var R:byte;
begin
 for I:=0 to 180 do
     for I2:=0 to 180 do
         if (visit(I+300,I2+200)=0) and
            (VBI[I+300,I+200]=0) and
            (random(61)>ln(I+I2+2)*10)
            then begin
                 VBI[I+300,I2+200]:=round(random(6)+1);
                 end
end;

var VIVAS:longint; {numero de células vivas}
procedure CalculateStep;
{ calculate the life and dead cels }
var I,I2:word;
    V,R:byte;
    COUNT:longint; { numero de celulas vivas antes }
begin
 COUNT:=0;
 for I:=1 to 639 do
     for I2:=1 to 479 do
         begin
         R:=VBI[I,I2];
         V:=visit(I,I2);
         if R>0 then {alive}
            begin
            inc(COUNT);
            if not({(R+4=V) or (R+2=V) or (R+3=V)}
                    {(R+5>V) and (V>R+1)}
                    (21>V) and (V>11) ) then
               VBF[I,I2]:=0;
            end
          else
            begin
            if (*(V=7) or (V=5) or (V=6)*)
               (16>V) and (V>9)
              then
               VBF[I,I2]:=(V mod 6)+1;
            end;  { external if }


         end;  { of for }
VIVAS:=COUNT;
end; { of procedure }

procedure TFORM1.RePaintAdd;
{ display only the modified cels }
var I,I2:word;
begin
 for I:=0 to 639 do
     for I2:=0 to 479 do
         if VBI[I,I2]<>VBF[I,I2] then
            begin
            VBI[I,I2]:=VBF[I,I2];
            PB.Canvas.Pixels[I,I2]:=TcorA[VBF[I,I2]];
            end;
end; { of procedure }

procedure TForm1.RepaintAll;
{ display all cels }
var I,I2:word;
begin
VBI:=VBF;
 for I:=0 to 639 do
     for I2:=0 to 479 do
         PB.Canvas.Pixels[I,I2]:=TcorA[VBF[I,I2]];
end;

procedure TForm1.InitPb;
{ put the starting cels }
var I,I2:word;
    R:byte;
begin
randomize;
 for I:=0 to 639 do
     for I2:=0 to 479 do
         if random(100)>98
            then begin
                 R:=round(random(6)+1);
                 PB.Canvas.Pixels[I,I2]:=TcorA[R];
                 VBI[I,I2]:=R;
                 end
            else begin
                 PB.Canvas.Pixels[I,I2]:=TcorA[0];
                 VBI[I,I2]:=0;
                 end;
end;


procedure TForm1.StBtnClick(Sender: TObject);
begin
InitPb;
Timer1.Enabled:=true;
end;

procedure TForm1.ClBtnClick(Sender: TObject);
begin
Close;
end;

procedure TForm1.ReStBtnClick(Sender: TObject);
var I:integer;
begin
for I:=1 to 500 do
      begin
      CalculateStep;
      RePaintAdd;
      if (I mod 10) = 0 then
          AddRan;
      VIVLAB.Caption:=IntToStr(VIVAS);
      end;
end;

procedure TForm1.PBPaint(Sender: TObject);
begin
RePaintAll;
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
if not(InTask or CheckBox1.Checked) then
   begin
   CalculateStep;
   RePaintAdd;
   VIVLAB.Caption:=IntToStr(VIVAS);
   end; { of external if }
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
InTask:=false;
end;

end.
