unit lifeai;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, Buttons, about;

type
  matr = array[0..399, 0..499] of boolean;

type
  matrptr = ^matr;

type
  TPaintBoxPtr = ^TPaintBox;

type
  TLife = object
    cel, cel2: matrptr;
    IM: TPaintBoxPtr;
    Sx, Sy, Dx, Dy: integer;  // Source X, Source Y, Destination X, Destination Y

    AllDead: boolean; // all dead flag
    BORDER: boolean;

    constructor Init(PIM: TPaintBoxPtr; PM1, PM2: matrptr;
      PSx, PSy, PDx, PDy: integer);
    procedure Run;  { um passo de execucao }
    function GetMorte: boolean;
    destructor Done;
    function CountNeighbors(x, y: integer): byte;
  end;

type
  TBlock = object
    IM: TPaintBoxPtr;
    BL: array[0..79, 0..99] of boolean;
    constructor Init(Video: TPaintBoxPtr);
    procedure Clear;
    procedure Reset;
    destructor Done;

  end;

type
  TLifes = array[0..79, 0..99] of TLife;

type
  TLifesPtr = ^TLifes;

type
  TSuperLife = object
    cel, cel2: matrptr;
    IM: TPaintBoxPtr;
    NextBL, Block: TBlock;
    Lifes: TLifesPtr;

    procedure Init(PIM: TPaintBoxPtr; PM1, PM2: matrptr);
    procedure Run;
    procedure Reset;

    procedure PrepRun; { Prepare Run;}
    destructor Done;
  end;


type
  TVideoBuffer = array[0..399, 0..499] of byte;

type
  { TForm1 }
  TForm1 = class(TForm)
    VIVLAB: TLabel;
    Timer1: TTimer;
    GroupBox1: TGroupBox;
    GroupBox2: TGroupBox;
    PB: TPaintBox;
    StBtn: TButton;
    ClBtn: TButton;
    CheckBox1: TCheckBox;
    About: TButton;
    procedure StBtnClick(Sender: TObject);
    procedure ClBtnClick(Sender: TObject);
    procedure ReStBtnClick(Sender: TObject);
    procedure PBPaint(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure AboutClick(Sender: TObject);
  private
    { Private declarations }
    CEL, CEL2: matrptr;
    L: TSuperLife;
    InTask: boolean;
    procedure InitPb;
    procedure RepaintAll;
    procedure CalculateStep; //calculate and paint the life and dead cels

  public
    { Public declarations }
  end;

const
  TcorA: array[0..12] of Tcolor =
    (clBlack, clMaroon, clOlive, clNavy, clTeal, clGray, clSilver,
    clRed, clGreen, clYellow, clPurple, clAqua, clolive);

var
  Form1: TForm1;

implementation

{$R *.lfm}
procedure TSuperLife.Init(PIM: TPaintBoxPtr; PM1, PM2: matrptr);
var
  xc, yc: integer; { XCounter,YCounter }
var
  XMIN, XMAX, YMIN, YMAX: integer;
begin
  new(Lifes);
  cel := PM1;
  cel2 := PM2;
  IM := PIM;
  Reset;
  Block.Init(IM);
  NextBl.Init(IM);

  for XC := 0 to 99 do
    for YC := 0 to 79 do
    begin
      XMIN := XC * 5;
      XMAX := XC * 5 + 4;
      YMIN := YC * 5;
      YMAX := YC * 5 + 4;
      if XMIN = 0 then
        Inc(XMIN);
      if YMIN = 0 then
        Inc(YMIN);
      if XMAX = 499 then
        Dec(XMAX);
      if YMAX = 399 then
        Dec(YMAX);
      Lifes^[YC, XC].Init(IM, cel, cel2, XMIN, YMIN, XMAX, YMAX);
    end;

end;

procedure TSuperLife.Reset;
var
  i, i2: integer;
begin
  for i := 0 to 499 do
    for i2 := 0 to 399 do
      cel^[i2, i] := IM^.Canvas.Pixels[i, i2] <> clBlack;
end;

procedure TSuperLife.Run;
var
  xc, yc: integer; { XCounter,YCounter }
begin
  NextBl.Clear;
  for XC := 0 to 99 do
    for YC := 0 to 79 do
      if Block.Bl[YC, XC] then
      begin
        Lifes^[YC, XC].Run;
        if not (Lifes^[YC, XC].GetMorte) then
        begin
          NextBl.Bl[YC, XC] := True;
          if Lifes^[YC, XC].Border then
          begin
            if XC > 0 then
              NextBl.Bl[YC, pred(XC)] := True;
            if XC < 99 then
              NextBl.Bl[YC, succ(XC)] := True;
            if YC > 0 then
              NextBl.Bl[pred(YC), XC] := True;
            if YC < 79 then
              NextBl.Bl[succ(YC), XC] := True;
          end;
        end;
      end;

end;

constructor TLife.Init(PIM: TPaintBoxPtr; PM1, PM2: matrptr; PSx, PSy, PDx, PDy: integer);
begin
  cel := PM1;
  cel2 := PM2;
  AllDead := True;
  IM := PIM;
  Sx := PSx;
  Sy := PSy;
  Dx := PDx;
  Dy := PDy;
end;

function TLife.GetMorte: boolean;
begin
  GetMorte := AllDead;
end;

procedure TSuperLife.PrepRun; { Prepare Run;}
begin
  cel2^ := cel^;
  Block := NextBl;
end;

destructor TSuperLife.Done;
begin
  dispose(Lifes);
end;

destructor TLife.Done;
begin
end;

procedure TLife.Run;
var
  xc, yc: integer; { XCounter,YCounter }
  NeighborsCnt: integer;{visinhos}
begin
  for yc := Sy to Dy do
    for xc := Sx to Dx do
    begin            (* BEGIN of FOR *)
      NeighborsCnt := CountNeighbors(xc, yc);
      cel^[yc, xc] := ((NeighborsCnt = 2) and (cel2^[yc, xc])) or (NeighborsCnt = 3);
      if (XC > 120) and (XC < 180) and (YC > 95) and (YC < 105) then
        cel^[yc, xc] := cel^[yc, xc] or (random(100) > 50);
    end;

  AllDead := True;
  BORDER := False;

  for yc := Sy to Dy do
  begin
    for xc := Sx to Dx do
    begin
      if cel^[yc, xc] then
      begin (* BEGIN of IF *)
        if (Xc = Sx) or (Xc = Dx) or (Yc = Sy) or (Yc = Dy) then
          BORDER := True;

        if not (cel2^[yc, xc]) then { testa se desenha }
        begin
          AllDead := False;
          IM^.Canvas.Pixels[xc, yc] := clAqua;
        end;
      end                                                    (* END of IF *)
      else
      begin
        if cel2^[yc, xc] then { testa se apaga }
        begin
          AllDead := False;
          IM^.Canvas.Pixels[xc, yc] := clBlack;
        end;
      end;
    end;
  end;
end;


function TLife.CountNeighbors(x, y: integer): byte;
var
  co: byte;
begin
  co := 0;
  if cel2^[pred(Y), pred(X)] then
    co := co + 1;
  if cel2^[pred(Y), x] then
    co := co + 1;
  if cel2^[pred(Y), succ(X)] then
    co := co + 1;

  if cel2^[Y, pred(X)] then
    co := co + 1;
  if cel2^[Y, succ(X)] then
    co := co + 1;

  if cel2^[succ(Y), pred(X)] then
    co := co + 1;
  if cel2^[succ(Y), X] then
    co := co + 1;
  if cel2^[succ(Y), succ(X)] then
    co := co + 1;
  CountNeighbors := co;
end;

constructor TBlock.Init(Video: TPaintBoxPtr);
begin
  IM := Video;
  Reset;
end;

procedure TBlock.Reset;
var
  x, y, xi, yi: integer;
begin
  Clear;
  for X := 0 to 99 do
    for Y := 0 to 79 do
      for XI := 0 to 4 do
        for YI := 0 to 4 do
          BL[Y, X] := BL[Y, X] or (IM^.Canvas.Pixels[Y * 5 + XI, X * 5 + YI] <> clBlack);
end;


procedure TBlock.Clear;
var
  x, y: integer;
begin
  for X := 0 to 99 do
    for Y := 0 to 79 do
      BL[Y, X] := False;
end;

destructor TBlock.Done;
begin
end;

procedure TForm1.CalculateStep;
{ calculate the life and dead cels }
begin
  L.PrepRun;
  L.Run;
end; { of procedure }

procedure TForm1.RepaintAll;
{ display all cels }
var
  X, Y: word;
  Count: word;
  COR: TColor;
begin
  for Y := 0 to 399 do
  begin
    X := 0;
    while X <= 499 do { desenha linha }
    begin
      Count := 0;
      while (CEL^[Y, X] = CEL^[Y, X + Count + 1]) and (X + Count + 1 <= 499) do
      begin
        Inc(Count);
      end;

      if CEL^[Y, X] then
        COR := clAqua
      else
        COR := clBlack;

      if Count > 0 then
      begin
        PB.Canvas.MoveTo(X, Y);
        Inc(X, Count);
        PB.Canvas.Pen.Color := COR;
        PB.Canvas.LineTo(X + 1, Y);
      end
      else
      begin
        PB.Canvas.Pixels[X, Y] := COR;
      end;

      Inc(X);
    end;     {end of desenha lina }
  end;{ of for}
end;

procedure TForm1.InitPb;
{ put the starting cels }
var
  I, I2: word;
begin
  randomize;
  for I := 0 to 499 do
    for I2 := 0 to 399 do
      CEL^[I2, I] := False;
end;


procedure TForm1.StBtnClick(Sender: TObject);
begin
  with Sender as TButton do
    Visible := False;
  InitPb;
  RePaintAll;
  Timer1.Enabled := True;
end;

procedure TForm1.ClBtnClick(Sender: TObject);
begin
  Close;
end;

procedure TForm1.ReStBtnClick(Sender: TObject);
begin
  CheckBox1.Checked := False;
end;

procedure TForm1.PBPaint(Sender: TObject);
begin
  if Assigned(CEL) then
    RePaintAll;
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
  if not (InTask or CheckBox1.Checked) then
  begin
    CalculateStep;
  end;
end;

procedure TForm1.FormCreate(Sender: TObject);
var
  XC: integer;
begin
  InTask := False;
  new(CEL);
  new(CEL2);
  InitPb;
  randomize;
  for XC := 140 to 160 do
    PB.Canvas.Pixels[XC, 100] := clAqua;

  L.Init(addr(PB), CEL, CEL2);
  L.NextBl.Bl[32, 20] := True;
  InitPb;
  RePaintAll;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  dispose(CEL);
  dispose(CEL2);
  L.Done;
end;

procedure TForm1.AboutClick(Sender: TObject);
begin
  AboutBox.Show;
  with Sender as TButton do
    Visible := False;
end;

end.
