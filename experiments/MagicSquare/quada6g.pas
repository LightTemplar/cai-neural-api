unit quada6g;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, LMessages, Messages, SysUtils, Classes, Graphics,
  Controls, Forms, Dialogs,
  StdCtrls, Buttons, ExtCtrls;

{-------------------magic suares---------------------------------------------}
const
  GRAU = 22;

const
  CGRAU2 = ((GRAU) * (GRAU - 1)) div 2;

const
  GRAU2 = GRAU * GRAU;

const
  NTROCA = (GRAU2 * (GRAU2 - 1)) div 2;
{const NTROCA2 =(NTROCA*NTROCA)-NTROCA;}
const
  TOTLIN = (((1 + GRAU2) * GRAU2) div 2) div GRAU;

{Valores para grau = 22}
{CGRAU2 = 231 ; GRAU2 = 484
 NTROCA = 117128 }

{Valores para grau = 26}
{CGRAU2 = 325 ; GRAU2 = 676
 NTROCA = 228150 }


const
  QueroArq = False;

type
  TAQUAD = array[1..GRAU2] of word;

type
  TTROCA = record
    A: word;
    B: word;
  end;

type
  TATROCA = array[1..NTROCA] of TTROCA;

type
  TALTROCA = array[1..CGRAU2] of TTROCA;

type
  TQUAD = object

    FOUND: boolean;
    QUAD: TAQUAD;
    TROCA: TATROCA;
    LTROCA: TALTROCA;
    CICLOS: longint;
    OLDCICLOS: longint;
    NFOUND: longint;
    WantFile, WantQuit: boolean;
    Distance: longint; {distance of answare}

    constructor Init(PGRAU: byte);

    destructor Done;

    procedure Mostra;
    procedure Gera(var Q: TAQUAD);
    procedure PTroca(var Q: TAQUAD; N: longint);
    procedure Run;
    procedure Try03;
    procedure TryAcheiA(var Q: TAQUAD);
    {procedure TryAchei(var Q:TAQUAD);}

  end;
{---------------------------form1------------------------------------------}


type
  TForm1 = class(TForm)
    GroupBox1: TGroupBox;
    BitBtn1: TBitBtn;
    GroupBox2: TGroupBox;
    OutList: TListBox;
    StopBtn: TBitBtn;
    SaveBtn: TBitBtn;
    SaveDialog1: TSaveDialog;
    GroupBox3: TGroupBox;
    GrauCBox: TComboBox;
    AboutBtn: TBitBtn;
    StateLab: TLabel;
    LabCicl: TLabel;
    LabNCicl: TLabel;
    LabDist: TLabel;
    LabNDist: TLabel;
    Timer1: TTimer;
    LabFound: TLabel;
    LabNFound: TLabel;
    LabSpeed: TLabel;
    LabNSpeed: TLabel;
    procedure BitBtn1Click(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure SaveBtnClick(Sender: TObject);
    procedure StopBtnClick(Sender: TObject);
    procedure AboutBtnClick(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
  private
    { Private declarations }
    Q6: TQUAD;

  public
    { Public declarations }
  end;

var
  Form1: TForm1;
  VGRAU, VCGRAU2, VGRAU2, VNTROCA, VTOTLIN: longint;

implementation

uses UABOUT;

procedure PrintLn(S: string); forward;
procedure Print(S: string); forward;
procedure PrintNum(Value: extended; N, D: integer); forward;
procedure PrintNumLn(Value: extended; N, D: integer); forward;

function EvalQuad(Q: TAQUAD): longint;
var
  Count, COUNT2: word;
  POS, POS2: word;
  SOMLIN, SOMCOL: array[1..GRAU] of longint;
  COLPRIN, COLSEC: longint;
  DIFTOTAL: longint;
begin
  DIFTOTAL := 0;

  for Count := 1 to VGRAU do
  begin
    SOMLIN[Count] := 0;
    SOMCOL[Count] := 0;
  end;

  COLPRIN := 0;
  COLSEC := 0;

  for Count := 1 to VGRAU do  { soma linhas e colunas }
    for COUNT2 := 1 to VGRAU do
    begin
      SOMLIN[COUNT2] := SOMLIN[COUNT2] + Q[VGRAU * pred(Count) + COUNT2];
      SOMCOL[COUNT2] := SOMCOL[COUNT2] + Q[VGRAU * pred(COUNT2) + Count];
    end;

  POS := 1;
  POS2 := VGRAU;

  for Count := 1 to VGRAU do   {soma diagonais}
  begin
    COLPRIN := COLPRIN + Q[POS];
    COLSEC := COLSEC + Q[POS2];
    POS := POS + succ(VGRAU);
    POS2 := POS2 + pred(VGRAU);
  end;

  DIFTOTAL := abs(COLPRIN - VTOTLIN) + abs(COLSEC - VTOTLIN);

  for Count := 1 to VGRAU do
    DIFTOTAL := DIFTOTAL + abs(VTOTLIN - SOMLIN[Count]) +
      abs(VTOTLIN - SOMCOL[Count]);


  EvalQuad := DIFTOTAL;
end; { of procedure }

function EvalQuadWithMax(Q: TAQUAD; Max: longint): longint;
var
  Count, COUNT2: word;
  POS, POS2: word;
  SOMLIN, SOMCOL: longint;
  COLPRIN, COLSEC: longint;
  DIFTOTAL: longint;
begin
  DIFTOTAL := 0;
  COLPRIN := 0;
  COLSEC := 0;

  Count := 1;
  while (Count <= VGRAU) and (DIFTOTAL <= Max) do
  begin
    SOMLIN := 0;
    SOMCOL := 0;
    COUNT2 := 1;
    while (COUNT2 <= VGRAU) do
    begin
      SOMLIN := SOMLIN + Q[VGRAU * pred(Count) + COUNT2];
      SOMCOL := SOMCOL + Q[VGRAU * pred(COUNT2) + Count];
      Inc(COUNT2);
    end;{ of internal while}
    DIFTOTAL := DIFTOTAL + abs(VTOTLIN - SOMLIN) + abs(VTOTLIN - SOMCOL);
    Inc(Count);
  end;{of external while}

  if (DIFTOTAL <= MAX) then
  begin
    POS := 1;
    POS2 := VGRAU;
    for Count := 1 to VGRAU do   {soma diagonais}
    begin
      COLPRIN := COLPRIN + Q[POS];
      COLSEC := COLSEC + Q[POS2];
      POS := POS + succ(VGRAU);
      POS2 := POS2 + pred(VGRAU);
    end; { of for }
    DIFTOTAL := DIFTOTAL + abs(COLPRIN - VTOTLIN) + abs(COLSEC - VTOTLIN);
  end; {of if}

  EvalQuadWithMax := DIFTOTAL;
end; { of procedure }

constructor TQUAD.Init(PGRAU: byte);
var
  Count, COUNT2: longint;
  POS: longint;
begin
  VGRAU := PGRAU;
  VCGRAU2 := ((VGRAU) * (VGRAU - 1)) div 2;
  VGRAU2 := VGRAU * VGRAU;
  VNTROCA := (VGRAU2 * (VGRAU2 - 1)) div 2;
  VTOTLIN := (((1 + VGRAU2) * VGRAU2) div 2) div VGRAU;

  WantQuit := False;

  Print('Dimensions:');
  PrintNumLn(VGRAU, 0, 0);
  Print('Cells:');
  PrintNumLn(VGRAU2, 0, 0);
  Print('Level 1:');
  PrintNumLn(VNTROCA, 0, 0);
  Print('Total sum per row:');
  PrintNumLn(VTOTLIN, 0, 0);
  CICLOS := 0;
  OLDCICLOS := 0;
  NFOUND := 0;
  Distance := 0;
  Randomize;
  POS := 0;
  for Count := 1 to VGRAU2 do  {gera tabela de trocas}
    for COUNT2 := succ(Count) to VGRAU2 do
    begin
      Inc(POS);
      TROCA[POS].A := Count;
      TROCA[POS].B := COUNT2;
    end;
  Gera(QUAD);
  PrintLn('--------------------------------------');

  POS := 0;
  for Count := 1 to VGRAU do
    for COUNT2 := succ(Count) to VGRAU do
    begin
      Inc(POS);
      LTROCA[POS].A := Count;
      LTROCA[POS].B := COUNT2;
    end;

end;

destructor TQUAD.Done;
begin
end;

procedure TQUAD.Gera(var Q: TAQUAD);
var
  Count: longint;
begin
  ;
  for Count := 1 to VGRAU2 do
    Q[Count] := Count;
  for Count := 1 to VNTROCA do
    PTROCA(Q, succ(round(random(VNTROCA))));
end;

function GetPos(X, Y: word): word;
begin
  GetPos := X + pred(Y) * VGRAU;
end;

procedure FlipDP(var OUT: TAQUAD; INP: TAQUAD);
var
  Count, COUNT2: longint;
begin
  for Count := 1 to VGRAU do
    for COUNT2 := 1 to VGRAU do
      OUT[GetPos(Count, COUNT2)] := INP[GetPos(COUNT2, Count)];
end; {of procedure }

procedure FlipDS(var OUT: TAQUAD; INP: TAQUAD);
var
  Count: longint;
begin
  for Count := 1 to VGRAU2 do
    OUT[Count] := INP[succ(VGRAU2 - Count)];
end;

procedure MostraQUAD(var QUAD: TAQUAD);
var
  Count, COUNT2: longint;
  POS: longint;
begin
  ;
  POS := 0;
  for Count := 1 to VGRAU do
  begin
    for COUNT2 := 1 to VGRAU do
    begin
      Inc(POS);
      PrintNum(QUAD[POS], 4, 0);
    end;
    PrintLn('');
  end;
  Application.ProcessMessages;
end;

procedure TQUAD.Mostra;
begin
  ;
  MostraQUAD(QUAD);
end;

procedure TQUAD.PTroca(var Q: TAQUAD; N: longint);
var
  AUX: word;
begin
  ;
  AUX := Q[TROCA[N].A];
  Q[TROCA[N].A] := Q[TROCA[N].B];
  Q[TROCA[N].B] := AUX;
end;

procedure TROCAD(var Q: TAQUAD; P1, P2: word);
var
  AUX: word;
begin
  AUX := Q[P1];
  Q[P1] := Q[P2];
  Q[P2] := AUX;
end;

procedure TQUAD.TryAcheiA(var Q: TAQUAD);
begin
  if EvalQuadWithMax(Q, 0) = 0 then
  begin
    Inc(NFOUND);
    PrintLn('');
    Print('Magic square has been found:');
    PrintNumLn(NFOUND, 0, 0);
    MOSTRAQUAD(Q);
  end;{of if}
end; {of procedure }


procedure TQUAD.Try03;
var
  USE: TAQUAD;
  OLDEVAL, EVAL: longint;
  Count, COUNT2: longint;
  BESTPOS1, BESTEVAL: longint;
  ATUAL1: longint;
  MAX: longint;
  LA: boolean; {local achei}
begin
  BESTPOS1 := 1;
  BESTEVAL := VNTROCA{EvalQuad(QUAD)};
  OLDEVAL := EvalQuad(QUAD);{BESTEVAL;}
  EVAL := OLDEVAL;
  USE := QUAD;
  Count := 0;
  LA := False;
  MAX := round(VNTROCA / 1.5);
  while (Count < MAX) and not (LA) do
  begin
    Inc(CICLOS);
    Inc(Count);
    USE := QUAD;
    ATUAL1 := succ(round(random(VNTROCA)));
    PTROCA(USE, ATUAL1);        {troca}
    EVAL := EvalQuadWithMax(USE, BESTEVAL);
    if EVAL <= BESTEVAL then
    begin
      BESTEVAL := EVAL;
      BESTPOS1 := ATUAL1;
    end; { of if}

    if EVAL <= OLDEVAL then
      LA := True;

    if (VGRAU > 19) and (random(1000) > 998) then
      Application.ProcessMessages;
  end;{ of FOR }

  PTROCA(QUAD, BESTPOS1);
  Application.ProcessMessages;
  Distance := BESTEVAL;
  (*if Random(100)>6 then
        begin
        {PrintNumln(BESTEVAL);}
        end;  *)

  if BESTEVAL = 0 then
  begin
    Inc(NFOUND);
    PrintLn('');
    Print('magic square found(achei quadrado magico):');
    PrintNumLn(NFOUND, 0, 0);
    MOSTRA;
    for Count := 1 to VCGRAU2 do        { profundidade 2 }
      for COUNT2 := succ(Count) to VCGRAU2 do
      begin
        USE := QUAD;
        TROCAD(USE, GetPos(LTROCA[COUNT2].A, LTROCA[Count].A),
          GetPos(LTROCA[COUNT2].B, LTROCA[Count].A));
        TROCAD(USE, GetPos(LTROCA[COUNT2].A, LTROCA[Count].B),
          GetPos(LTROCA[COUNT2].B, LTROCA[Count].B));
        TryAcheiA(USE);

        USE := QUAD;
        TROCAD(USE, GetPos(LTROCA[COUNT2].A, LTROCA[Count].A),
          GetPos(LTROCA[COUNT2].A, LTROCA[Count].B));
        TROCAD(USE, GetPos(LTROCA[COUNT2].B, LTROCA[Count].A),
          GetPos(LTROCA[COUNT2].B, LTROCA[Count].B));
        TryAcheiA(USE);

        USE := QUAD;
        TROCAD(USE, GetPos(LTROCA[COUNT2].A, LTROCA[Count].A),
          GetPos(LTROCA[COUNT2].B, LTROCA[Count].B));
        TROCAD(USE, GetPos(LTROCA[COUNT2].B, LTROCA[Count].A),
          GetPos(LTROCA[COUNT2].A, LTROCA[Count].B));
        TryAcheiA(USE);

      end;{of for}

    Gera(Quad);
  end;
end;


procedure TQUAD.Run;
begin
  while not (WantQuit) do
  begin
    Try03;
  end;
end;



{$R *.lfm}

procedure TForm1.BitBtn1Click(Sender: TObject);
begin
  try
    BitBtn1.Enabled := False;
    StopBtn.Enabled := True;
    StateLab.Caption := 'Running';
    Q6.Init(StrToInt(GrauCBox.Text));
    Timer1.Enabled := True;
    Q6.RUN;
    Q6.Done;
  finally
    StateLab.Caption := 'IDLE';
    Timer1.Enabled := False;
    StopBtn.Enabled := False;
    BitBtn1.Enabled := True;
  end; { of try}
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  Q6.WantQuit := True;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  GrauCBox.ItemIndex := 0;
end;

procedure TForm1.SaveBtnClick(Sender: TObject);
begin
  if SaveDialog1.Execute then
    OutList.Items.SaveToFile(SaveDialog1.FileName);
end;

procedure TForm1.StopBtnClick(Sender: TObject);
begin
  Q6.WantQuit := True;
end;

procedure TForm1.AboutBtnClick(Sender: TObject);
begin
  AboutBox.Show;
end;

procedure TForm1.Timer1Timer(Sender: TObject);
begin
  LabNCicl.Caption := IntToStr(Q6.Ciclos div 1000) + ' k';
  LabNDist.Caption := IntToStr(Q6.Distance);
  LabNFound.Caption := IntToStr(Q6.NFound);
  LabNSpeed.Caption := IntToStr((Q6.CICLOS - Q6.OLDCICLOS) div 4000) + ' k/s';
  Q6.OLDCICLOS := Q6.CICLOS;
end;


var
  Saida: string;

procedure PrintLn(S: string);
begin
  if Form1.OutList.Items.Count > 30000 then
    Form1.OutList.Items.Delete(0);
  Form1.OutList.Items.Add(Saida + S);
  Application.ProcessMessages;
  Saida := '';
end;

procedure Print(S: string);
begin
  Saida := Saida + S;
end;

procedure PrintNum(Value: extended; N, D: integer);
var
  R: string;
begin
  Str(Value: N: D, R);
  Print(R);
end;

procedure PrintNumLn(Value: extended; N, D: integer);
begin
  PrintNum(Value, N, D);
  PrintLn('');
end;

begin { of unit}
  Saida := '';
end.
