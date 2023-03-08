unit udum;
// this unit has been deprecated. Use it at your own risk.

{$MODE Delphi}

{...}

interface

uses Classes, RegExpr, LCLIntf, LCLType, Forms, FileUtil{,shdocvw};

function Sign(X: extended): extended; { tipo TNeuronFunction }
{ devolve sinal do operando }
function TanH(X: extended): extended; { tipo TNeuronFunction }
{ tangente hiperbolica }
function DTanH(X: extended): extended;  { tipo TNeuronFunction }
{ derivada da tangente hiperbolica }
function POT(numero, elevado: extended): extended;
{*  Eleva o numero a sua potencia.
     exemplo : 3^4= pot(3,4) *}
function LOG(num, base: extended): extended;
function EXPON(base, resultado: extended): extended;
{* EXPON = numero que torna verdadeira
         a equacao: base ^ expon = resultado    *}
function IMAX(X, Y: integer): integer;
function IMIN(X, Y: integer): integer;
function EMAX(X, Y: extended): extended;
function EMIN(X, Y: extended): extended;

procedure EMessage(S: string);
{ Exibe mensagem de erro }
procedure WMessage(S: string);
{ Exibe mensagem de aviso }
function Confirma(S: string): boolean;
{Se confirma entao eh true}

function GetFileSize(FileName: string): longint;
{devolve tamanho do arquivo}
function PegaDataHora(FNAME: string): integer;
{devolve data e hora de criacao}
function SetaDataHora(FNAME: string; DataHora: integer): integer;
{seta data e hora de criacao}
procedure GetFileComponents(FullName: string; var Drive: char;
  var Path, Name: string);

function AddIfNotFound(SRC, ARG: string): string;
{ se ARG nao ocorre em SCR, entao adiciona ARG a SCR}

function AddIfNotFoundStr(SRC, ARG, S: string): string;
{ se ARG nao ocorre em SCR, entao adiciona S a SCR}

function AddBarraInvertida(S: string): string;
{ adiciona barra invertida caso necessario }

function TryDeleteFile(FName: string): boolean;
// tenta deletar. Se consegue, retorna TRUE;

function DiskFreeAtFile(FName: string): longint;
// return free disk space at the position of File

procedure CopyFile(S, D: ShortString);
// Source Destination

function FileReadLock(FN: string): boolean;
{ testa se tem lock para leitura }

type
  TStringListPair = class
  private
    FA, FB: TStringList;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Clear;
    procedure Add(S1, S2: string);
    function Count: integer;
    procedure MultipleReplace(var S: WideString);
  published
    property A: TStringList read FA;
    property B: TStringList read FB;
  end;



type
  IODiskBuffer = record
    Buf: array[0..2047] of byte;{buffer propriamente dito}
    NUM: integer; {numero de bytes no buffer }
    Modified: boolean;{ se o buffer foi modificado}
    TAB: longint; { numero da tabela (pagina) }
  end;


type
  IODISK = object
  public
    F: file of byte;
    NOME: string;          { nome do arquivo   }
    BUFIN: IODiskBuffer;   { buffer de leitura }
    BUFOUT: IODiskBuffer;  { buffer de escrita }
    POS: longint;

    constructor Init(PNOME: string);
    { abre o arquivo }
    { open a file }

    destructor Done;
    { fecha arquivo }
    { close file }

    function GetCurrentPosition: longint;
    {pega posicao atual}

    function ReadB: byte;
    { read a byte }

    procedure WriteB(B: byte);
    { write a byte }

    procedure SeekByte(P: longint);
    { seek using bytes }

    procedure SeekRel(P: longint);
    {seek relativo}

    procedure ReadSizedReg(Pont: pointer; S: longint);
    { read a SIZED register }

    procedure WriteSizedReg(Pont: pointer; S: longint);
    { write a SIZED register }

    procedure ReadSizedRegOnPos(PPOS: longint; Pont: pointer; S: longint);
    { read a SIZED register }

    procedure WriteSizedRegOnPos(PPOS: longint; Pont: pointer; S: longint);
    { write a SIZED register }

    function ReadString: string;
    {read a string }

    procedure WriteString(S: string);
    { write a string}

    function ReadNChars(N: integer): string;
    { read N cars and return a string }

    function ReadNCharsOnPos(PPOS: longint; N: integer): string;
    { read N cars and return a string }

    function FindStr(S: string): longint;
               { return the String position or -1
               if not found. The search is forward
               from the current position }

    function Read(PPOS: longint): byte;
    { le um byte da posicao POS }
    { read a byte on position POS }

    procedure Write(PPOS: longint; INF: byte);
    { escreve um byte INF na posicao POS }
    { write a byte on position POS }

    function TAMANHO: longint;
    { devolve o tamanho do arquivo }
    { returns the byte's number of a file }

    procedure HARD;
    { gravacao fisica do buffer de saida }
    { record buffers}

    procedure RELOAD;
    { le os buffers novamente do disco }
    { reload buffers }

  end;
//------------------------------------------------------------------------------
//                            WEB & REGEX
//------------------------------------------------------------------------------
function CreateURLListFromHTML(var HTML: TStringList; UrlType: string): TStringList;
// cria uma lista com as URLs encontradas no texto do tipo UrlType

function CreateWWWListFromHTML(var HTML: string): TStringList;
// cria uma lista com as WWWs encontradas no texto do tipo UrlType

function NowOnSeconds: TDateTime;
// retorna hora atual em segundos

implementation

uses
  Messages, SysUtils, Graphics, Controls, Dialogs,
  Grids, Buttons, Printers,
  ExtCtrls;

function JAND(A, B: extended): extended;
begin
  JAND := A * B;
end;

function JOR(A, B: extended): extended;
begin
  JOR := A + B - JAND(A, B);
end;

function JNOT(A: extended): extended;
begin
  JNOT := 1 - A;
end;

function LETRA(car: char): boolean;
begin
  letra := (upCASE(car) >= 'A') and (upCASE(car) <= 'Z');
end;

function BRANCO(CAR: char): boolean;
begin
  branco := (car = ' ') or (car = chr(9)) or (car = chr(13));
end;

function DELIM(car: char): boolean;
begin
  if pos(car, '+-/*%^=()$&|') <> 0 then
    delim := True
  else
    delim := False;
end;

function DIGITO(CAR: char): boolean;
begin
  digito := ((car >= '0') and (car <= '9'));
end;


//------------------------------------------------------------------------------
//                            WEB & REGEX
//------------------------------------------------------------------------------

function CreateURLListFromHTML(var HTML: TStringList; UrlType: string): TStringList;
  // cria uma lista com as URLs encontradas no texto do tipo UrlType
var
  Found: boolean;
  RE: TRegExpr;
  FoundURL: string;
begin
  RE := TRegExpr.Create;
  RE.ModifierI := True;
  Result := TStringList.Create;
  try
    RE.Expression := 'href="?(' + UrlType + '[^ ">]+)';
    RE.Compile;
    Found := RE.Exec(HTML.Text);
    while Found do
    begin
      FoundURL := RE.Match[1];
      Result.Add(FoundURL);
      Found := RE.ExecNext;
    end;
  except
  end;
end;

function CreateWWWListFromHTML(var HTML: string): TStringList;
  // cria uma lista com as WWWs encontradas no texto do tipo UrlType
var
  Found: boolean;
  RE: TRegExpr;
  FoundURL: string;
begin
  RE := TRegExpr.Create;
  RE.ModifierI := True;
  Result := TStringList.Create;
  try
    RE.Expression := 'www\.[^ "<>\'']+';
    RE.Compile;
    Found := RE.Exec(HTML);
    while Found do
    begin
      FoundURL := RE.Match[0];
      if Result.IndexOf(FoundURL) = -1 then
        Result.Add(FoundURL);
      Found := RE.ExecNext;
    end;
  except
  end;
end;

function NowOnSeconds: TDateTime;
  // retorna hora atual em segundos
begin
  NowOnSeconds := Now * 86400;
end;

type
  T64Kbytes = array[1..655350000] of byte;

type
  T64KbytesPtr = ^T64Kbytes;
//----------------------------MATH-----------------------------------
function Sign(X: extended): extended; { tipo TNeuronFunction }
  { devolve sinal do operando }
begin
  if X = 0 then
    Sign := 1
  else
    Sign := X / abs(X);
end;

{ funcoes de ativacao }

function TanH(X: extended): extended; { tipo TNeuronFunction }
  { tangente hiperbolica }
begin
  TanH := (1 - exp(-2 * X)) / (1 + exp(-2 * X));
end;

function DTanH(X: extended): extended;  { tipo TNeuronFunction }
  { derivada da tangente hiperbolica }
begin
  DTanH := 1 - sqr(TanH(X));
end;

function POT(numero, elevado: extended): extended;
{*  Eleva o numero a sua potencia.
     exemplo : 3^4= pot(3,4) *}
begin
  pot := exp(elevado * ln(numero));
end;

function LOG(num, base: extended): extended;
begin
  log := ln(num) / ln(base);
end;

function EXPON(base, resultado: extended): extended;
{* EXPON = numero que torna verdadeira
         a equacao: base ^ expon = resultado    *}
begin
  expon := log(resultado, base);
end;

function IMAX(X, Y: integer): integer;
begin
  if X > Y then
    IMAX := X
  else
    IMAX := Y;
end;

function IMIN(X, Y: integer): integer;
begin
  if X < Y then
    IMIN := X
  else
    IMIN := Y;
end;

function EMAX(X, Y: extended): extended;
begin
  if X > Y then
    EMAX := X
  else
    EMAX := Y;
end;

function EMIN(X, Y: extended): extended;
begin
  if X < Y then
    EMIN := X
  else
    EMIN := Y;
end;
//-----------------------Simple Dialogs------------------------

procedure WMessage(S: string);
{ Exibe mensagem de aviso }
begin
  MessageDlg(S, mtWarning, [mbOK], 0);
end;

procedure EMessage(S: string);
{ Exibe mensagem de erro }
begin
  MessageDlg(S, mtError, [mbOK], 0);
end;

function Confirma(S: string): boolean;
  {Se confirma entao eh true}
begin
  Confirma := (MessageDlg(S + '?', mtConfirmation, [mbYes, mbNo], 0) = mrYes);
end;

//------------------------FILES-------------------------------------

function FileReadLock(FN: string): boolean;
var
  F: file of byte;
  OldFileMode: longint;
begin
  OldFileMode := FileMode;
  FileMode := 0;
  AssignFile(F, FN);
    {$I-}
  Reset(F);
{$I+}
  if (IOResult <> 0) then
  begin
    FileReadLock := True;
  end
  else
  begin
    FileReadLock := False;
    CloseFile(F);
  end;
  FileMode := OldFileMode;
end; // of function FileLock


function GetFileSize(FileName: string): longint;
  {devolve tamanho do arquivo}
var
  R, OldFileMode: longint;
  F: file of byte;
begin
  OldFileMode := FileMode;
  if FileExistsUTF8(FileName) { *Converted from FileExists*  } then
  begin
    FileMode := 0;
    AssignFile(F, FileName);
   {$I-}
    Reset(F);
    if IOResult <> 0 then
    begin
      R := -1;
    end
    else
    begin
      //R:=FileSize(F){ div 1024};
      CloseFile(F);
    end;
   {$I+}
  end
  else
  begin
    R := -1;
  end;
  GetFileSize := R;
  FileMode := OldFileMode;
end;

procedure GetFileComponents(FullName: string; var Drive: char;
  var Path, Name: string);
begin
  Path := ExtractFileDir(FullName);
  Drive := FullName[1];
  Delete(Path, 1, 2);
  Name := ExtractFileName(FullName);
end;

function AddIfNotFound(SRC, ARG: string): string;
  { se ARG nao ocorre em SCR, entao adiciona ARG a SCR}
var
  R: string;
begin
  if POS(AnsiUpperCase(ARG), AnsiUpperCase(SRC)) = 0 then
    R := SRC + ARG
  else
    R := SRC;
  AddIfNotFound := R;
end;


function PegaDataHora(FNAME: string): integer;
var
  FileHandle: integer;
begin
  FileHandle := FileOpen(FName, 0);
  if FileHandle > 0 then
  begin
    PegaDataHora := FileGetDate(FileHandle);
    FileClose(FileHandle);
  end
  else
    PegaDataHora := 0;

end; // of PegaDataHora

function TryDeleteFile(FName: string): boolean;
  // tenta deletar. Se consegue, retorna TRUE;
const
  csCommonFile: integer = 32;
begin
  if FileExistsUTF8(FName) { *Converted from FileExists*  } then
  begin
    try
      //FileSetAttrUTF8(FName); { *Converted from FileSetAttr*  }
      //DeleteFileUTF8(FNAME); { *Converted from DeleteFile*  }
    except
    end; // of try
  end;
  TryDeleteFile := not (FileExistsUTF8(FName) { *Converted from FileExists*  });
end;

function SetaDataHora(FNAME: string; DataHora: integer): integer;
var
  FileHandle: integer;
begin
  FileHandle := FileOpen(FName, 2);
  if FileHandle > 0 then
  begin
    //SetaDataHora:=FileSetDateUTF8(FileHandle,DataHora); { *Converted from FileSetDate*  }
    FileClose(FileHandle);
  end
  else
    SetaDataHora := 1;
end; // of PegaDataHora

function DiskFreeAtFile(FName: string): longint;
  // return free disk space at the position of File
var
  D: char;
  P, N: string;
  DriveNumber: byte;
begin
  if FName = 'NUL' then
  begin
    DiskFreeAtFile := MaxLongint;
    exit;
  end;

  GetFileComponents(FName, D, P, N);
  DriveNumber := Ord(UpCase(D)) - Ord('A') + 1;
  DiskFreeAtFile := DiskFree(DriveNumber);
end;


(*function AddIfNotFound(SRC,ARG:string):string;
{ se ARG nao ocorre em SCR, entao adiciona ARG a SCR}
var R:string;
begin
if POS(AnsiUpperCase(ARG),AnsiUpperCase(SRC))=0
   then R:=SRC+ARG
   else R:=SRC;
AddIfNotFound:=R;
end; *)

function AddIfNotFoundStr(SRC, ARG, S: string): string;
  { se ARG nao ocorre em SCR, entao adiciona S a SCR}
var
  R: string;
begin
  if POS(AnsiUpperCase(ARG), AnsiUpperCase(SRC)) = 0 then
    R := SRC + S
  else
    R := SRC;
  AddIfNotFoundStr := R;
end;

function AddBarraInvertida(S: string): string;
  { adiciona barra invertida caso necessario }
var
  R: string;
begin
  R := S;
  if R[Length(R)] <> '\' then
    R := R + '\';
  AddBarraInvertida := R;
end;

procedure CopyFile(S, D: ShortString);
// Source Destination
var
  I, O: IODisk;
  Cont, TAM: longint;
begin
  I.Init(S);
  O.Init(D);
  TAM := I.Tamanho;
  for Cont := 1 to TAM do
    O.WriteB(I.ReadB);
  I.Done;
  O.Done;
end;

// -------------------------IODISK----------------------------------
constructor IODISK.Init(PNOME: string);
  { abre o arquivo }
begin
  NOME := PNOME;
  POS := 0;
  AssignFile(F, NOME);
  if FileExistsUTF8(NOME) { *Converted from FileExists*  } then
    reset(F{,1})
  else
    rewrite(F{,1});
  seek(F, 0);

  { prepara buffer de entrada }
  BUFIN.TAB := 0;
  BUFIN.MODIFIED := False;
  blockread(F, BUFIN.BUF, 2048, BUFIN.NUM);

  {prepara arquivo de saida }
  seek(F, 0);
  BUFOUT.TAB := 0;
  BUFOUT.MODIFIED := False;
  blockread(F, BUFOUT.BUF, 2048, BUFOUT.NUM);
end;

function IODISK.Read(PPOS: longint): byte;
  { le um byte da posicao POS }
var
  TAB: longint;    { tabela para alocar }
  POSONTAB: word;  { posicao na tabela  }
  R: byte;
begin
  TAB := PPOS shr 11;
  POSONTAB := PPOS and 2047;
  if BUFOUT.TAB = TAB then        { procura na tabela de escrita }
  begin
    R := BUFOUT.BUF[POSONTAB];
  end
  else
  begin
    if BUFIN.TAB = TAB then      { procaura na tabela de leitura }
    begin
      R := BUFIN.BUF[POSONTAB];
    end
    else
    begin                        { le outra tabela }
      seek(F, TAB shl 11);
      BUFIN.TAB := TAB;
      BUFIN.MODIFIED := False;
      blockread(F, BUFIN.BUF, 2048, BUFIN.NUM);
      R := BUFIN.BUF[POSONTAB];
    end;
  end;
  Read := R;
end; { of function }


procedure IODISK.Write(PPOS: longint; INF: byte);
{ escreve um byte INF na posicao POS }
var
  TAB: longint;    { tabela para alocar }
  POSONTAB: word;  { posicao na tabela  }
begin
  TAB := PPOS shr 11;
  POSONTAB := PPOS and 2047;
  if BUFIN.TAB = TAB then
  begin
    BUFIN.BUF[POSONTAB] := INF;    { grava informacao no buffer de leitura }
  end;

  if BUFOUT.TAB = TAB then
  begin
    BUFOUT.BUF[POSONTAB] := INF;
    BUFOUT.MODIFIED := True;
  end
  else
  begin
    if BUFOUT.MODIFIED then
    begin
      seek(F, BUFOUT.TAB shl 11);           { grava buffer antigo }
      blockwrite(F, BUFOUT.BUF, BUFOUT.NUM);
      BUFOUT.MODIFIED := False;
    end;

    seek(F, TAB shl 11);
    BUFOUT.TAB := TAB;
    blockread(F, BUFOUT.BUF, 2048, BUFOUT.NUM);
    BUFOUT.BUF[POSONTAB] := INF;
    BUFOUT.MODIFIED := True;
  end;

  if BUFOUT.NUM < succ(POSONTAB) then
    BUFOUT.NUM := succ(POSONTAB); {GARANTE GRAVACAO DENTRO DO BUFFER}
end; { of procedure }

function IODISK.TAMANHO: longint;
  { devolve o tamanho do arquivo }
begin
  //TAMANHO:=filesize(F);
end;

procedure IODISK.HARD;
{ gravacao fisica do buffer de saida }
begin
  if BUFOUT.MODIFIED then
  begin
    seek(F, BUFOUT.TAB shl 11);           { grava buffer antigo }
    blockwrite(F, BUFOUT.BUF, BUFOUT.NUM);
    BUFOUT.MODIFIED := False;
  end;
end;

procedure IODISK.RELOAD;
{ le os buffers novamente do disco }
begin
  { prepara buffer de entrada }
  BUFIN.MODIFIED := False;
  seek(F, BUFIN.TAB shl 11);
  blockread(F, BUFIN.BUF, 2048, BUFIN.NUM);

  {prepara arquivo de saida }
  seek(F, BUFOUT.TAB shl 11);
  BUFOUT.MODIFIED := False;
  blockread(F, BUFOUT.BUF, 2048, BUFOUT.NUM);
end;

destructor IODISK.Done;
  { fecha arquivo }
begin
  HARD;
  CloseFile(F);
end;

function IODISK.GetCurrentPosition: longint;
  {pega posicao atual}
begin
  GetCurrentPosition := POS;
end;

function IODISK.ReadB: byte;
  { read a byte }
begin
  ReadB := Read(POS);
  Inc(POS);
end;

procedure IODISK.WriteB(B: byte);
{ write a byte }
begin
  Write(POS, B);
  Inc(POS);
end;

procedure IODISK.SeekByte(P: longint);
{ seek using bytes }
begin
  POS := P;
end;

procedure IODISK.SeekRel(P: longint);
{seek relativo}
begin
  SeekByte(POS + P);
end;

procedure IODISK.ReadSizedReg(Pont: pointer; S: longint);
{ read a SIZED register }
var
  Buf: T64KbytesPtr;
  Cont: longint;
begin
  Buf := Pont;
  for Cont := 1 to S do
    Buf^[Cont] := ReadB;
end;

procedure IODISK.WriteSizedReg(Pont: pointer; S: longint);
{ write a SIZED register }
var
  Buf: T64KbytesPtr;
  Cont: longint;
begin
  Buf := Pont;
  for Cont := 1 to S do
    WriteB(Buf^[Cont]);
end;

procedure IODISK.ReadSizedRegOnPos(PPOS: longint; Pont: pointer; S: longint);
{ read a SIZED register on a defined position}
var
  Buf: T64KbytesPtr;
  Cont: longint;
begin
  Buf := Pont;
  for Cont := 1 to S do
  begin
    Buf^[Cont] := Read(PPOS);
    Inc(PPOS);
  end;
end;

procedure IODISK.WriteSizedRegOnPos(PPOS: longint; Pont: pointer; S: longint);
{ write a SIZED register on a defined position}
var
  Buf: T64KbytesPtr;
  Cont: longint;
begin
  Buf := Pont;
  for Cont := 1 to S do
  begin
    Write(PPOS, Buf^[Cont]);
    Inc(PPOS);
  end;
end;

                {$BOOLEVAL OFF }
function IODISK.FindStr(S: string): longint;
               { return the String position or -1
               if not found. The search is forward
               from the current position }
var
  Found: boolean;
  Len: longint; //lenght of S
  Last: longint;
  B1: byte;
begin
  Len := length(S);
  if Len > 0 then
  begin  // main code
    Found := False;
    B1 := Ord(S[1]);
    Last := TAMANHO - Len;
    while (POS < Last) and not (Found) do
    begin
      while (POS < Last) and (B1 <> ReadB) do
        {nothing};
      if B1 = Read(POS - 1) // internal if
      then
      begin
        Found := (S = ReadNCharsOnPos(POS - 1, Len));
      end;  // internal if
    end;// of while
    if Found then
      FindStr := POS - 1
    else
      FindStr := -1;
  end // main code
  else
    FindStr := POS;
end; // of procedure

function IODISK.ReadString: string;
  {read a string }
var
  S: string;
  B: byte;
  //              Cont:longint;
begin
  B := ReadB;
  S := '';
  while B <> 0 do
  begin
    S := S + chr(B);
    B := ReadB;
  end;
  ReadString := S;
end;{ of procedure }

procedure IODISK.WriteString(S: string);
{ write a string}
var
  Cont: longint;
begin
  for Cont := 1 to Length(S) do
    WriteB(Ord(S[Cont]));
  WriteB(0);
end;

function IODISK.ReadNChars(N: integer): string;
  { read N cars and return a string }
var
  Cont: longint;
  S: string;
begin
  S := '';
  for Cont := 1 to N do
    S := S + chr(ReadB);
  ReadNChars := S;
end;

function IODISK.ReadNCharsOnPos(PPOS: longint; N: integer): string;
  { read N cars and return a string }
var
  OLDPOS: longint;
begin
  OLDPOS := POS;
  POS := PPOS;
  ReadNCharsOnPos := ReadNChars(N);
  POS := OLDPOS;
end;

{ TStringListPair }

procedure TStringListPair.Add(S1, S2: string);
begin
  FA.Add(S1);
  FB.Add(S2);
end;

procedure TStringListPair.Clear;
begin
  FA.Clear;
  FB.Clear;
end;

function TStringListPair.Count: integer;
begin
  Count := FA.Count;
end;

constructor TStringListPair.Create;
begin
  FA := TStringList.Create;
  FB := TStringList.Create;
end;

destructor TStringListPair.Destroy;
begin
  inherited;
  FA.Free;
  FB.Free;
end;


procedure TStringListPair.MultipleReplace(var S: WideString);
var
  i: integer;
begin
  if FA.Count > 0 then
  begin
    for I := 0 to FA.Count - 1 do
    begin
      S := StringReplace(S, FA[I], FB[I], [rfReplaceAll, rfIgnoreCase]);
    end;
  end;
end;

end.
