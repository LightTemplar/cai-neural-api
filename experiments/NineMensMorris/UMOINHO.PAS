unit UMoinho;

{Felipe}
{Guilherme}
{Joao Paulo Schwarz Schuler}
interface

type
  TPos = byte; { # Posicao no tabuleiro }

type
  TPosRange = 1..24; { # Posicao no tabuleiro }

type
  TTipoPedra = 0..5;

type
  T012 = 0..2;

type
  T12 = 1..2;

const
  peVazio = 0;  {pedras}
  peBranco = 1;
  pePreto = 2;
  peBrancoSel = 3;
  pePretoSel = 4;
  peVermelho = 5;

const
  csJogador: array[TTipoPedra] of byte = (0, 1, 2, 1, 2, 0);
{ responde a que jogador pertence um tipo de pedra }

const
  Melhores: array[TPosRange] of TPos = (10, 12, 14, 16,
    18, 20, 22, 24,
    2, 4, 6, 8,
    1, 3, 5, 7,
    9, 11, 13, 15,
    17, 19, 21, 23);

const
  Movimentos: array[TPosRange] of { # movimentos possiveis }
    array[1..4] of TPos = (
    (8, 2, 0, 0),      {1}
    (10, 1, 3, 0),     {2}
    (4, 2, 0, 0),      {3}
    (3, 5, 12, 0),     {4}
    (4, 6, 0, 0),      {5}
    (14, 7, 5, 0),     {6}
    (8, 6, 0, 0),      {7}
    (1, 7, 16, 0),     {8}
    (16, 10, 0, 0),    {9}
    (2, 18, 9, 11),    {10}
    (12, 10, 0, 0),    {11}
    (11, 13, 20, 4),   {12}
    (12, 14, 0, 0),    {13}
    (22, 6, 15, 13),   {14}
    (16, 14, 0, 0),    {15}
    (9, 15, 8, 24),    {16}
    (24, 18, 0, 0),    {17}
    (10, 17, 19, 0),   {18}
    (20, 18, 0, 0),    {19}
    (19, 21, 12, 0),   {20}
    (20, 22, 0, 0),    {21}
    (14, 23, 21, 0),   {22}
    (24, 22, 0, 0),    {23}
    (17, 23, 16, 0));  {24}

const
  MTrilha: array[TPosRange] of { # posicoes da mesma trilha :
                             1 e 2 para horizontal ; 3 e 4 para vertical}
    array[1..4] of TPos = (
    (2, 3, 7, 8),      {1}
    (1, 3, 10, 18),    {2}
    (1, 2, 4, 5),      {3}
    (12, 20, 3, 5),    {4}
    (6, 7, 3, 4),      {5}
    (5, 7, 14, 22),    {6}
    (5, 6, 1, 8),      {7}
    (16, 24, 1, 7),    {8}
    (10, 11, 15, 16),  {9}
    (9, 11, 2, 18),    {10}
    (9, 10, 12, 13),   {11}
    (4, 20, 11, 13),   {12}
    (14, 15, 11, 12),  {13}
    (13, 15, 6, 22),   {14}
    (13, 14, 9, 16),   {15}
    (8, 24, 9, 15),    {16}
    (18, 19, 23, 24),  {17}
    (17, 19, 2, 10),   {18}
    (17, 18, 20, 21),  {19}
    (4, 12, 19, 21),   {20}
    (22, 23, 19, 20),  {21}
    (21, 23, 14, 6),   {22}
    (21, 22, 17, 24),  {23}
    (8, 16, 17, 23));  {24}


type
  TOcupCel = record {# ocupacao das trilhas verticais e horizontais}
    V: T012; { # numero de pedras na trilha vertical}
    H: T012; { # numero de pedras na trilha horizontal}
  end;

type
  TOcup = array[TPosRange{Posicoes}, 1..2{tipo de pedra do jogador}] of TOcupCel;
                  {# ocupacao por trilha para cada celula para cada
                     tipo de pedra}
type
  TTabuleiroArray = array[TPosRange] of TTipoPedra;

type
  TTabuleiro = object  { # tabuleiro }
  public
    Tab: TTabuleiroArray;
    Ocupacao: TOcup;
    pecas: array[1..2] of byte; {Numero de pecas do jogador 1 e 2}

    constructor Init;
    procedure Limpa;
    {limpa tabuleiro}
    procedure PoemPedra(Posicao: TPosRange; Tipo: TTipoPedra);
    {insere pedra no tabuleiro}
    procedure DeletaPedra(Posicao: TPosRange);
    {deleta pedra do tabuleiro}
    procedure DeslocaPedra(PosInicial, PosFinal: TPosRange);
    {desloca pedra de posicao}
    function MovimentoPossivel(PosIni, PosFim: TPosRange): boolean;
    {testa se movimento eh possivel}
    procedure DesselecionaTrilha(Pos: TPosRange);
    { desseleciona pedras nas trilhas que passam por Pos}
    function AnalisaTrilha(Pos: TPosRange): byte;
                  { Verifica se forma trilha e seta valores;
                  devolve numero de trilhas formadas }


    destructor Done;
  end;

type
  TRetJoga = object
    Achei: boolean;
    MelhorIni, MelhorFim: TPosRange;
    Nota: longint;
    procedure Seta(PAchei: boolean; PMelhorIni, PMelhorFim: TPosRange;
      PNota: longint);
  end;

type
  TTesta = function(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;


type
  TRetEscolhePos = object
    Pos: TPosRange;
    Nota: longint;

    procedure Seta(PPos: TPosRange; PNota: longint);
  end;

type
  TTestaPos = function(MO: TTabuleiro; Jog: T12; Pos: TPosRange): TRetEscolhePos;


implementation

constructor TTabuleiro.Init;
begin
  Limpa;
end;

procedure TTabuleiro.Limpa;
{limpa tabuleiro}
  procedure LimpaTab;
  var
    I: integer;
  begin
    for I := 1 to 24 do
      Tab[I] := peVazio;
  end; {of procedure LimpaTab}

  procedure LimpaOcupacao;
  var
    I: integer;
    Cel: TOcupCel;
  begin
    Cel.H := 0;
    Cel.V := 0;
    for I := 1 to 24 do
    begin
      Ocupacao[I, 1] := Cel;
      Ocupacao[I, 2] := Cel;
    end;
  end;  { of procedure LimpaOcup; }

begin { of procedure Limpa}
  Pecas[1] := 0;
  Pecas[2] := 0;
  LimpaTab;
  LimpaOcupacao;
end;{ of procedure Limpa}

procedure TTabuleiro.PoemPedra(Posicao: TPosRange; Tipo: TTipoPedra);
{insere pedra no tabuleiro}
  procedure Incrementa(Jogador: byte);
  var
    I: integer;
    Ocups: TPos; { posicoes para incremento de ocupacoes }
  begin
    Inc(Pecas[Jogador]);
    for I := 1 to 2 do
    begin
      Ocups := MTrilha[Posicao, I];
      Inc(Ocupacao[Ocups, Jogador].H);
    end; { of for}

    for I := 3 to 4 do
    begin
      Ocups := MTrilha[Posicao, I];
      Inc(Ocupacao[Ocups, Jogador].V);
    end; { of for}
  end; { of procedure incrementa }

var
  Jogador: byte;

begin { of procedure PoemPedra }
  Jogador := csJogador[Tipo];
  Tab[Posicao] := Tipo;
  if Jogador <> 0 then
  begin
    Incrementa(Jogador);
    AnalisaTrilha(Posicao);
  end;

end;

procedure TTabuleiro.DeletaPedra(Posicao: TPosRange);
{deleta pedra do tabuleiro}
  procedure Decrementa(Jogador: byte);
  var
    I: integer;
    Ocups: TPos; { posicoes para decremento de ocupacoes }
  begin
    Dec(Pecas[Jogador]);
    for I := 1 to 2 do
    begin
      Ocups := MTrilha[Posicao, I];
      Dec(Ocupacao[Ocups, Jogador].H);
    end; { of for}

    for I := 3 to 4 do
    begin
      Ocups := MTrilha[Posicao, I];
      Dec(Ocupacao[Ocups, Jogador].V);
    end; { of for}
  end; { of procedure decrementa }

var
  Tipo: TTipoPedra;
  Jogador: byte;
begin { of procedure DeletaPedra }
  Tipo := Tab[Posicao];
  Tab[Posicao] := peVazio;
  Jogador := csJogador[Tipo];
  if Jogador <> 0 then
  begin
    Decrementa(Jogador);
    DesselecionaTrilha(Posicao);
  end;
end;

procedure TTabuleiro.DeslocaPedra(PosInicial, PosFinal: TPosRange);
{desloca pedra de posicao}
var
  Tipo: TTipoPedra;
begin
  Tipo := Tab[PosInicial];
  DeletaPedra(PosInicial);
  PoemPedra(PosFinal, csJogador[Tipo]);
end;

function TTabuleiro.MovimentoPossivel(PosIni, PosFim: TPosRange): boolean;
  {testa se movimento eh possivel}

begin
  MovimentoPossivel := (Tab[PosFim] = peVazio) and
    ((Movimentos[PosIni][1] = PosFim) or
    (Movimentos[PosIni][2] = PosFim) or
    (Movimentos[PosIni][3] = PosFim) or
    (Movimentos[PosIni][4] = PosFim));
end;



procedure TTabuleiro.DesselecionaTrilha(Pos: TPosRange);
{ desseleciona pedras nas trilhas que passam por Pos}
var
  I: integer;
  PosP1: TPosRange;
begin

  for I := 1 to 2 do
  begin
    PosP1 := MTrilha[Pos][I];
    if Ocupacao[PosP1, csJogador[Tab[PosP1]]].V = 2 then
    {Tab[PosP1]:=csJogador[Tab[PosP1]]+2}
    else
      Tab[PosP1] := csJogador[Tab[PosP1]];
  end; { of for }

  for I := 3 to 4 do
  begin
    PosP1 := MTrilha[Pos][I];
    if Ocupacao[PosP1, csJogador[Tab[PosP1]]].H = 2 then
    {Tab[PosP1]:=csJogador[Tab[PosP1]]+2}
    else
      Tab[PosP1] := csJogador[Tab[PosP1]];
  end; { of for }

end;

function TTabuleiro.AnalisaTrilha(Pos: TPosRange): byte;
{ Verifica se forma ou nao trilha e seta valores;
  devolve numero de trilhas formadas }
var
  R: byte; {resultado}
  PosP1, PosP2: TPosRange; {posicao da pedra 1 e 2 }

  procedure Seleciona;
  begin
    Inc(R);
    Tab[Pos] := csJogador[Tab[Pos]] + 2;
    Tab[PosP1] := csJogador[Tab[PosP1]] + 2;
    Tab[PosP2] := csJogador[Tab[PosP2]] + 2;
  end;

begin { of procedure AnalisaTrilha }
  R := 0;
  PosP1 := MTrilha[Pos][1];        { cotrilhantes }
  PosP2 := MTrilha[Pos][2];
  if (csJogador[Tab[PosP1]] = csJogador[Tab[PosP2]]) and    { se iguais }
    (csJogador[Tab[PosP1]] = csJogador[Tab[Pos]]) and (Tab[Pos] <> peVazio) then
    Seleciona;                                      { seleciona }

  PosP1 := MTrilha[Pos][3];
  PosP2 := MTrilha[Pos][4];
  if (csJogador[Tab[PosP1]] = csJogador[Tab[PosP2]]) and
    (csJogador[Tab[PosP1]] = csJogador[Tab[Pos]]) and (Tab[Pos] <> peVazio) then
    Seleciona;

  AnalisaTrilha := R;
end;

destructor TTabuleiro.Done;
begin
end;

procedure TRetJoga.Seta(PAchei: boolean; PMelhorIni, PMelhorFim: TPosRange; PNota: longint);
begin
  Achei := PAchei;
  MelhorIni := PMelhorIni;
  MelhorFim := PMelhorFim;
  Nota := PNota;
end;

procedure TRetEscolhePos.Seta(PPos: TPosRange; PNota: longint);
begin
  Pos := PPos;
  Nota := PNota;
end;


end.
