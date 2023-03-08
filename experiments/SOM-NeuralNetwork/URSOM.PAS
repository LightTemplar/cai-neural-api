unit URSOM;

interface

const
  MaxNeuron = 10;

type
  TPos = object
    X, Y: integer;
  end;

type
  TPesos = array[1..2] of extended;

type
  TEntrada = TPesos;

type
  TNArray = object
    X: TEntrada;
    FatorDeCorrecao: extended;
    W: array[1..MaxNeuron, 1..MaxNeuron] of TPesos; // linha,coluna,peso
    Y: array[1..MaxNeuron, 1..MaxNeuron] of extended; // saidas

    function GetWinnerPos: TPos;
    procedure CalculaY;
    procedure MakeRandomWs;
    procedure DefineEntrada(X1, X2: extended);
    procedure Treina(X1, X2: extended);
  end;

//function CriaPesos(X,Y):
implementation

uses UVect;

procedure TNArray.MakeRandomWs;
var
  I, J, K: integer;
begin
  for I := 1 to MaxNeuron do
    for J := 1 to MaxNeuron do
      for k := 1 to 2 do
      begin
        W[I, J, K] := random(64000) / 64000;
      end;
end;

procedure TNArray.CalculaY;
var
  I, J: integer;
begin
  for I := 1 to MaxNeuron do
    for J := 1 to MaxNeuron do
      Y[I, J] := Distancia(X, W[I, J]);
end;

function TNArray.GetWinnerPos: TPos;
var
  I, J: integer;
  MenorY: extended;
begin
  MenorY := 100000000000;
  CalculaY;
  for I := 1 to MaxNeuron do
    for J := 1 to MaxNeuron do
    begin
      if Y[I, J] < MenorY then
      begin
        MenorY := Y[I, J];
        Result.X := I;
        Result.Y := J;
      end;
    end; // of for
end;// of procedure

procedure TNArray.DefineEntrada(X1, X2: extended);
//var modulo:extended;
begin
(*modulo:=sqrt(sqr(X1)+sqr(X2))+0.001;
X[1]:=X1/modulo;
X[2]:=X2/modulo;*)
  X[1] := X1;
  X[2] := X2;
end;

procedure Corrige(var W: TPesos; X: TEntrada; Fator: extended);
var
  Correcao: TPesos;
begin
  Correcao := X;
  SubtracaoVetorial(Correcao, W);
  ProdutoPorEscalar(Correcao, Fator);
  SomaVetorial(W, Correcao);
end;

procedure TNArray.Treina(X1, X2: extended);
var
  PV: TPos; // Posicao do Vencedor
begin
  DefineEntrada(X1, X2);
  PV := GetWinnerPos;
  Corrige(W[PV.X, PV.Y], X, FatorDeCorrecao);
  if PV.X > 1 then
    Corrige(W[PV.X - 1, PV.Y], X, FatorDeCorrecao * 0.7);
  if PV.Y > 1 then
    Corrige(W[PV.X, PV.Y - 1], X, FatorDeCorrecao * 0.7);
  if PV.X < MaxNeuron then
    Corrige(W[PV.X + 1, PV.Y], X, FatorDeCorrecao * 0.7);
  if PV.Y < MaxNeuron then
    Corrige(W[PV.X, PV.Y + 1], X, FatorDeCorrecao * 0.7);

  if (PV.X > 1) and (PV.Y > 1) then
    Corrige(W[PV.X - 1, PV.Y - 1], X, FatorDeCorrecao * 0.5);
  if (PV.X < MaxNeuron) and (PV.Y < MaxNeuron) then
    Corrige(W[PV.X + 1, PV.Y + 1], X, FatorDeCorrecao * 0.5);
  if (PV.X > 1) and (PV.Y < MaxNeuron) then
    Corrige(W[PV.X - 1, PV.Y + 1], X, FatorDeCorrecao * 0.5);
  if (PV.X < MaxNeuron) and (PV.Y > 1) then
    Corrige(W[PV.X + 1, PV.Y - 1], X, FatorDeCorrecao * 0.5);

  if PV.X > 2 then
    Corrige(W[PV.X - 2, PV.Y], X, FatorDeCorrecao * 0.3);
  if PV.Y > 2 then
    Corrige(W[PV.X, PV.Y - 2], X, FatorDeCorrecao * 0.3);
  if PV.X < MaxNeuron - 1 then
    Corrige(W[PV.X + 2, PV.Y], X, FatorDeCorrecao * 0.3);
  if PV.Y < MaxNeuron - 1 then
    Corrige(W[PV.X, PV.Y + 2], X, FatorDeCorrecao * 0.3);

end;



end.
