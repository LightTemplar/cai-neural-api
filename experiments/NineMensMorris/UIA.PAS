unit UIA;

interface

uses UMoinho;

{ rotinas de avaliacao (Teste(s)}
function Teste1(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
function Teste2(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
function Teste3(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
function Teste4(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
function EscolhePos(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; TestaPos: TTestaPos): TRetEscolhePos;

function EscolheCome(MO: TTabuleiro; Jog: T12): integer;
{escolhe pedra para comer}

{ rotinas de jogo JOGA_*}
function Joga(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;
function Joga_A(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;
function Joga_B(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;

function Joga3(MO: TTabuleiro; Jog: T12): TRetJoga;
{ joga com 3 pecas }

procedure SetMaxProf(MP: integer);
{seta profundidade maxima}

implementation

function MaxI(X, Y: longint): longint;
begin
  if X > Y then
    MaxI := X
  else
    MaxI := Y;
end;

function Teste1(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
var
  R, I: longint;
begin
  I := MaxI(10 * (MO.Ocupacao[PosIni, Jog].V -
    MO.Ocupacao[PosIni, 3 - Jog].V), 10 *
    (MO.Ocupacao[PosIni, Jog].H - MO.Ocupacao[PosIni, 3 - Jog].H));

  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;
  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V -
      MO.Ocupacao[PosFim, 3 - Jog].V), 100 *
      (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
    R := R + Round(Random(3000));
  end;
  if Jog = 1 then
    R := -R;
  if R >= I then
    Teste1.Seta(True, PosIni, PosFim, R)
  else
    Teste1.Seta(True, PosIni, PosFim, 0);
end;

function Teste2(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
var
  R: longint;
  RE: TRetJoga;
begin
  R := 0;
  if Jog <> 1 then
  begin
    MO.DeslocaPedra(PosIni, PosFim);
    if MO.AnalisaTrilha(PosFim) > 0 then
      R := 5000
    else
      R := 1 * Round(Random(3000));

    if Jog = 1 then
      R := -R;
    RE.Seta(True, PosIni, PosFim, R);
    Teste2 := RE;
  end;
end;

function EscolhePos(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; TestaPos: TTestaPos): TRetEscolhePos;
  { escolhe posicao para por uma pedra}
var
  I, J: TPosRange;
begin
  for I := 1 to 24 do        { forma trilha ? }
  begin
    if ((MO.Ocupacao[I, Jog].V = 2) or (MO.Ocupacao[I, Jog].H = 2)) and
      (MO.Tab[I] = peVazio) then
    begin
      EscolhePos.Pos := I;
      exit;
    end; { of if }
  end; { of for }

  for I := 1 to 24 do        { evita trilha ? }
  begin
    if ((MO.Ocupacao[I, 3 - Jog].V = 2) or (MO.Ocupacao[I, 3 - Jog].H = 2)) and
      (MO.Tab[I] = peVazio) then
    begin
      EscolhePos.Pos := I;
      exit;
    end; { of if }
  end; { of for }

  for I := 1 to 24 do        { forma 2 ? }
  begin
    if (MO.Ocupacao[I, Jog].V + (MO.Ocupacao[I, Jog].H) = 2) and
      ((MO.Ocupacao[I, 3 - Jog].V + MO.Ocupacao[I, 3 - Jog].H) = 0) and
      (MO.Tab[I] = peVazio) then
    begin
      EscolhePos.Pos := I;
      exit;
    end; { of if }
  end; { of for }

  for I := 1 to 24 do        { evita 2 ? }
  begin
    if (MO.Ocupacao[I, 3 - Jog].V + (MO.Ocupacao[I, 3 - Jog].H) = 2) and
      (MO.Ocupacao[I, Jog].V + (MO.Ocupacao[I, Jog].H) = 0) and
      (MO.Tab[I] = peVazio) then
    begin
      EscolhePos.Pos := I;
      exit;
    end; { of if }
  end; { of for }

  for I := 1 to 24 do        { escolhe livre }
  begin
    if (MO.Ocupacao[Melhores[I], Jog].V + (MO.Ocupacao[Melhores[I], Jog].H) = 0) and
      (MO.Ocupacao[Melhores[I], 3 - Jog].V + (MO.Ocupacao[Melhores[I], 3 - Jog].H) = 0) and
      (MO.Tab[Melhores[I]] = peVazio) then
    begin
      EscolhePos.Pos := Melhores[I];
      exit;
    end; { of if }
  end; { of for }

  J := 1;
  while MO.Tab[Melhores[J]] <> peVazio do
    Inc(J);
  EscolhePos.Pos := Melhores[J];
end;

function EscolheCome(MO: TTabuleiro; Jog: T12): integer;
  {escolhe pedra para comer do Jog}
var
  I, J, ADV: integer;
  PADJ: TPosRange;
begin
  ADV := 3 - Jog;
  if MO.Pecas[Jog] = 4 then { jogador tem 4 pecas?}
  begin
    for J := 1 to 24 do     {external for}
    begin
      if (MO.Tab[J] = Jog) and (MO.Ocupacao[J, Jog].V + MO.Ocupacao[J, Jog].H = 2) then
      begin
        EscolheCome := J;
        exit;
      end; {of if}
    end; { of for }

    for J := 1 to 24 do     {external for}
    begin
      if (MO.Tab[J] = Jog) and ((MO.Ocupacao[J, Jog].V - MO.Ocupacao[J, ADV].V = 1) or
        (MO.Ocupacao[J, Jog].H - MO.Ocupacao[J, ADV].H = 1)) then
      begin
        EscolheCome := J;
        exit;
      end; {of if}
    end; { of for }
  end; { of if }

  for J := 1 to 24 do     {external for}
  begin
    if (MO.Tab[Melhores[J]] = Jog) then
    begin     {external if}
      for I := 1 to 4 do
      begin {internal for}
        PADJ := MTRILHA[Melhores[J]][I];
        if (MO.Tab[PADJ] = peVazio) and   {internal if}
          (((MO.Ocupacao[PADJ, Jog].H = 2) and (I > 2)) or { vai formar moinho?}
          ((MO.Ocupacao[PADJ, Jog].V = 2)) and (I < 3)) then
        begin
          EscolheCome := Melhores[J];
          exit;
        end; {of internal if}
      end; { of internal for }
    end; { of external if }
  end; { of external for }


  J := 1;
  while (MO.Tab[Melhores[J]] <> Jog) and (J < 24) do   // se nao escolheu ainda
    Inc(J);                                    // pega primeira nao em moinho
  if (J = 24) and (MO.Tab[Melhores[J]] <> Jog) then
    EscolheCome := 0
  else
  begin
    EscolheCome := Melhores[J];
    exit;
  end;

  J := 1;
  while (CsJogador[MO.Tab[Melhores[J]]] <> Jog) and (J < 24)  // se nao escolheu ainda
    do
    Inc(J);                                    // pega primeira
  if (J = 24) and (CsJogador[MO.Tab[Melhores[J]]] <> Jog) then
    EscolheCome := 0
  else
  begin
    EscolheCome := Melhores[J];
    exit;
  end;

end; { of procedure EscolheCome}

function Joga(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;
var
  Melhor: TRetJoga;
  Velho: TTabuleiro;
var
  I, J: integer;
  NTeste, Teste: TRetJoga; { NTeste : teste no nivel }

begin
  Dec(Profundidade);
  Velho := MO;
  Melhor.Seta(False, 1, 1, 0);
  for I := 1 to 24 do
  begin
    if csJogador[MO.Tab[I]] = Jog then   { procura pedras }
    begin
      J := 1;
      while (J <= 4) and (Movimentos[I][J] <> 0) do  { procura movimentos }
      begin
        if MO.Tab[Movimentos[I][J]] = peVazio  { pode mover }
        then
        begin
          Teste := Testa(MO, Jog, I, Movimentos[I][J]);
          NTeste := Teste;
          if Teste.Nota = 5000 then
            Teste.Nota := Teste.Nota + (Profundidade + 1)
          else if (Profundidade > 0) and (abs(Teste.Nota) <> 5000) then
          begin
            MO.DeslocaPedra(I, Movimentos[I][J]);
            Teste := Joga(MO, 3 - Jog, Profundidade, Testa);
            MO := Velho;
            if not (Teste.Achei) then
            begin
              Teste.Seta(True, I, Movimentos[I][J], 1);
            end;
          end; { of internal if }

                            (* if (Teste.Nota>=Melhor.Nota) and (Teste.Achei) then
                                begin
                                Melhor.Seta(true,NTeste.MelhorIni,NTeste.MelhorFim,Teste.Nota);
                                end; { of internal if }  *)

          if Teste.Achei then
          begin
            if ((Teste.Nota >= Melhor.Nota) and not (Melhor.Achei)) or
              (Teste.Nota > Melhor.Nota) or
              (Melhor.Achei and (Teste.Nota = Melhor.Nota) and
              (random(1000) > 100)) then
              Melhor.Seta(True, NTeste.MelhorIni, NTeste.MelhorFim, Teste.Nota);
          end;
        end; { of external if pode mover}
        Inc(J);
      end; { of while }
    end; { of if procura pedras }
  end; { of for }
  Joga := Melhor;
end; { of procedure Joga }

function TestaFormaMoinho(var MO: TTabuleiro; I: integer{posicao};
  Jog: T12;
  PodePoderia: boolean{pode=false;poderia=true}
  ): boolean;
var
  J: integer;
  Forma: boolean;
begin
  Forma := False;
  if Mo.Pecas[Jog] < 4     { external if } then
  begin
    if (((MO.Tab[I] = peVazio) or PodePoderia) and
      ((MO.Ocupacao[I, Jog].H = 2) or (MO.Ocupacao[I, Jog].V = 2)))
    then
    begin
      Forma := True;
    end;
  end
  else
  begin { external if }
    if (((MO.Tab[I] = peVazio) or PodePoderia) and
      ((MO.Ocupacao[I, Jog].H = 2) or (MO.Ocupacao[I, Jog].V = 2)))
    then
    begin
      for J := 1 to 4 do
      begin
        if (Movimentos[I, J] <> 0) and
          (CsJogador[MO.Tab[Movimentos[I, J]]] = Jog) and
          (((MO.Ocupacao[I, Jog].V = 2) and
          (MTrilha[I, 3] <> Movimentos[I, J]) and
          (MTrilha[I, 4] <> Movimentos[I, J])) or
          ((MO.Ocupacao[I, Jog].H = 2) and
          (MTrilha[I, 1] <> Movimentos[I, J]) and
          (MTrilha[I, 2] <> Movimentos[I, J])))
        then
        begin
          Forma := True;
        end;
      end; { of J for }
    end;
  end; { external if }
  TestaFormaMoinho := Forma;
end; { of function TestaFormaMoinho }

function Joga3(MO: TTabuleiro; Jog: T12): TRetJoga;
  { joga com 3 pecas }
var
  ADV: integer;

  procedure TestaOndeFormaMoinho(var Forma: boolean; var Onde: integer; Jog: T12);
  var
    I: integer;
  begin
    Forma := False;
    for I := 1 to 24 do                      { forma moinho? Onde ?}
      if TestaFormaMoinho(MO, I, Jog, False) then
      begin
        Forma := True;
        Onde := I;
        exit;
      end;
  end; { of procedure TestaMoinho }

var
  PodeCome, FormaMoinho, ADVFormaMoinho: boolean;
  I, OndeFormaMoinho, ADVOndeFormaMoinho, PecaMove: integer;

begin  { of function Joga3}
  ADV := 3 - Jog;    {adversario}
  PodeCome := False;  {existe peca para comer?}
  for I := 1 to 24 do
    PodeCome := PodeCome or (MO.Tab[I] = ADV);

  TestaOndeFormaMoinho(FormaMoinho, OndeFormaMoinho, Jog);
  TestaOndeFormaMoinho(ADVFormaMoinho, ADVOndeFormaMoinho, ADV);

  PecaMove := 0;        { escolhe peca para mover }
  for I := 1 to 24 do   { escolhe peca com 2 vizinhos amigos na mesma trilha}
    if csJogador[MO.Tab[I]] = Jog then
    begin
      if (MO.Ocupacao[I, Jog].H = 2) or (MO.Ocupacao[I, Jog].V = 2) then
        PecaMove := I;
    end;


  if (PecaMove = 0) and FormaMoinho then{ se nao encontrou, escolhe peca com 1 vizinho }
    for I := 1 to 24 do
      if (csJogador[MO.Tab[I]] = Jog) and
        (MO.Ocupacao[I, Jog].H + MO.Ocupacao[I, Jog].V < 2) and
        (MTrilha[I, 1] <> OndeFormaMoinho) and
        (MTrilha[I, 2] <> OndeFormaMoinho) and
        (MTrilha[I, 3] <> OndeFormaMoinho) and
        (MTrilha[I, 4] <> OndeFormaMoinho) and
        (({(MO.Pecas[ADV]<>4) and} FormaMoinho) or
        not (TestaFormaMoinho(MO, I, ADV, True)))
      {adversario nao poderia formar moinho } then
      begin
        PecaMove := I;
      end;


  if PecaMove = 0 then        { se nao encontrou, escolhe peca com 1 vizinho }
    for I := 1 to 24 do
      if (csJogador[MO.Tab[I]] = Jog) and
        (MO.Ocupacao[I, Jog].H + MO.Ocupacao[I, Jog].V < 2) and
        (({(MO.Pecas[ADV]<>4) and} FormaMoinho) or
        not (TestaFormaMoinho(MO, I, ADV, True)))
      {adversario nao poderia formar moinho } then
      begin
        PecaMove := I;
      end;

  if PecaMove = 0 then    { se nao encontrou, escolhe primeira peca que encontrar }
    for I := 1 to 24 do
      if (csJogador[MO.Tab[I]] = Jog) and
        (({(MO.Pecas[ADV]<>4) and} FormaMoinho) or
        not (TestaFormaMoinho(MO, I, ADV, True)))
      {adversario nao poderia formar moinho } then
      begin
        PecaMove := I;
      end;

  if PecaMove = 0 then    { se nao encontrou, escolhe primeira peca que encontrar }
    for I := 1 to 24 do
      if csJogador[MO.Tab[I]] = Jog then
      begin
        PecaMove := I;
      end;

  if PodeCome and FormaMoinho and not (ADVFormaMoinho and (MO.Pecas[ADV] = 4)) then
  begin
    Joga3.Seta(True, PecaMove, OndeFormaMoinho, 5000);
    exit;
  end;

  if {not(PodeCome) and }ADVFormaMoinho then
  begin
    Joga3.Seta(True, PecaMove, ADVOndeFormaMoinho, 5000);
    exit;
  end;


  {procura por posicao com 1 vizinho em cada trilha e nenhum adversario}
  {...}

  {procura por posicao com 1 vizinho e nehum inimigo na trilha na horizontal}
  for I := 1 to 24 do
    if (MO.Tab[I] = 0) and ((MO.Ocupacao[I, Jog].H = 1) and
      (MO.Ocupacao[I, 3 - Jog].H = 0)) then
    begin
      if (MTrilha[I, 1] <> PecaMove) and (MTrilha[I, 2] <> PecaMove)
      then
      begin
        Joga3.Seta(True, PecaMove, I, 5000);
        exit;
      end;
    end;

  {procura por posicao com 1 vizinho e nehum inimigo na trilha na vertical}
  for I := 1 to 24 do
    if (MO.Tab[I] = 0) and ((MO.Ocupacao[I, Jog].V = 1) and
      (MO.Ocupacao[I, 3 - Jog].V = 0)) then
    begin
      if (MTrilha[I, 3] <> PecaMove) and (MTrilha[I, 4] <> PecaMove)
      then
      begin
        Joga3.Seta(True, PecaMove, I, 5000);
        exit;
      end;
    end;

  {procura por qualquer posicao vazia}
  for I := 1 to 24 do
    if (MO.Tab[I] = 0) then
    begin
      Joga3.Seta(True, PecaMove, I, 5000);
      exit;
    end;

end;

function Teste3(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
  { usada em conjunto com Joga_A }
var
  R: longint;
begin
{I:=MaxI(10*(MO.Ocupacao[PosIni,Jog].V-MO.Ocupacao[PosIni,3-Jog].V),
        10*(MO.Ocupacao[PosIni,Jog].H-MO.Ocupacao[PosIni,3-Jog].H));}

  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;
  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V - MO.Ocupacao[PosFim, 3 - Jog].V),
      100 * (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
    R := R + Round(Random(3000));
  end;
  Teste3.Seta(True, PosIni, PosFim, R);
end;

function Teste4(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
  { usada em conjunto com Joga_A }
var
  R: longint;
begin
  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;
  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V), 100 *
      (MO.Ocupacao[PosFim, Jog].H)){+};
    {R:=R+Round(Random(30));}
  end;
  Teste4.Seta(True, PosIni, PosFim, R);
end;

var
  MaxProf: integer;

{seta profundidade maxima}

procedure SetMaxProf(MP: integer);
begin
  MaxProf := MP - 1;
end;

function Teste31(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
  { usada em conjunto com Joga_A }
var
  R, I, J: longint;
begin
  I := 0;

  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;

  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    if (MO.Ocupacao[PosFim, 3 - Jog].V = 2) or (MO.Ocupacao[PosFim, 3 - Jog].H = 2)
    then
    begin
      I := PosFim;
      for J := 1 to 4 do
      begin
        if (Movimentos[I, J] <> 0) and
          (   { movimento existe }
          (CsJogador[MO.Tab[Movimentos[I, J]]] = 3 - Jog) and  { adjacente ocupado por adversario }
          ((MO.Ocupacao[I, 3 - Jog].V = 2) and      { 2x1}
          (MTrilha[I, 3] <> Movimentos[I, J]) and
          (MTrilha[I, 4] <> Movimentos[I, J])) or
          ((MO.Ocupacao[I, 3 - Jog].H = 2) and
          (MTrilha[I, 1] <> Movimentos[I, J]) and
          (MTrilha[I, 2] <> Movimentos[I, J])))
        then
          R := 4998;
      end; { of for }
      if R <> 4998 then
        R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V - MO.Ocupacao[PosFim, 3 - Jog].V),
          100 *
          (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
    end
    else
    begin
      R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V - MO.Ocupacao[PosFim, 3 - Jog].V),
        100 * (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
    end;
    R := R + Round(Random(3000));
  end;
  Teste31.Seta(True, PosIni, PosFim, R);
end;

function Teste32(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
  { usada em conjunto com Joga_A }
var
  R, I: longint;
begin
  I := 0;

  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;

  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    I := PosFim;
    if TestaFormaMoinho(MO, I, 3 - Jog, True) then
      R := 5000
    else
    begin
      R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V - MO.Ocupacao[PosFim, 3 - Jog].V),
        100 * (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
      R := R + Round(Random(3000));
    end;
  end;
  Teste32.Seta(True, PosIni, PosFim, R);
end;

function Teste33(MO: TTabuleiro; Jog: T12; PosIni, PosFim: TPosRange): TRetJoga;
  { usada em conjunto com Joga_A ou Joga_B}
var
  R, I: longint;
begin
  I := 0;

  R := 0;
  MO.DeslocaPedra(PosIni, PosFim);
  Randomize;

  if MO.AnalisaTrilha(PosFim) > 0 then
    R := 5000
  else
  begin
    I := PosFim;
    if TestaFormaMoinho(MO, I, 3 - Jog, True) then
      R := 5000
    else
    begin
      R := MaxI(100 * (MO.Ocupacao[PosFim, Jog].V - MO.Ocupacao[PosFim, 3 - Jog].V),
        100 * (MO.Ocupacao[PosFim, Jog].H - MO.Ocupacao[PosFim, 3 - Jog].H)){+};
      R := R + Round(Random(3000));
      if TestaFormaMoinho(MO, PosIni, 3 - Jog, True) then
        R := 0;
    end;
  end;
  Teste33.Seta(True, PosIni, PosFim, R);
end;


function Joga_A(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;
  { usada em conjunto com Teste3}
var
  Melhor: TRetJoga;
  Velho: TTabuleiro;
var
  I, J: integer;
  NTeste, Teste: TRetJoga; { NTeste : teste no nivel }

begin
  Dec(Profundidade);
  Velho := MO;
  Melhor.Seta(False, 1, 1, 0);
  for I := 1 to 24 do
  begin
    if csJogador[MO.Tab[I]] = Jog then   { procura pedras }
    begin
      J := 1;
      while (J <= 4) and (Movimentos[I][J] <> 0) do  { procura movimentos }
      begin
        if MO.Tab[Movimentos[I][J]] = peVazio  { pode mover }
        then
        begin
          if Profundidade =
            MaxProf then
            Teste := Teste33(MO, Jog, I, Movimentos[I][J])
          else
            Teste := Testa(MO, Jog, I, Movimentos[I][J]);
          NTeste := Teste;
          if Teste.Nota > 3999 then
            Teste.Nota := Teste.Nota + (Profundidade + 1)
          else if (Profundidade > 0) and (abs(Teste.Nota) <> 5000) then
          begin
            MO.DeslocaPedra(I, Movimentos[I][J]);
            Teste := Joga(MO,{3-}Jog, Profundidade, Testa);
            MO := Velho;
            if not (Teste.Achei) then
            begin
              Teste.Seta(True, I, Movimentos[I][J], 1);
            end;
          end; { of internal if }

          if Teste.Achei then
          begin
            if ((Teste.Nota >= Melhor.Nota) and not (Melhor.Achei)) or
              (Teste.Nota > Melhor.Nota) or
              (Melhor.Achei and (Teste.Nota = Melhor.Nota) and
              (random(1000) > 100)) then
              Melhor.Seta(True, NTeste.MelhorIni, NTeste.MelhorFim, Teste.Nota);
          end;
        end; { of external if pode mover}
        Inc(J);
      end; { of while }
    end; { of if procura pedras }
  end; { of for }
  Joga_A := Melhor;
end; { of procedure Joga }

function Joga_B(MO: TTabuleiro; Jog: T12;
  Profundidade: byte; Testa: TTesta): TRetJoga;
  { usada em conjunto com Teste33}
var
  Melhor: TRetJoga;
  Velho: TTabuleiro;
var
  I, J: integer;
  NTeste, Teste: TRetJoga; { NTeste : teste no nivel }

begin
  Dec(Profundidade);
  Velho := MO;
  Melhor.Seta(False, 1, 1, 0);
  for I := 1 to 24 do
  begin
    if csJogador[MO.Tab[I]] = Jog then   { procura pedras }
    begin
      J := 1;
      while (J <= 4) and (Movimentos[I][J] <> 0) do  { procura movimentos }
      begin
        if MO.Tab[Movimentos[I][J]] = peVazio  { pode mover }
        then
        begin
          if Profundidade =
            MaxProf then
            Teste := Teste33(MO, Jog, I, Movimentos[I][J])
          else
            Teste := Testa(MO, Jog, I, Movimentos[I][J]);
          NTeste := Teste;
          if (Teste.Nota > 3999) or (Teste.Nota < 10)
          then
            Teste.Nota := Teste.Nota + (Profundidade + 1)
          else if (Profundidade > 0)
          {and (abs(Teste.Nota)<>5000)} then
          begin
            MO.DeslocaPedra(I, Movimentos[I][J]);
            Teste := Joga_B(MO,{3-}Jog, Profundidade, Testa);
            Teste.Nota :=
              MaxI(Teste.Nota - 1, Testa(
              MO, Jog, I, Movimentos[I][J]).Nota);
            MO := Velho;
            if not (Teste.Achei) then
            begin
              Teste.Seta(True, I, Movimentos[I][J], MaxProf);
            end;
          end; { of internal if }

          if Teste.Achei then
          begin
            if ((Teste.Nota >= Melhor.Nota) and not (Melhor.Achei)) or
              (Teste.Nota > Melhor.Nota) or
              (Melhor.Achei and (Teste.Nota = Melhor.Nota) and
              (random(1000) > 500)) then
              Melhor.Seta(True, NTeste.MelhorIni, NTeste.MelhorFim, Teste.Nota);
          end;
        end; { of external if pode mover}
        Inc(J);
      end; { of while }
    end; { of if procura pedras }
  end; { of for }
  Joga_B := Melhor;
end; { of procedure Joga }


end.
