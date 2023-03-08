unit URobMin; // Predator and Prey Unit v1.1

interface

const csLeft  = 0;
      csRight = 1;
      csUp    = 2;
      csDown  = 3;
      csHold  = 4;

      csDescarregado = 0;
      csCarregado = 1;

      iBase = 0; // indice da base
      iMina = 1; // indice da mina


type TPos = record // as posicoes variam de 0 ateh L-1
            X,Y:extended;
            Estado:byte;  //Descarregado ou Carregado
            end;

const csMaxAgents = 5;

type TOtherPos = array[0..csMaxAgents-2] of TPos;


type TRobMin = object
              AgentP:array[0..csMaxAgents-1] of TPos; // 0:Prey ; 1..4 Predators
              L:longint;  // tamanho do mundo
              Superposition:boolean; // permite superposicao
              FLinkedBorder:boolean;

              procedure DefineL(PL:longint);
              procedure DefineSuperposition(S:boolean);
              procedure DefineLinkedBorder(LB:boolean);
              procedure Left(P:byte);
              procedure Right(P:byte);
              procedure Up(P:byte);
              procedure Down(P:byte);
              function MoveAgent(Direction,P:byte):boolean;
              procedure MoveAgentR(DX,DY:extended;P:byte);
              procedure RandomMove(P:byte);
              procedure RandomPossibleMove(P:byte);
              function GetRelativePositions(P:byte):TOtherPos;
              function GetRelativeDistance(P:byte):TOtherPos;
              function Distance(P,Q:byte):extended;
              procedure RandomPos;
              procedure DefinePos(P:byte;X,Y:extended);
              function FixD(X:extended):extended;
              procedure FixPos(var X,Y:extended);
              function UsedPos(X,Y:extended):boolean;
              function Encounter(P1,P2:byte);
              end;

implementation

procedure TPPrey.DefineL(PL:longint);
begin
L:=PL;
end;

procedure TPPrey.FixPos(var X,Y:extended);
begin
if X<0 then X:=X+L;
if X>=L then X:=X-L;
if Y<0 then Y:=Y+L;
if Y>=L then Y:=Y-L;
end;

procedure TPPrey.MoveAgentR(DX,DY:extended;P:byte);
begin
AgentP[P].X:=AgentP[P].X+DX;
AgentP[P].Y:=AgentP[P].Y+DY;
FixPos(AgentP[P].X,AgentP[P].Y);
end;

procedure TPPrey.Left(P:byte);
var NextX:extended;
begin
NextX:=AgentP[P].X-1;
if NextX<0
      then begin
           if FLinkedBorder
              then NextX:=NextX+L
              else NextX:=0;
           end;
if not(UsedPos(NextX,AgentP[P].Y))
   then AgentP[P].X:=NextX;
end;

procedure TPPrey.Right(P:byte);
var NextX:extended;
begin
NextX:=(AgentP[P].X + 1);
if NextX>=L
   then begin
        if FLinkedBorder
           then NextX:=NextX-L
           else NextX:=L-1;
        end;
if not(UsedPos(NextX,AgentP[P].Y))
   then AgentP[P].X:=NextX;
end;

procedure TPPrey.Up(P:byte);
var NextY:extended;
begin
NextY:=AgentP[P].Y-1;
if NextY<0
   then begin
        if FLinkedBorder
           then NextY:=NextY+L
           else NextY:=0;
        end;
if not(UsedPos(AgentP[P].X,NextY))
   then AgentP[P].Y:=NextY;
end;

procedure TPPrey.Down(P:byte);
var NextY:extended;
begin
NextY:=(AgentP[P].Y + 1);
if NextY>=L
   then begin
        if FLinkedBorder
           then NextY:=NextY-L
           else NextY:=L-1;
        end;
if not(UsedPos(AgentP[P].X,NextY))
   then AgentP[P].Y:=NextY;
end;

function TPPrey.MoveAgent(Direction,P:byte):boolean;
var OX,OY:extended;
begin
OX:=AgentP[P].X;
OY:=AgentP[P].Y;
Case Direction of
     0: Left(P);
     1: Right(P);
     2: Up(P);
     3: Down(P);
     end; // of case;
MoveAgent:= (OX<>AgentP[P].X) or (OY<>AgentP[P].Y);
end;

procedure TPPrey.RandomMove(P:byte);
begin
MoveAgent(Random(4),P);
end;

procedure TPPrey.RandomPossibleMove(P:byte);
var OX,OY:extended;
    I:integer;
begin
OX:=AgentP[P].X;
OY:=AgentP[P].Y;
for I:=1 to 8
    do begin
       RandomMove(P);
       if (OX<>AgentP[P].X) or (OY<>AgentP[P].Y)
          then exit;
       end;
end;

function TPPrey.FixD(X:extended):extended;
var Max:longint;
begin
if FLinkedBorder
   then begin
        Max:=( L div 2)+1;
        if X< -Max
           then X:=X+L
           else begin
                if X > Max
                   then X:=X-L;
                end;
        end;
FixD:=X;
end;

function TPPrey.GetRelativePositions(P:byte):TOtherPos;
var I,PR : integer;
    PosX,PosY:extended; // Agent P Position
begin
PR:=0;
PosX:=AgentP[P].X;
PosY:=AgentP[P].Y;
for I:=0 to 4 do
    begin
    if I<>P then
       begin
       GetRelativePositions[PR].X:=FixD(AgentP[I].X-PosX);
       GetRelativePositions[PR].Y:=FixD(AgentP[I].Y-PosY);
       Inc(PR);
       end; // of if
    end; // of for
end;

function TPPrey.GetRelativeDistance(P:byte):TOtherPos;
var I,PR : integer;
    PosX,PosY:extended; // Agent P Position
var Max:extended;
begin
Max:=( L div 2)+1;
PR:=0;
PosX:=AgentP[P].X;
PosY:=AgentP[P].Y;
for I:=0 to 4 do
    begin
    if I<>P then
       begin
       GetRelativeDistance[PR].X:=( ((FixD(AgentP[I].X-PosX)){-1})/MAX);
       GetRelativeDistance[PR].Y:=( ((FixD(AgentP[I].Y-PosY)){-1})/MAX);
       Inc(PR);
       end; // of if
    end; // of for
end;

function TPPrey.Distance(P,Q :byte):extended;
var Dx,Dy:extended;
begin
Dx:=FixD(AgentP[P].X-AgentP[Q].X);
Dy:=FixD(AgentP[P].Y-AgentP[Q].Y);
Distance:=sqrt(Dx*Dx+Dy*Dy);
end;

procedure TPPrey.RandomPos;
var I : integer;
begin
for I:=0 to 4 do
    begin
    AgentP[I].X:=Random(L);
    AgentP[I].Y:=Random(L);
    end;
end;

procedure TPPrey.DefinePos(P:byte;X,Y:extended);
begin
AgentP[P].X:=X;
AgentP[P].Y:=Y;
end;

function TPPrey.UsedPos(X,Y:extended):boolean;
var I : integer;
    F:boolean;
begin
F:=false;
if not(Superposition) then
   for I:=0 to 4 do
       begin
       F:=F or ( (AgentP[I].X=X) and (AgentP[I].Y=Y) );
       end;
UsedPos:=F;
end;

procedure TPPrey.DefineSuperposition(S:boolean);
begin
Superposition:=S;
end;

procedure TPPrey.DefineLinkedBorder(LB:boolean);
begin
FLinkedBorder:=LB;
end;

function TPPrey.Encounter(P1,P2:byte);
var dx,dy:extended;
begin
dx:=abs(AgentP[P1].X - AgentP[P2].X);
dy:=abs(AgentP[P1].Y - AgentP[P2].Y);
Encounter:= (dx<=1) and (dy<=1);
end;


end.
