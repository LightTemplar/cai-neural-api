unit URobMin2;

interface

const
  csNumberActions = 7;

const
  // Actions
  csLeft = 0;
  csRight = 1;
  csUp = 2;
  csDown = 3;
  csHold = 4;

  csLoad = 5;
  csUnLoad = 6;

  csUpLeft = 7;
  csUpRight = 8;
  csDownLeft = 9;
  csDownRight = 10;
  csCenter = 11;

  //States
  csLoaded = 0;
  csUnloaded = 1;

  // base index
  csBaseIdx = 0;

  // mining index
  csMiningIdx = 1;

type
  TPosition = record
    // X and Y can vary from 0 until WorldLength-1.
    X, Y: extended;

    //States can be unloaded (csUnloaded)and loaded (csLoaded).
    State: longint;
  end;

const
  csMaxAgents = 5;

type
  TOtherPos = array[0..csMaxAgents - 2] of TPosition;

type
  TArtificialMiningWorld = object
    AgentP: array[0..csMaxAgents - 1] of TPosition;
    ChoosedMoves: array[0..csMaxAgents - 1] of byte;

    // number of movements.
    NMoves: longint;
    FWorldLength: longint;

    // Allows Superposition?
    Superposition: boolean;
    FLinkedBorder: boolean;
    NumUnLoad, NumLoad: longint;

    function TestCapture: boolean;
    procedure DefineWorldLength(PL: longint);
    procedure DefineSuperposition(S: boolean);
    procedure DefineLinkedBorder(LB: boolean);
    procedure Left(P: byte);
    procedure Right(P: byte);
    procedure Up(P: byte);
    procedure Down(P: byte);
    procedure UnLoad(P: byte);
    procedure Load(P: byte);
    function MoveAgent(Direction, P: byte): boolean;
    procedure ExecuteChoosedMoves;
    procedure ChooseMove(Direction, P: byte);
    procedure MoveAgentR(DX, DY: extended; P: byte);
    procedure RandomMove(P: byte);
    procedure RandomPossibleMove(P: byte);
    function GetRelativePositions(P: byte): TOtherPos;
    function GetRelativeDistance(P: byte): TOtherPos;
    function Distance(P, Q: byte): extended;

    // Manhatam Distance
    function MD(P, Q: byte): extended;

    // Manhatam Distance
    function MDPos(P, Q: TPosition): extended;

    // Norm Distance
    function NDPos(P, Q: TPosition): extended;

    procedure RandomPos;
    procedure DefinePos(P: byte; X, Y: extended);
    function FixD(X: extended): extended;
    procedure FixPos(var X, Y: extended);
    function UsedPos(X, Y: extended): boolean;
    function CalcNextPos(Direction, P: byte): TPosition;
    function Encounter(P1, P2: byte): boolean;
  end;

implementation

function TArtificialMiningWorld.TestCapture: boolean;
var
  J: integer;
  R: extended;
  D: TOtherPos;
begin
  R := 0;
  D := GetRelativePositions(0);
  for J := 0 to 3 do
    R := R + sqrt(sqr(D[J].X) + sqr(D[J].Y));
  TestCapture := (R <= 4);
end;


procedure TArtificialMiningWorld.DefineWorldLength(PL: longint);
begin
  FWorldLength := PL;
  NumUnLoad := 0;
  NumLoad := 0;
  NMoves := 0;
end;

procedure TArtificialMiningWorld.FixPos(var X, Y: extended);
begin
  if X < 0 then
    X := X + FWorldLength;
  if X >= FWorldLength then
    X := X - FWorldLength;
  if Y < 0 then
    Y := Y + FWorldLength;
  if Y >= FWorldLength then
    Y := Y - FWorldLength;
end;

procedure TArtificialMiningWorld.MoveAgentR(DX, DY: extended; P: byte);
begin
  AgentP[P].X := AgentP[P].X + DX;
  AgentP[P].Y := AgentP[P].Y + DY;
  FixPos(AgentP[P].X, AgentP[P].Y);
end;

procedure TArtificialMiningWorld.Left(P: byte);
begin
  AgentP[P] := CalcNextPos(csLeft, P);
end;

procedure TArtificialMiningWorld.Right(P: byte);
begin
  AgentP[P] := CalcNextPos(csRight, P);
end;

procedure TArtificialMiningWorld.Up(P: byte);
begin
  AgentP[P] := CalcNextPos(csUp, P);
end;

procedure TArtificialMiningWorld.Down(P: byte);
begin
  AgentP[P] := CalcNextPos(csDown, P);
end;

procedure TArtificialMiningWorld.UnLoad(P: byte);
begin
  if (AgentP[P].State = 1) and Encounter(P, 0) then
  begin
    Inc(NumUnLoad);
    Dec(AgentP[P].State);
    Inc(AgentP[0].State);
  end;
end;

procedure TArtificialMiningWorld.Load(P: byte);
var
  I: integer;
begin
  if AgentP[P].State = 0 then
  begin
    for I := 1 to 3 do
    begin
      if Encounter(P, I) then
      begin
        Inc(NumLoad);
        Inc(AgentP[P].State);
        Dec(AgentP[I].State);
        if AgentP[I].State = 0 then
        begin
          AgentP[I].X := -10;
          AgentP[I].Y := -10;
        end;
        exit;
      end;
    end;
  end;
end;

function TArtificialMiningWorld.CalcNextPos(Direction, P: byte): TPosition;
var
  NextX, NextY: extended;
begin
  NextX := AgentP[P].X;
  NextY := AgentP[P].Y;
  case Direction of
    csLeft:
    begin
      NextX := AgentP[P].X - 1;
      if NextX < 0 then
      begin
        if FLinkedBorder then
          NextX := NextX + FWorldLength
        else
          NextX := 0;
      end;
    end;
    csRight:
    begin
      NextX := (AgentP[P].X + 1);
      if NextX >= FWorldLength then
      begin
        if FLinkedBorder then
          NextX := NextX - FWorldLength
        else
          NextX := FWorldLength - 1;
      end;
    end;
    csUp:
    begin
      NextY := AgentP[P].Y - 1;
      if NextY < 0 then
      begin
        if FLinkedBorder then
          NextY := NextY + FWorldLength
        else
          NextY := 0;
      end;
    end;
    csDown:
    begin
      NextY := (AgentP[P].Y + 1);
      if NextY >= FWorldLength then
      begin
        if FLinkedBorder then
          NextY := NextY - FWorldLength
        else
          NextY := FWorldLength - 1;
      end;
    end;
  end; // of case;

  if UsedPos(NextX, NextY) then
  begin
    NextX := AgentP[P].X;
    NextY := AgentP[P].Y;
  end;
  CalcNextPos.X := NextX;
  CalcNextPos.Y := NextY;
  CalcNextPos.State := AgentP[P].State;
end;

function TArtificialMiningWorld.MoveAgent(Direction, P: byte): boolean;
var
  OX, OY: extended;
begin
  OX := AgentP[P].X;
  OY := AgentP[P].Y;
  case Direction of
    csLeft: Left(P);
    csRight: Right(P);
    csUp: Up(P);
    csDown: Down(P);
    csLoad: Load(P);
    csUnLoad: UnLoad(P);
    csUpLeft:
    begin
      Up(P);
      Left(P);
    end;
    csUpRight:
    begin
      Up(P);
      Right(P);
    end;
    csDownLeft:
    begin
      Down(P);
      Left(P);
    end;
    csDownRight:
    begin
      Down(P);
      Right(P);
    end;
    csCenter:
    begin
      AgentP[P].X := FWorldLength div 2;
      AgentP[P].Y := FWorldLength div 2;
    end;
  end; // of case;
  MoveAgent := ((OX <> AgentP[P].X) or (OY <> AgentP[P].Y)) and (Direction <> 4);
  Inc(NMoves);
end;

procedure TArtificialMiningWorld.ExecuteChoosedMoves;
var
  I: longint;
begin
  for I := Low(ChoosedMoves) to High(ChoosedMoves) do
  begin
    MoveAgent(ChoosedMoves[I], I);
    ChoosedMoves[I] := csHold;
  end;
end;

procedure TArtificialMiningWorld.ChooseMove(Direction, P: byte);
begin
  ChoosedMoves[P] := Direction;
end;

procedure TArtificialMiningWorld.RandomMove(P: byte);
begin
  MoveAgent(Random(4), P);
end;

procedure TArtificialMiningWorld.RandomPossibleMove(P: byte);
var
  OX, OY: extended;
  I: integer;
begin
  OX := AgentP[P].X;
  OY := AgentP[P].Y;
  for I := 1 to 8 do
  begin
    RandomMove(P);
    if (OX <> AgentP[P].X) or (OY <> AgentP[P].Y) then
      exit;
    AgentP[P].X := OX;
    AgentP[P].Y := OY;
  end;
end;


function TArtificialMiningWorld.FixD(X: extended): extended;
var
  Max: longint;
begin
  if FLinkedBorder then
  begin
    Max := (FWorldLength div 2) + 1;
    if X < -Max then
      X := X + FWorldLength
    else
    begin
      if X > Max then
        X := X - FWorldLength;
    end;
  end;
  FixD := X;
end;

function TArtificialMiningWorld.GetRelativePositions(P: byte): TOtherPos;
var
  I, PR: integer;
  // Agent P Position
  PosX, PosY: extended;
begin
  PR := 0;
  PosX := AgentP[P].X;
  PosY := AgentP[P].Y;
  for I := 0 to 4 do
  begin
    if I <> P then
    begin
      GetRelativePositions[PR].X := FixD(AgentP[I].X - PosX);
      GetRelativePositions[PR].Y := FixD(AgentP[I].Y - PosY);
      Inc(PR);
    end; // of if
  end; // of for
end;

function TArtificialMiningWorld.GetRelativeDistance(P: byte): TOtherPos;
var
  I, PR: integer;
  // Agent P Position
  PosX, PosY: extended;
var
  Max: extended;
begin
  Max := (FWorldLength div 2) + 1;
  PR := 0;
  PosX := AgentP[P].X;
  PosY := AgentP[P].Y;
  for I := 0 to 4 do
  begin
    if I <> P then
    begin
      GetRelativeDistance[PR].X := (((FixD(AgentP[I].X - PosX))) / MAX);
      GetRelativeDistance[PR].Y := (((FixD(AgentP[I].Y - PosY))) / MAX);
      Inc(PR);
    end; // of if
  end; // of for
end;

function TArtificialMiningWorld.MD(P, Q: byte): extended;
begin
  MD := MDPos(AgentP[P], AgentP[Q]);
end;

function TArtificialMiningWorld.MDPos(P, Q: TPosition): extended;
var
  Dx, Dy: extended;
begin
  Dx := FixD(P.X - Q.X);
  Dy := FixD(P.Y - Q.Y);
  MDPos := Abs(Dx) + Abs(Dy);
end;

function TArtificialMiningWorld.NDPos(P, Q: TPosition): extended;
var
  Dx, Dy: extended;
begin
  Dx := FixD(P.X - Q.X);
  Dy := FixD(P.Y - Q.Y);
  NDPos := sqrt(sqr(Dx) + sqr(Dy));
end;

function TArtificialMiningWorld.Distance(P, Q: byte): extended;
var
  Dx, Dy: extended;
begin
  Dx := FixD(AgentP[P].X - AgentP[Q].X);
  Dy := FixD(AgentP[P].Y - AgentP[Q].Y);
  Distance := sqrt(Dx * Dx + Dy * Dy);
end;

procedure TArtificialMiningWorld.RandomPos;
var
  I: integer;
begin
  for I := 0 to 4 do
  begin
    AgentP[I].X := Random(FWorldLength);
    AgentP[I].Y := Random(FWorldLength);
  end;
end;

procedure TArtificialMiningWorld.DefinePos(P: byte; X, Y: extended);
begin
  AgentP[P].X := X;
  AgentP[P].Y := Y;
end;

function TArtificialMiningWorld.UsedPos(X, Y: extended): boolean;
var
  I: integer;
  F: boolean;
begin
  F := False;
  if not (Superposition) then
    for I := 0 to 4 do
    begin
      F := F or ((AgentP[I].X = X) and (AgentP[I].Y = Y));
    end;
  UsedPos := F;
end;

procedure TArtificialMiningWorld.DefineSuperposition(S: boolean);
begin
  Superposition := S;
end;

procedure TArtificialMiningWorld.DefineLinkedBorder(LB: boolean);
begin
  FLinkedBorder := LB;
end;

function TArtificialMiningWorld.Encounter(P1, P2: byte): boolean;
var
  dx, dy: extended;
begin
  dx := abs(AgentP[P1].X - AgentP[P2].X);
  dy := abs(AgentP[P1].Y - AgentP[P2].Y);
  Encounter := (dx <= 1) and (dy <= 1);
end;

end.
