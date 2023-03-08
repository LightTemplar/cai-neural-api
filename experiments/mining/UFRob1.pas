unit UFRob1;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls, URobMin2, ExtCtrls, Menus,
  neuralbyteprediction, neuralab, neuralplanbuilder, Buttons, neuralgeneric;

const
  csStateByteLength = 8;
  csWorldLength = 100;
  csMaxPlanejamento = round(csWorldLength * 1.5);

// Agent Information.
type
  TMiningAgentInf = array[0..csStateByteLength - 1] of byte;

  // The Agent
  TMiningAgent = object
    LearningAndPredict: TEasyLearnAndPredictClass;
    CompositePlanObj: TCompositePlan;
    PreviousXPosition, PreviousYPosition, PreviousAction: byte;
    PlannedActionsCnt: longint;
    PlanningErrorCnt: longint;
    RandomActionsCnt: longint;
    LastPlaned: boolean;
    FNextState, FCurrentState: array [0..csStateByteLength - 1] of byte;
    LastStates, LastPlanedStates: TActionStateList;
    LastTargetPlan: longint;

    FPreferred: boolean;
    PreferredAction: byte;

    // Creates needed internal memory structures for learning and planning.
    constructor Init(pActionByteLen, pStateByteLen: word);

    // Given a current state, this function is capable of choosing an action and
    // build/optimize plans as required.
    function ChooseActionAndPlan
      (
    {input} var pCurrentState: array of byte;
    {input}     UsePlanning, UseLearning: boolean): word;

    // This function predicts the next state and returns TRUE
    // if the ACTION is in the PATH of the target state (load/unload).
    function PredictNextState
      (
    {input/output} var CurrentState: array of byte;
    {input}        Action: byte): boolean;

    // This function predicts the next state and returns TRUE
    // if the ACTION brings to the "targed" state in just one step.
    function EasyPredictNextState
      (
    {input/output} var pCurrentState: array of byte;
    {input}        Action: byte): boolean;
  end;


type

  { TForm1 }

  TForm1 = class(TForm)
    GrMundo: TGroupBox;
    LBStatus: TLabel;
    Shape1: TShape;
    Shape2: TShape;
    Shape3: TShape;
    Shape4: TShape;
    Shape5: TShape;
    OpenDialog1: TOpenDialog;
    SaveDialog1: TSaveDialog;
    BBComeca: TBitBtn;
    BBOpcoes: TBitBtn;
    BBSair: TBitBtn;
    LabAg1: TLabel;
    LabAg2: TLabel;
    LabAg3: TLabel;
    LabAg4: TLabel;
    LabAg5: TLabel;
    LabNMoves: TLabel;
    BBPlanos: TBitBtn;
    BBMedicao: TBitBtn;
    LabCiclos: TLabel;
    procedure Button1Click(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure RunManySimulations(Sender: TObject);
    procedure MMStopClick(Sender: TObject);
    procedure MMExitClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var Action: TCloseAction);
    procedure BBSairClick(Sender: TObject);
    procedure BBOpcoesClick(Sender: TObject);
    procedure BBPlanosClick(Sender: TObject);
    procedure BBMedicaoClick(Sender: TObject);
  private
    { Private declarations }
    FShouldStop: boolean; // indicates that the process should stop
    IniPos: array[0..4] of TPosition; // initial positions
    Speed: extended; // multiplicador de velocidade para agentes
  public
    { Public declarations }
    MiningAgent: TMiningAgent;
    FMiningWorld: TArtificialMiningWorld;
    D: TOtherPos;
    ED: TOtherPos;
    S: array[0..4] of TShape;
    Labs: array[0..4] of TLabel;
    FSavedRandSeed: TRandom;
    LastNow: TDateTime;
    FSuccessCount: longint; // Number of Successes
    FPredictionErrors: integer;

    function RunOneSimulation: extended;
    procedure RunLearning;
    function MoveNeuralAgent(var pMiningAgent: TMiningAgent; P: byte): extended;
    function SimulateOneCycle: extended;
    procedure Mostra;
    procedure MostraPlanos;
    procedure PrintPropPop;
    procedure DefineInitialPositions;
    procedure ApplyInitialPositions;

  end;


var
  Form1: TForm1;

implementation

uses UForOptMin, UVPlan;

{$R *.lfm}

constructor TMiningAgent.Init(pActionByteLen, pStateByteLen: word);
begin
  LearningAndPredict.Initiate(pActionByteLen, pStateByteLen, False{pFullEqual},
    100{relationTableSize}, 40{pNumberOfSearches}, True{pUseCache});
  LearningAndPredict.BytePred.FUseBelief := True;
  LearningAndPredict.BytePred.FGeneralize := True;
  CompositePlanObj.Init(Self.PredictNextState, Self.EasyPredictNextState,
    csNumberActions, csStateByteLength);
  PlannedActionsCnt := 0;
  PlanningErrorCnt := 0;
  RandomActionsCnt := 0;
  LastPlaned := False;
  LastStates.Init(csStateByteLength);
  LastPlanedStates.Init(csStateByteLength);
  FPreferred := False;
end;

// Is this the action that will allow the agent to load/unload (the ultimate target)?
function IsTargetAction(C: array of byte; Action: byte): boolean;
begin
  IsTargetAction :=
    ((C[3] = 1) {loaded} and (C[4] = 1) {close to destination} and
    (Action = csUnLoad)) or // unload
    ((C[3] = 0) {not loaded} and (C[5] = 1) {close to origin} and (Action = csLoad));
  // load
end;

function TMiningAgent.PredictNextState(var CurrentState: array of byte;
  Action: byte): boolean;
var
  CLixo: array[0..csStateByteLength - 1] of byte;
  LP: boolean;
  LIXO: byte;
begin
  CompositePlanObj.SaveState;
  PredictNextState := False;
  LastTargetPlan := -1;
  if EasyPredictNextState(CurrentState, Action) then
    PredictNextState := True
  else
  begin
    if CompositePlanObj.ToAct(CurrentState, LIXO, LP, CLixo) then
    begin
      LastTargetPlan := CompositePlanObj.LastUsedPlan;
      PredictNextState := True;
    end;
  end;
  CompositePlanObj.RestoreState;
end;

function TMiningAgent.EasyPredictNextState(var pCurrentState: array of byte;
  Action: byte): boolean;
var
  localCurrentState: array[0..csStateByteLength - 1] of byte;
  PredictedState: array[0..csStateByteLength - 1] of byte;
begin
  ABCopy(localCurrentState, pCurrentState);
  localCurrentState[0] := Action + 1;
  LearningAndPredict.Predict(localCurrentState, localCurrentState, PredictedState);
  ABCopy(pCurrentState, PredictedState);
  if IsTargetAction(localCurrentState, Action) then
    EasyPredictNextState := True
  else
    EasyPredictNextState := False;
end;

function bFixD(B: byte): byte;
begin
  if B > 15 then
    bFixD := abs(30 - B)
  else
    bFixD := B;
end;

function TMiningAgent.ChooseActionAndPlan(var pCurrentState: array of byte; UsePlanning, UseLearning: boolean): word;

  procedure AddState(var localCurrentState: array of byte; localAction: byte);
  begin
    if (localAction <> 4) then
    begin
      if LastStates.NumStates = MaxStates - 1 then
        LastStates.RemoveFirst;
      LastStates.Include(localCurrentState, localAction);
    end;
  end; // of procedure AddState

  procedure IncludeLastPlanedState(var localCurrentState: array of byte;
    localAction: byte);
  begin
    if (localAction <> 4) then
    begin
      if LastPlanedStates.NumStates = MaxStates - 1 then
        LastPlanedStates.RemoveFirst;
      LastPlanedStates.Include(localCurrentState, localAction);
    end;
  end; // of procedure IncludeLastPlanedState

  function MakePlan(var localCurrentState: array of byte): boolean;
  var
    Action: byte;
    LastAction: boolean;
  begin
    MakePlan := False;
    LastPlaned := False;

    if CompositePlanObj.MultipleRun(localCurrentState, csMaxPlanejamento, 1) then
    begin
      if (LastTargetPlan <> -1) then
        CompositePlanObj.CollapsePlans(CompositePlanObj.LastPlanedPlan, LastTargetPlan);

      if CompositePlanObj.ToAct(localCurrentState, Action, LastAction, FNextState) then
      begin
        LastPlaned := True;
        ChooseActionAndPlan := Action;
        MakePlan := True;
      end;
    end
    else
    begin
      if FormOpt.CBShowPlan.Checked then
        FormViewPlans.ShowPlan(CompositePlanObj.Plans[CompositePlanObj.LastPlanedPlan],
          csWorldLength);

      Application.ProcessMessages;
    end;
  end;// of MakePlan

  procedure AddPlan(notCollapsePlans: boolean);
  begin
    if (LastStates.NumStates > 0) then
    begin
      AddState(pCurrentState, PreviousAction);
      LastStates.RemoveAllCicles;
      CompositePlanObj.ReceivePlan(LastStates);
      if not (notCollapsePlans) then
        CompositePlanObj.CollapsePlans(CompositePlanObj.LastReceivedPlan,
          CompositePlanObj.LastUsedPlan);
      LastStates.Clear;
    end;
  end;

var
  I: longint;
  localCurrentState: array [0..csStateByteLength - 1] of byte;
  localAction: byte;
  lastAction, WantQuit: boolean;
  LastUsedPlan: longint;

begin
  ChooseActionAndPlan := 4;
  WantQuit := False;
  ABClear(localCurrentState);
  ABCopy(localCurrentState, pCurrentState);

  if not (LastPlaned) and (FCurrentState[3] <> pCurrentState[3]) then
    AddPlan(True);

  LastUsedPlan := CompositePlanObj.LastUsedPlan;
  if FormOpt.TBOptimization.Position > random(100) // Optimization ?
  then
  begin
    if random(2) > 0 then
      CompositePlanObj.MultipleOptimize(localCurrentState, 5, 20)
    else
      CompositePlanObj.Optimize(localCurrentState, 95);
    if LastPlaned then
    begin
      CompositePlanObj.MultipleOptimizeLastUsedPlan(localCurrentState, 5, 200);
      CompositePlanObj.MultipleOptimizeLastUsedPlan(localCurrentState, 95, 10);
    end;
  end;

  if LastUsedPlan <> CompositePlanObj.LastUsedPlan then
    Writeln('ERROR!!!! Plan used and changed along optimization!!!');

  if not (WantQuit) and UseLearning then
  begin
    for I := 0 to csNumberActions - 1 do
      if IsTargetAction(pCurrentState, I) then
      begin
        localAction := I;
        ChooseActionAndPlan := localAction;
        WantQuit := True;
      end;
  end;

  if not (WantQuit) and UsePlanning then
  begin
    if FormOpt.CBEliminaIncorreto.Checked and // should eliminate incorrect plan?
      LastPlaned and not (ABCmp(FNextState, pCurrentState)) and
      not (FCurrentState[3] <> pCurrentState[3]) then
    begin
      CompositePlanObj.InvalidateLastUsedPlan;
      Inc(PlanningErrorCnt);
    end;

    if CompositePlanObj.ToAct(localCurrentState, localAction,
      lastAction, FNextState) then
    begin // follows planning
      if not (LastPlaned) then
        AddPlan(False);

      if (LastPlanedStates.FastExists(pCurrentState) <> -1) // Is this a  cycle?
      then
        CompositePlanObj.InvalidateLastUsedPlan; // delete cycling plan.

      ChooseActionAndPlan := localAction;
      Inc(PlannedActionsCnt);
      LastPlaned := True;
      WantQuit := True;//exit;
    end
    else
    begin // create plan

      // not a good outcome ???
      if not (IsTargetAction(FCurrentState, PreviousAction)) and
        LastPlaned and not (FCurrentState[3] <> pCurrentState[3])
      then
        CompositePlanObj.InvalidateLastUsedPlan;    //invalidate plan

      AddState(pCurrentState, PreviousAction);
      if FormOpt.CBNovoPlano.Checked // should create new plans ???
      then
      begin
        if MakePlan(localCurrentState) then
        begin
          LastStates.Clear;
          WantQuit := True;//exit;
        end;
      end
      else
      begin  // do not create a new plan
        LastPlaned := False;
      end;
    end;
  end
  else
  begin // do not use planning
    LastPlaned := False;
    AddState(pCurrentState, PreviousAction);
  end;

  if not (WantQuit) and LastPlaned then
    Writeln('ERROR:Last Planed1');

  if not (WantQuit) and (random(100) < FormOpt.TBRandom.Position) then
  begin
    if random(csWorldLength * csNumberActions) = 0 then
    begin
      FPreferred := (random(2) > 0);
      PreferredAction := random(csNumberActions);
    end;

    if FPreferred and (random(2) > 0) then
      localAction := PreferredAction
    else
      localAction := random(csNumberActions);

    ChooseActionAndPlan := localAction;
    Inc(RandomActionsCnt);
    WantQuit := True;
  end;

  if LastPlaned then
    IncludeLastPlanedState(pCurrentState, localAction)
  else
  begin
    LastPlanedStates.Clear;
    CompositePlanObj.ForgetLastUsedPlan;
  end;

  if IsTargetAction(FCurrentState, PreviousAction) then
    LastPlanedStates.Clear;
end;

procedure TForm1.Button1Click(Sender: TObject);
begin
  FMiningWorld.DefineWorldLength(csWorldLength);
  FMiningWorld.DefineSuperposition(False);
  FMiningWorld.DefineLinkedBorder(False);
  FMiningWorld.DefinePos(0, 50, 50);
  FMiningWorld.DefinePos(1, 20, 10);
  FMiningWorld.DefinePos(2, 29, 9);
  FMiningWorld.DefinePos(3, 3, 8);
  FMiningWorld.DefinePos(4, 14, 7);
  D := FMiningWorld.GetRelativePositions(0);
  ED := FMiningWorld.GetRelativeDistance(0);
  FShouldStop := False;
  while not (FShouldStop) do
  begin
    FMiningWorld.RandomMove(random(5));
    Mostra;
    Application.ProcessMessages;
  end;
end;

procedure TForm1.Mostra;
const
  F = 300 / csWorldLength;
var
  X, Y, I: integer;

begin
  for I := 0 to 4 do
  begin
    X := round(FMiningWorld.AgentP[I].X * F) + 10;
    Y := round(FMiningWorld.AgentP[I].Y * F) + 10;
    S[I].Left := X;
    S[I].Top := Y;
    Labs[I].Left := X + 5;
    Labs[I].Top := Y + 5;
    Labs[I].Caption := IntToStr(FMiningWorld.AgentP[I].State);
  end;
  LabNMoves.Caption := IntToStr(FMiningWorld.NMoves);
  if FormOpt.CBShowPlan.Checked then
    MostraPlanos;
  Application.ProcessMessages;
end;

procedure TForm1.MostraPlanos;
var
  I: longint;
begin
  FormViewPlans.Clear;
  for I := Low(MiningAgent.CompositePlanObj.Plans)
    to High(MiningAgent.CompositePlanObj.Plans) do
  begin
    if MiningAgent.CompositePlanObj.Plans[I].Found then
      FormViewPlans.ShowPlan(MiningAgent.CompositePlanObj.Plans[I], FMiningWorld.FWorldLength);
  end;
  for I := 0 to 4 do
    FormViewPlans.DefinePos(FMiningWorld.AgentP[I].X / FMiningWorld.FWorldLength,
      FMiningWorld.AgentP[I].Y / FMiningWorld.FWorldLength, I);
end;

procedure TForm1.Timer1Timer(Sender: TObject);
var
  J: integer;
begin
  for J := 1 to 100 do ;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  randomize;
  MiningAgent.Init(csStateByteLength, csStateByteLength);
  Speed := 1;
  S[0] := Shape1;
  S[1] := Shape2;
  S[2] := Shape3;
  S[3] := Shape4;
  S[4] := Shape5;
  Labs[0] := LabAg1;
  Labs[1] := LabAg2;
  Labs[2] := LabAg3;
  Labs[3] := LabAg4;
  Labs[4] := LabAg5;

  FMiningWorld.DefineWorldLength(csWorldLength);
  FMiningWorld.DefineSuperposition(False{true});
  FMiningWorld.DefineLinkedBorder(False);
end;

function TForm1.MoveNeuralAgent(var pMiningAgent: TMiningAgent; P: byte): extended;

  procedure Popperian;
  var
    localCurrentState: array[0..csStateByteLength - 1] of byte;
    PredictedStates: array[0..csStateByteLength - 1] of byte;
    LocalAction: word;
    R: extended;
  begin
    R := 1;

    ABClear(localCurrentState);
    localCurrentState[1] := Round(FMiningWorld.AgentP[P].X) + 1;
    localCurrentState[2] := Round(FMiningWorld.AgentP[P].Y) + 1;
    localCurrentState[3] := FMiningWorld.AgentP[P].State;

    if FMiningWorld.Encounter(0, 4) then
      localCurrentState[4] := 1;

    if FMiningWorld.Encounter(1, 4) or FMiningWorld.Encounter(2, 4) or
      FMiningWorld.Encounter(3, 4) then
      localCurrentState[5] := 1;

    localCurrentState[6] := (FMiningWorld.FWorldLength div 2) + 1;

    FPredictionErrors := FPredictionErrors +
      Round(pMiningAgent.LearningAndPredict.newStateFound(localCurrentState));

    LocalAction := pMiningAgent.ChooseActionAndPlan(localCurrentState,
      FormOpt.ChPlan.Checked, FormOpt.CHABC.Checked);

    pMiningAgent.PreviousXPosition := Round(FMiningWorld.AgentP[P].X);
    pMiningAgent.PreviousYPosition := Round(FMiningWorld.AgentP[P].Y);
    pMiningAgent.PreviousAction := LocalAction;

    localCurrentState[0] := LocalAction + 1;

    ABClear(PredictedStates);
    pMiningAgent.LearningAndPredict.Predict(localCurrentState, localCurrentState,
      PredictedStates);

    if LocalAction <> 4 then
      FMiningWorld.MoveAgent(LocalAction, P);

    MoveNeuralAgent := R;

    localCurrentState[0] := 0;
    ABCopy(pMiningAgent.FCurrentState, localCurrentState);
  end; // of procedure

begin
  Popperian;
  MoveNeuralAgent := 1;
end; // of procedure

function TForm1.SimulateOneCycle: extended;
var
  R: extended;
begin
  R := 0;
  R := R + MoveNeuralAgent(MiningAgent, 4);
  if FormOpt.ChMostra.Checked then
    Mostra;
  SimulateOneCycle := R;
end;


procedure TForm1.ApplyInitialPositions;
var
  I: integer;
begin
  for I := 0 to 4 do
    FMiningWorld.AgentP[I] := IniPos[I];
  MiningAgent.LastStates.Clear;
  MiningAgent.LastPlanedStates.Clear;
end;


function GetOneOrMinusOne: longint;
begin
  GetOneOrMinusOne := random(2) * 2 - 1;
end;

procedure TForm1.DefineInitialPositions;

  function CreateMine: TPosition;
  var
    PosBase, DistMin: extended;
    DX, DY: longint;
  begin
    PosBase := (FMiningWorld.FWorldLength div 2) + 1;
    DistMin := csWorldLength * 0.4;
    DX := random(round(DistMin));
    DY := round(DistMin) - DX;
    DX := DX * GetOneOrMinusOne;
    DY := DY * GetOneOrMinusOne;
    CreateMine.State := 100;
    CreateMine.X := PosBase + DX;
    CreateMine.Y := PosBase + DY;
  end;

var
  I: integer;
begin
  IniPos[0].X := FMiningWorld.FWorldLength div 2;
  IniPos[0].Y := FMiningWorld.FWorldLength div 2;
  IniPos[0].State := 0;
  for I := 1 to 3 do
  begin
    IniPos[I] := CreateMine;
  end;
  IniPos[4].State := 0;
  IniPos[4].X := (FMiningWorld.FWorldLength div 2) + 1;
  IniPos[4].Y := (FMiningWorld.FWorldLength div 2) + 1;
end;

procedure TForm1.PrintPropPop;
var
  K: longint;
begin
  Write('Planning: (right:wrong)');
  for K := 4 to 4 do
  begin
    Write(MiningAgent.PlannedActionsCnt: 10, ':', MiningAgent.PlanningErrorCnt);
    MiningAgent.PlannedActionsCnt := 0;
    MiningAgent.PlanningErrorCnt := 0;
  end;
  writeln;

  Write(' Guesses:');
  for K := 4 to 4 do
  begin
    Write(MiningAgent.RandomActionsCnt: 10);
    MiningAgent.RandomActionsCnt := 0;
  end;
  writeln;
  writeln(' Cache memory usage:',
    MiningAgent.LearningAndPredict.FCache.Used: 20: 10);
  writeln(' Cache Hits:',
    MiningAgent.LearningAndPredict.FCache.HitsOverAll: 20: 10);
  Writeln(' Loaded units:', FMiningWorld.NumLoad, '   Unloaded units:',
    FMiningWorld.NumUnLoad);
  Writeln(' Movements:', FMiningWorld.NMoves);
  FMiningWorld.NMoves := 0;
  FMiningWorld.NumLoad := 0;
  FMiningWorld.NumUnLoad := 0;
end; // or procedure


procedure TForm1.MMStopClick(Sender: TObject);
begin
  FShouldStop := True;
end;

procedure TForm1.MMExitClick(Sender: TObject);
begin
  Close;
end;

procedure TForm1.FormClose(Sender: TObject; var Action: TCloseAction);
begin
  FShouldStop := True;
  FormOpt.ChMostra.Checked := False;
end;

procedure TForm1.BBSairClick(Sender: TObject);
begin
  Close;
end;

procedure TForm1.BBOpcoesClick(Sender: TObject);
begin
  FormOpt.Show;
  FormOpt.Top := Self.Top;
  FormOpt.Left := Self.Left + Self.Width;
end;

procedure TForm1.BBPlanosClick(Sender: TObject);
begin
  FormViewPlans.Show;
end;

procedure TForm1.BBMedicaoClick(Sender: TObject);
begin
  PrintPropPop;
  ApplyInitialPositions;
end;

function TForm1.RunOneSimulation: extended;
var
  I, Runs: integer;
  localFoundSuccess: boolean;
begin
  Runs := StrToInt(FormOpt.EdCiclos.Text);
  ApplyInitialPositions;
  I := 0;
  localFoundSuccess := False;

  repeat
    Inc(I);

    if I mod 200 = 0 then
    begin
      Mostra;
      Application.ProcessMessages;
    end;

    if not FShouldStop then
      SimulateOneCycle;

    if (300 = FMiningWorld.AgentP[0].State) then
    begin
      localFoundSuccess := True;
      Inc(FSuccessCount);
    end;

  until FShouldStop or (FMiningWorld.NMoves >= Runs) or localFoundSuccess;

  RunOneSimulation := 0;
  Application.ProcessMessages;
end;

procedure TForm1.RunLearning;
var
  I, Runs: integer;
begin
  Runs := 1000000;
  FPredictionErrors := 0;
  ApplyInitialPositions;
  I := 0;
  FormOpt.TBOptimization.Position := 0;
  FormOpt.ChMostra.Checked := False;
  FormOpt.ChPlan.Checked := False;
  FormOpt.CBNovoPlano.Checked := False;
  FormOpt.CBEliminaIncorreto.Checked := False;

  Application.ProcessMessages;

  repeat
    Inc(I);

    if not FShouldStop then
      SimulateOneCycle;

    if (I mod 10000=0) then
    begin
      WriteLn(FPredictionErrors);
      FPredictionErrors := 0;
    end;

    if I mod 2000 = 0 then
    begin
      Mostra;
      Application.ProcessMessages;
    end;

  until FShouldStop or (FMiningWorld.NMoves >= Runs);

  Application.ProcessMessages;
end;

procedure TForm1.RunManySimulations(Sender: TObject);
begin
  with Sender as TBitBtn do
    Enabled := False;
  FSuccessCount := 0;
  FShouldStop := False;
  DefineInitialPositions;

  LBStatus.Caption := 'Neural Network is Learning - Please Wait';
  RunLearning;

  FormOpt.TBOptimization.Position := 20;
  FormOpt.ChPlan.Checked := True;
  FormOpt.CBNovoPlano.Checked := True;
  FormOpt.CBEliminaIncorreto.Checked := True;
  FormOpt.ChMostra.Checked := True;

  FormOpt.ChMostra.Enabled := True;
  FormOpt.ChPlan.Enabled := True;
  FormOpt.CBNovoPlano.Enabled := True;
  FormOpt.CBEliminaIncorreto.Enabled := True;

  LBStatus.Caption := 'Robot is Mining - Please Enjoy :-)';

  BBPlanos.Enabled := true;
  BBOpcoes.Enabled := true;
  FormOpt.CBShowPlan.Checked := true;

  BBOpcoesClick(Sender);
  BBPlanosClick(Sender);

  while not (FShouldStop) do
  begin
    RunOneSimulation;
  end;
  with Sender as TBitBtn do
    Enabled := True;
end;

end.
