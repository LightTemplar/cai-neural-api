unit UForOptMin;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ComCtrls, StdCtrls, ExtCtrls;

type

  { TFormOpt }

  TFormOpt = class(TForm)
    GroupPopOpt: TGroupBox;
    CHPlan: TCheckBox;
    CHABC: TCheckBox;
    GroupBox1: TGroupBox;
    LabRND: TLabel;
    TBRandom: TTrackBar;
    GroupAmbiente: TGroupBox;
    ChMostra: TCheckBox;
    LabCiclos: TLabel;
    EdCiclos: TEdit;
    CBNovoPlano: TCheckBox;
    CBEliminaIncorreto: TCheckBox;
    GBOtim: TGroupBox;
    TBOptimization: TTrackBar;
    LabOptimization: TLabel;
    CBShowPlan: TCheckBox;
    CBDebug: TCheckBox;
    procedure CHPlanChange(Sender: TObject);
    procedure TBRandomChange(Sender: TObject);
    procedure TBOptimizationChange(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  FormOpt: TFormOpt;

implementation

{$R *.lfm}

procedure TFormOpt.TBRandomChange(Sender: TObject);
begin
LabRND.Caption:=IntToStr(TBRandom.Position)+'%';
end;

procedure TFormOpt.CHPlanChange(Sender: TObject);
begin

end;

procedure TFormOpt.TBOptimizationChange(Sender: TObject);
begin
LabOptimization.Caption:=IntToStr(TBOptimization.Position)+'%';
end;

end.