program PMinera;

{$MODE Delphi}

uses
  Forms, Interfaces,
  UFRob1 in 'UFRob1.pas' {Form1},
  UForOptMin in 'UForOptMin.pas' {FormOpt},
  UVPlan in 'UVPlan.pas' {FormViewPlans};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TFormOpt, FormOpt);
  Application.CreateForm(TFormViewPlans, FormViewPlans);
  Application.Run;
end.
