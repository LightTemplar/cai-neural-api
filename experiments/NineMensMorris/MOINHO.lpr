program Moinho;

{$MODE Delphi}

uses
  Forms, Interfaces,
  princ in 'princ.pas' {Principal},
  dialog in 'dialog.pas' {ResultDlg},
  UMoinho in 'UMoinho.pas',
  UAUX in 'UAUX.pas' {A},
  UIA in 'UIA.pas',
  UVence in 'UVence.pas' {FormVence};

{$R *.res}

begin
  Application.Initialize;
  Application.Title := 'Game';
  Application.CreateForm(TPrincipal, Principal);
  Application.CreateForm(TResultDlg, ResultDlg);
  Application.CreateForm(TA, A);
  Application.CreateForm(TFormVence, FormVence);
  Application.Run;
end.
