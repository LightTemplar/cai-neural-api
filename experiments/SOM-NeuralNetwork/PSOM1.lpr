program PSOM1;

{$MODE Delphi}

uses
  Forms, Interfaces,
  USOM1 in 'USOM1.pas' {Form1},
  URSOM in 'URSOM.pas';

begin
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
