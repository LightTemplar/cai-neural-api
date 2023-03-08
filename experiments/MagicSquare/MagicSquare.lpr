program MagicSquare;

{$MODE Delphi}

uses
  Forms,
  Interfaces,
  quada6g in 'quada6g.pas' {Form1},
  UDARR in 'UDARR.pas',
  uabout in 'uabout.pas' {AboutBox};

{$R *.res}

begin
  Application.Initialize;
  Application.Title := 'Magic Square Maker';
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TAboutBox, AboutBox);
  Application.Run;
end.
