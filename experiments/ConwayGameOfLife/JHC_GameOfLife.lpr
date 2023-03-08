program JHC_GameOfLife;

{$MODE Delphi}

uses
  Forms,
  Interfaces,
  lifeai in 'lifeai.pas' {Form1},
  about in 'about.pas' {AboutBox};

{$R *.res}

begin
  Application.Initialize;
  Application.Title := 'Another J. H. C. Life Game';
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TAboutBox, AboutBox);
  Application.Run;
end.
