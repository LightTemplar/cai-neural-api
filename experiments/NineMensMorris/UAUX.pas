unit UAUX;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  StdCtrls;

type
  TA = class(TForm)
    VH: TLabel;
    NVH: TLabel;
    VV: TLabel;
    NVV: TLabel;
    VH2: TLabel;
    VV2: TLabel;
    NVH2: TLabel;
    NVV2: TLabel;
    Pos: TLabel;
  private
    { Private declarations }
  public
    { Public declarations }
    procedure Mostra(P, H, V, H2, V2: integer);
  end;

var
  A: TA;

implementation

{$R *.lfm}

procedure TA.Mostra(P, H, V, H2, V2: integer);
begin
  Pos.Caption := IntToStr(P);

  NVH.Caption := IntToStr(H);
  NVV.Caption := IntToStr(V);
  NVH2.Caption := IntToStr(H2);
  NVV2.Caption := IntToStr(V2);
end;

end.
