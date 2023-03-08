unit dialog;

{$MODE Delphi}

interface

uses LCLIntf, LCLType, SysUtils, Classes, Graphics, Forms, Controls, StdCtrls,
  Buttons, ExtCtrls;

type
  TResultDlg = class(TForm)
    OKBtn: TButton;
    Bevel1: TBevel;
    Label1: TLabel;
    procedure OKBtnClick(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  ResultDlg: TResultDlg;

implementation

{$R *.lfm}

procedure TResultDlg.OKBtnClick(Sender: TObject);
begin
  Resultdlg.Close;
end;

end.
