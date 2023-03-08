unit UVence;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, Buttons;

type

  { TFormVence }

  TFormVence = class(TForm)
    LabJog: TLabel;
    LabVence: TLabel;
    BitBtn1: TBitBtn;
    Bevel1: TBevel;
    procedure Bevel1ChangeBounds(Sender: TObject);
    procedure BitBtn1Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
    procedure Vence(S: string);
  end;

var
  FormVence: TFormVence;

implementation

{$R *.lfm}

procedure TFormVence.Bevel1ChangeBounds(Sender: TObject);
begin

end;

procedure TFormVence.BitBtn1Click(Sender: TObject);
begin
  Close;
end;

procedure TFormVence.Vence(S: string);
begin
  LabJog.Caption := S;
  ShowModal;
  Close;
end;

end.
