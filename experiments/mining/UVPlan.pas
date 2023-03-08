unit UVPlan;
//Unit View Plans
//This unit allows plans to be viewed.

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, neuralplanbuilder;

type
  TFormViewPlans = class(TForm)
    Image1: TImage;
    Shape1: TShape;
    Shape2: TShape;
    Shape3: TShape;
    Shape4: TShape;
    Shape5: TShape;
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  S:array[0..4] of TShape;
  procedure Line(X1,Y1,X2,Y2:extended;cor:TColor);
  procedure Clear;
  procedure DefinePos(X1,Y1:extended;P:byte);
  procedure ShowPlan(var P:TPlan; L:extended);
  procedure ShowPlanActionStateList(var V:TActionStateList; L:extended);
  end;

var
  FormViewPlans: TFormViewPlans;

implementation

procedure TFormViewPlans.Line(X1,Y1,X2,Y2:extended;cor:TColor);
begin
Image1.Canvas.Pen.Color := cor;
Image1.Canvas.MoveTo(round(X1*(Image1.Width-10)),round(Y1*(Image1.Height-10)));
Image1.Canvas.LineTo(round(X2*(Image1.Width-10)),round(Y2*(Image1.Height-10)));
end;

procedure TFormViewPlans.Clear;
begin
Image1.Canvas.Brush.Color := clwhite;
Image1.Canvas.FillRect(Rect(0,0,Image1.Width,Image1.Height));
end;

procedure TFormViewPlans.DefinePos(X1,Y1:extended;P:byte);
begin
S[P].Left:=round(X1*(Image1.Width-10));
S[P].Top:=round(Y1*(Image1.Height-10));
end;

procedure TFormViewPlans.ShowPlanActionStateList(var V:TActionStateList; L:extended);
var R:longint;
    X1,Y1,X2,Y2:extended;
begin
               for R:=0 to V.NumStates-2
                   do begin
                      X1:=(V.ListStates[R][1]-1)/L;
                      Y1:=(V.ListStates[R][2]-1)/L;
                      X2:=(V.ListStates[R+1][1]-1)/L;
                      Y2:=(V.ListStates[R+1][2]-1)/L;
                      if V.ListStates[R][3]=0
                         then Line(X1,Y1,X2,Y2,clBlack)
                         else Line(X1,Y1,X2,Y2,clRed)
                      end;
end;

procedure TFormViewPlans.ShowPlan(var P:TPlan; L:extended);
begin
ShowPlanActionStateList(P.FPlan,L);
end;

{$R *.lfm}

procedure TFormViewPlans.FormCreate(Sender: TObject);
begin
S[0]:=Shape1;
S[1]:=Shape2;
S[2]:=Shape3;
S[3]:=Shape4;
S[4]:=Shape5;
end;

end.
