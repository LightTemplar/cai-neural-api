unit USOM1;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, StdCtrls, URSOM, Buttons, ComCtrls;

type
  TForm1 = class(TForm)
    GroupBox1: TGroupBox;
    Image1: TImage;
    GroupBox2: TGroupBox;
    Timer1: TTimer;
    RBQuadrado: TRadioButton;
    RBCirculo: TRadioButton;
    RBTriangulo: TRadioButton;
    PBFatorDeCorrecao: TProgressBar;
    procedure BitBtn1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure RBQuadradoClick(Sender: TObject);
  private
    { Private declarations }
    SOM: TNArray;
    procedure Line(X1, Y1, X2, Y2: extended);
    procedure Desenha;
    procedure TreinaQuadrado;
  public
    { Public declarations }
  end;

var
  Form1: TForm1;


implementation

uses UVect;

{$R *.lfm}
procedure TForm1.Line(X1, Y1, X2, Y2: extended);
begin
  Image1.Canvas.MoveTo(round(X1 * Image1.Width), round(Y1 * Image1.Height));
  Image1.Canvas.LineTo(round(X2 * Image1.Width), round(Y2 * Image1.Height));
end;

procedure TForm1.BitBtn1Click(Sender: TObject);
begin
(*Line(Random(Image1.Width),Random(Image1.Height),
     Random(Image1.Width),Random(Image1.Height));*)
  TreinaQuadrado;
  Image1.Canvas.FillRect(Rect(0, 0, Image1.Width, Image1.Height));
  //Image1.Canvas.StretchDraw(Image1.Canvas.ARect, Image1.Picture.Bitmap);
  Desenha;
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  SOM.FatorDeCorrecao := 1;
  SOM.MakeRandomWs;
end;

procedure TForm1.Desenha;
var
  I, J: integer;
begin
  for I := 1 to MaxNeuron - 1 do
    for J := 1 to MaxNeuron - 1 do
    begin
      Line(SOM.W[I, J, 1], SOM.W[I, J, 2],
        SOM.W[I, J + 1, 1], SOM.W[I, J + 1, 2]);
      Line(SOM.W[I, J, 1], SOM.W[I, J, 2],
        SOM.W[I + 1, J, 1], SOM.W[I + 1, J, 2]);
    end;
  for J := 1 to MaxNeuron - 1 do
    Line(SOM.W[MaxNeuron, J, 1], SOM.W[MaxNeuron, J, 2],
      SOM.W[MaxNeuron, J + 1, 1], SOM.W[MaxNeuron, J + 1, 2]);
  for I := 1 to MaxNeuron - 1 do
    Line(SOM.W[I, MaxNeuron, 1], SOM.W[I, MaxNeuron, 2],
      SOM.W[I + 1, MaxNeuron, 1], SOM.W[I + 1, MaxNeuron, 2]);

end;

procedure TForm1.TreinaQuadrado;
var
  I: longint;
  A, M: extended;
  X: TEntrada;
  Origem, Vetor: TEntrada;
  b: byte;
begin
  for I := 1 to 80 do
  begin
    if RBQuadrado.Checked then
    begin
      X[1] := random(Image1.Width) / Image1.Width;
      X[2] := random(Image1.Height) / Image1.Height;
    end;
    if RBCirculo.Checked then
    begin
      A := random(10000) / 1000;
      //M:=random(1000)/10000;
      M := 1;
      X[1] := ((Sin(A) * M) + 1) / (2);
      X[2] := ((Cos(A) * M) + 1) / (2);
    end;
    if RBTriangulo.Checked then
    begin
      b := random(3);
      case b of
        0:
        begin
          CarregaVetor(Origem, [0.1, 0.1]);
          CarregaVetor(Vetor, [0.8, 0]);
        end;
        1:
        begin
          CarregaVetor(Origem, [0.1, 0.1]);
          CarregaVetor(Vetor, [0.4, 0.9]);
        end;
        2:
        begin
          CarregaVetor(Origem, [0.5, 0.9]);
          CarregaVetor(Vetor, [0.4, -0.8]);
        end;
      end; // of case
      M := random(10000) / 10000;
      ProdutoPorEscalar(Vetor, M);
      SomaVetorial(Origem, Vetor);
      X := Origem;
    end; // of if
    SOM.Treina(X[1], X[2]);
  end;// of FOR
  SOM.FatorDeCorrecao := SOM.FatorDeCorrecao * 0.99;
  PBFatorDeCorrecao.Position := Round(SOM.FatorDeCorrecao * 100);
end; // of procedure;


procedure TForm1.RBQuadradoClick(Sender: TObject);
begin
  SOM.FatorDeCorrecao := 1;
  SOM.MakeRandomWs;
end;

end.
