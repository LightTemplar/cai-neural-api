unit princ;

{$MODE Delphi}

interface

uses
  LCLIntf, LCLType, SysUtils, Classes, Graphics, Controls, Forms, Dialogs,
  ExtCtrls, ComCtrls, Menus, Buttons, UMOINHO;

type TShapePtr = ^TShape; { # }

type
  tabuleiro_val = array[1..24] of integer;
  tabuleiro_pos = array[0..24] of byte;

  minimax_return = record
                      valor:integer;
                      tabuleiro:tabuleiro_pos;
                   end;

  movimentos_possiveis_pedra = record
                         nro_pedra,
                         total:byte;
                         mov:array[1..24]of byte;
                      end;

  movimentos_possiveis_total = record
                                  total_pedras,
                                  total_mov:byte;
                                  pedra:array[1..9]of movimentos_possiveis_pedra;
                               end;

  TPrincipal = class(TForm)
    Panel1: TPanel;
    SB: TStatusBar;
    Tabuleiro: TImage;
    Pos1: TShape;
    Pos2: TShape;
    Pos3: TShape;
    Pos4: TShape;
    Pos5: TShape;
    Pos6: TShape;
    Pos7: TShape;
    Pos8: TShape;
    Pos9: TShape;
    Pos10: TShape;
    Pos11: TShape;
    Pos12: TShape;
    Pos13: TShape;
    Pos14: TShape;
    Pos15: TShape;
    Pos16: TShape;
    Pos17: TShape;
    Pos18: TShape;
    Pos19: TShape;
    Pos20: TShape;
    Pos21: TShape;
    Pos22: TShape;
    Pos23: TShape;
    Pos24: TShape;
    MainMenu1: TMainMenu;
    Jogo1: TMenuItem;
    Novo1: TMenuItem;
    N1: TMenuItem;
    Sair1: TMenuItem;
    Opcoes1: TMenuItem;
    Pedras1: TMenuItem;
    Oponentes1: TMenuItem;
    N1PX2P1: TMenuItem;
    N1PXCOM1: TMenuItem;
    Dificuldade1: TMenuItem;
    N11: TMenuItem;
    N21: TMenuItem;
    N31: TMenuItem;
    N41: TMenuItem;
    N51: TMenuItem;
    N61: TMenuItem;
    N71: TMenuItem;
    N81: TMenuItem;
    N91: TMenuItem;
    COMX1P1: TMenuItem;
    N42: TMenuItem;
    N52: TMenuItem;
    N62: TMenuItem;
    Cor1: TMenuItem;
    Brano1: TMenuItem;
    Preto1: TMenuItem;

    procedure FormCreate(Sender: TObject);
    procedure Novo1Click(Sender: TObject);
    procedure Pos1MouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure N1PXCOM1Click(Sender: TObject);
    procedure N31Click(Sender: TObject);
    procedure Sair1Click(Sender: TObject);
    procedure N81Click(Sender: TObject);
    procedure Preto1Click(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure COMX1P1Click(Sender: TObject);
    procedure N1PX2P1Click(Sender: TObject);
  private
    { Private declarations }
    STCOMP:boolean;
    MTab:TTabuleiro; {# tabuleiro em memoria}
      vez, (* true se é a vez do jogador 1 e false se do jogador 2*)
  // ------------------------------------------------------------------
  COM,comeco,branco,fim:boolean; (*indica se o jogador 2 é o computador ou não*)
  nro_adv:array[1..2]of byte;
  tempo,profundidade,total_pedras:byte;
  move_j,
  come_j, (*nro de pedras que um jogador tem direito de comer*)
  ini_j, (*nro de pedras que cada jogador comeca o jogo*)
  np_j:array[1..2]of byte; (*nro de peças dos jogadores no momento*)
  //--------------------------------------------------------------------
    procedure altera_shape(var pos:TShape;jogador:byte);
    { Muda os atributos de um determindado shape}
    procedure MostraBarra;
    {apresenta barra de status}
    procedure jogaP(posj,jogador:byte);
    procedure Passa_a_vez;
    procedure Novo_jogo;
    procedure LimpaJogMenuItems;
    {limpa checkeds dos itens de menu}
  public
    { Public declarations }
    Shapes: array [1..24] of TShapePtr; { # ponteiros para shapes }
    procedure altera_pos(pedra,jogador:byte);
    (* Muda os atributos de uma determinada posição do tabuleiro*)
    procedure desenha_tabuleiro;
    procedure vitoria(jogador:byte);

  end;
var
  Principal: TPrincipal;


implementation

uses UIA, UVence;

{$R *.lfm}


(******************************)
(* PROCEDURE DESENHA TABULEIRO*)
(* De acordo com um tabuleiro passado (posicoes das pedras), desenha
 as pedras na tela *)
procedure TPrincipal.desenha_tabuleiro{(tab:tabuleiro_pos)};
var i:integer;
begin
  Application.ProcessMessages;
  Principal.Tabuleiro.repaint;
  for i:=1 to 24 do
       {altera_pos(i,tab[i]);}
       altera_pos(i,MTab.tab[i]);
  MostraBarra;
  Application.ProcessMessages;
end;

(**************************************************************************)
(*PROCEDURE VITÓRIA *)
(* Exibe a mensagem de fim de jogo*)
procedure TPrincipal.Vitoria(jogador:byte);
begin
end;


procedure TPrincipal.MostraBarra;
{apresenta barra de status}
begin
     if vez then SB.Panels[0].text:= 'Next turn: 1P'
            else begin
                 if COM then SB.Panels[0].text:= 'Next turn: COM'
                        else SB.Panels[0].text:= 'Next turn: 2P';
                 end;
     SB.Panels[1].text:= 'Rocks 1P: " '+inttostr(MTab.Pecas[1])+' "';
     if COM
        then SB.Panels[2].text:= 'Rocks COM: " '+inttostr(MTab.Pecas[2])+' "'
        else SB.Panels[2].text:= 'Rocks 2P: " '+inttostr(MTab.Pecas[2])+' "';
end;
(**************************************************************************)
(* PROCEDURE PASSA A VEZ *)
(* Passa a vez de jogar para o adversário *)
procedure TPrincipal.passa_a_vez;
begin
     vez:=not(vez);
     np_j[1]:=Principal.MTab.Pecas[1];
     np_j[2]:=Principal.MTab.Pecas[2];
     MostraBarra;
     Application.ProcessMessages;
end;

(**************************************************************************)
(* PROCEDURE JOGA *)
(* Possibilita ao jogador 1P ou 2P realizar uma jogada *)
procedure TPrincipal.jogaP(posj,jogador:byte);
var adversario:byte;
    AUX:TretEscolhePos;
    AUXJOG:TRetJoga;

    procedure ComePedra(jogador:byte);
    {computador come pedra}
    var AUXCome:integer;
    begin
    AUXCome:=EscolheCome(MTAB,jogador);
    if AUXCome<>0
       then MTab.DeletaPedra(AUXCome);
    end;

   procedure JogaComp;
    {computador escolhe movimento}
    begin
    move_j[jogador]:=0;
    {Vez:=false;}
    passa_a_vez;
    if MTab.Pecas[3-jogador]>3
       then begin
            SetmaxProf(Profundidade);
            AUXJOG:=Joga_B(MTab,3-jogador,Profundidade,Teste4);
            {AUXJOG:=Joga_B(MTab,3-jogador,Profundidade,Teste3);}
  {melhor}  {AUXJOG:=Joga_A(MTab,3-jogador,Profundidade,Teste3);}
            {AUXJOG:=Joga(MTab,3-jogador,5,Teste1)}
            end
       else AUXJOG:=Joga3(MTab,3-jogador);
    MTab.DeslocaPedra(AUXJOG.MelhorIni,AUXJOG.MelhorFim);
    if MTab.AnalisaTrilha(AUXJOG.MelhorFim) > 0
       then ComePedra(jogador);
    passa_a_vez;
    end;

    procedure PoemComp;
    {computador escolhe onde por peca}
    begin
    passa_a_vez;
    AUX:=EscolhePos(Principal.Mtab,3-jogador,0,nil);
    Mtab.PoemPedra(AUX.Pos,3-jogador);
    dec(ini_j[3-jogador]);
    if MTab.AnalisaTrilha(AUX.Pos) > 0
       then ComePedra(jogador);
    passa_a_vez;
    end;

begin
   adversario:=nro_adv[jogador];

   if ini_j[jogador]>0 then  { primeiro if }
   begin
   (* Fase inicial do jogo, onde coloca-se todas as pecas *)
   if (MTab.Tab[posj] = 0) and (come_j[jogador] = 0) and
   (move_j[jogador]=0) then
      begin
         {Principal.altera_pos(posj,jogador);}
         MTab.PoemPedra(Posj,jogador); {#}
         desenha_tabuleiro;
         come_j[jogador]:={analisa_trilha(posj,jogador,tabuleiro_ocup,true);}
                          MTab.AnalisaTrilha(posj);
         dec(ini_j[jogador]);
         inc(np_j[jogador]);
         if (come_j[jogador] = 0) then
            begin
            if (ini_j[3-jogador]>0)
               then PoemComp
               else JogaComp;
            end
         else {testa_fim};
         exit;
      end;
   end
   else
      begin
      (* Seleciona peça para movimentar *)
      if (csJogador[MTab.Tab[posj]]=jogador)
      and (come_j[jogador]=0) and (ini_j[jogador]=0)and(move_j[jogador]=0)
      then
         begin
            {Altera_pos(posj,5);}
            MTab.Tab[posj]:=5;
            move_j[jogador]:=posj;
            exit;
         end;

      (* Desfaz seleção anterior *)
      if Mtab.Tab[posj]=5
      then  begin
            {Altera_pos(posj,jogador);}
            MTab.Tab[posj]:=jogador;
            MTab.AnalisaTrilha(posj);
            move_j[jogador]:=0;
            exit;
            end;

      (* Move peça de um lugar para o outro *)
      if (move_j[jogador]>0) and
      (MTab.MovimentoPossivel(move_j[jogador],posj) or
       ((Mtab.Pecas[jogador]=3) and (MTab.Tab[posj]=peVazio))) then
          begin
           {move_pedra(move_j[jogador],posj,jogador,tabuleiro_ocup,true)};
            MTab.Tab[move_j[jogador]]:=jogador;
            MTab.DeslocaPedra(move_j[jogador],posj);
            come_j[jogador]:=MTab.AnalisaTrilha(posj);
            Desenha_tabuleiro;
            if COM and (come_j[jogador] = 0)
               then JogaComp;{joga_computador;}
            exit;
          end;
      end; { of primeiro if }

      (* Seleciona peça que quer comer *)
      if (csJogador[MTab.Tab[posj]] = adversario) and
         (come_j[jogador] > 0) then
         begin
            MTab.DeletaPedra(posj);
            Desenha_tabuleiro;
            come_j[jogador]:=come_j[jogador]-1;
            if come_j[jogador]=0 then
               begin
               if ini_j[jogador]>0
                  then PoemComp
                  else JogaComp;
               end;
            exit;
         end;


end;

(**************************)
(* PROCEDURE ALTERA SHAPE *)
(* Muda os atributos de uma determinada posição do tabuleiro*)
procedure TPrincipal.altera_shape(var pos:TShape;jogador:byte);
begin
     if not(branco) then
        begin
           if (jogador = 1) or (jogador=3) then jogador:=jogador+1
           else if (jogador = 2) or (jogador=4) then jogador:=jogador-1;
        end;
     case jogador of

  peVazio:begin                            { # limpa}
             with pos do
             begin
               pen.style:=psclear;
               brush.style:=bsclear;
             end;
          end;

 peBranco:begin                            { # branco normal }
             with pos do
             begin
                pen.style:=psSolid;
                pen.color:=clBlack;
                brush.style:=bsSolid;
                brush.color:=clWhite;
             end;
          end;

  pePreto:begin                           { # prento normal }
             with pos do
             begin
                pen.style:=psSolid;
                pen.color:=clBlack;
                brush.style:=bsSolid;
                brush.color:=clBlack;
             end;
          end;

  peBrancoSel:begin                           { # branco selecionado }
             with pos do
             begin
                pen.style:=psSolid;
                pen.color:=clYellow;
                brush.style:=bsSolid;
                brush.color:=clWhite;
             end;
          end;

  pePretoSel:begin                           { # prento selecionado }
             with pos do
             begin
                pen.style:=psSolid;
                pen.color:=clYellow;
                brush.style:=bsSolid;
                brush.color:=clBlack;
             end;
          end;
  peVermelho:pos.pen.color:=clRed;
     end;
end;

(**************************************************************************)
(* PROCEDURE ALTERA POS *)
(* Muda os atributos de uma determinada posição do tabuleiro*)
procedure TPrincipal.altera_pos(pedra,jogador:byte);
begin
altera_shape(Shapes[Pedra]^,jogador);
end;
(**************************************************************************)
procedure TPrincipal.Pos1MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
   posj:byte;
   qq:TShape;
begin

     qq:=TShape(Sender);
     posj:=qq.tag div 10;
//A.Show;
{            A.Mostra(posj,
                     MTab.Ocupacao[posj,csJogador[Mtab.Tab[posj]]].H,
                     MTab.Ocupacao[posj,csJogador[Mtab.Tab[posj]]].V,
                     MTab.Ocupacao[posj,3-csJogador[Mtab.Tab[posj]]].H,
                     MTab.Ocupacao[posj,3-csJogador[Mtab.Tab[posj]]].V); }
     if vez
        then jogaP(posj,1)
        else if not(COM)
             then jogaP(posj,2);
     Desenha_Tabuleiro;

      if (MTab.Pecas[1]<3) and (ini_j[1]=0)then
         begin
         FormVence.Vence('Player 2');
         Novo_Jogo;
         end;

      if (MTab.Pecas[2]<3) and (ini_j[2]=0) then
         begin
         FormVence.Vence('Player 1');
         Novo_Jogo;
         end;

end;
(**************************************************************************)


procedure TPrincipal.FormCreate(Sender: TObject);
begin
MTab.Init;
Shapes[1]:=addr(Pos1);
Shapes[2]:=addr(Pos2);
Shapes[3]:=addr(Pos3);
Shapes[4]:=addr(Pos4);
Shapes[5]:=addr(Pos5);
Shapes[6]:=addr(Pos6);
Shapes[7]:=addr(Pos7);
Shapes[8]:=addr(Pos8);
Shapes[9]:=addr(Pos9);
Shapes[10]:=addr(Pos10);
Shapes[11]:=addr(Pos11);
Shapes[12]:=addr(Pos12);
Shapes[13]:=addr(Pos13);
Shapes[14]:=addr(Pos14);
Shapes[15]:=addr(Pos15);
Shapes[16]:=addr(Pos16);
Shapes[17]:=addr(Pos17);
Shapes[18]:=addr(Pos18);
Shapes[19]:=addr(Pos19);
Shapes[20]:=addr(Pos20);
Shapes[21]:=addr(Pos21);
Shapes[22]:=addr(Pos22);
Shapes[23]:=addr(Pos23);
Shapes[24]:=addr(Pos24);

     Randomize;
     novo_jogo;
     nro_adv[1]:=2;
     nro_adv[2]:=1;
     tempo:=1;

end;

(***********************)
(* PROCEDURE NOVO JOGO *)
(* Realiza todos os procedimentos para a realização de um novo jogo *)
procedure TPrincipal.Novo_jogo;
var i:integer;
begin
     np_j[1]:=0;
     np_j[2]:=0;
     MTab.Limpa;
     if MainMenu1.Items[1][1][0].Checked then
        COM := false else COM:=true;
     if MainMenu1.Items[1][1][1].Checked then
        begin
           vez:=true;
           comeco:=true;
        end;
     if MainMenu1.Items[1][1][2].Checked then
        begin
           vez:=false;
           comeco:=false;
        end;
     vez:=not(vez);
     passa_a_vez;
     i:=0;
     with MainMenu1 do
        begin
             if items[1][0][0].checked then i:=4;
             if items[1][0][1].checked then i:=5;
             if items[1][0][2].checked then i:=6;
             if items[1][0][3].checked then i:=7;
             if items[1][0][4].checked then i:=8;
             if items[1][0][5].checked then i:=9;
        end;
    with MainMenu1 do
        begin
             if items[1][3][0].checked then branco:=true;
             if items[1][3][1].checked then branco:=false;
        end;
     total_pedras:=i;
     ini_j[1]:=i;
     ini_j[2]:=i;
     move_j[1]:=0;
     move_j[2]:=0;
     come_j[1]:=0;
     come_j[2]:=0;
     with MainMenu1 do
        begin
             if items[1][2][0].checked then i:=1;
             if items[1][2][1].checked then i:=3;
             if items[1][2][2].checked then i:=5;
             if items[1][2][3].checked then i:=7;
             if items[1][2][4].checked then i:=9;
             if items[1][2][5].checked then i:=11;
        end;
     profundidade:=i;
     fim:=false;
     if not(vez) and COM then
        begin
        {joga_computador}
        STCOMP:=true;
        MTab.PoemPedra(Melhores[1],2); {#}
        dec(ini_j[2]);
        inc(np_j[2]);
        passa_a_vez;
        end
     else STCOMP:=false;
Desenha_Tabuleiro;
end;

procedure LimpaMenuItem(var M:TMenuItem);
var I:integer;
begin
for I:=0 to M.Items[I].Count-1
    do M.items[I].checked:=false
end;

procedure TPrincipal.N31Click(Sender: TObject);
begin
     with Principal.MainMenu1 do
        begin
             items[1][2][0].checked:=false;
             items[1][2][1].checked:=false;
             items[1][2][2].checked:=false;
             items[1][2][3].checked:=false;
             items[1][2][4].checked:=false;
             items[1][2][5].checked:=false;
        end;
     TMenuItem(Sender).checked:=true;
     profundidade:=strtoint(TMenuItem(Sender).caption);
     novo_jogo;
end;

procedure TPrincipal.Sair1Click(Sender: TObject);
begin
     Principal.close;
end;

procedure TPrincipal.N81Click(Sender: TObject);
begin
    with Principal.MainMenu1 do
        begin
             items[1][0][0].checked:=false;
             items[1][0][1].checked:=false;
             items[1][0][2].checked:=false;
             items[1][0][3].checked:=false;
             items[1][0][4].checked:=false;
             items[1][0][5].checked:=false;
        end;
     TMenuItem(Sender).checked:=true;
     ini_j[1]:=strtoint(TMenuItem(Sender).caption);
     ini_j[2]:=ini_j[1];
     novo_jogo;
end;
procedure TPrincipal.Novo1Click(Sender: TObject);
begin
   novo_jogo;
end;

procedure TPrincipal.Preto1Click(Sender: TObject);
begin
    with Principal.MainMenu1 do
        begin
             items[1][3][0].checked:=false;
             items[1][3][1].checked:=false;
        end;
     TMenuItem(Sender).checked:=true;
     if TMenuItem(Sender).caption = 'Branco' then
        branco:=true
     else branco:=false;
     novo_jogo;
end;


procedure TPrincipal.FormDestroy(Sender: TObject);
begin
MTab.Done;
end;

procedure TPrincipal.LimpaJogMenuItems;
{limpa checkeds dos itens de menu}
var I:integer;
begin
for I:=0 to MainMenu1.items[1][1].Count-1 do
    begin
    MainMenu1.items[1][1][I].checked:=false;
    end;
end;

procedure TPrincipal.N1PXCOM1Click(Sender: TObject);
begin
LimpaJogMenuItems;
TMenuItem(Sender).checked:=true;
  vez:=true;
  COM:=true;
novo_jogo;
end;

procedure TPrincipal.COMX1P1Click(Sender: TObject);
begin
LimpaJogMenuItems;
TMenuItem(Sender).checked:=true;
  vez:=false;
  COM:=true;
novo_jogo;
Desenha_Tabuleiro;

end;

procedure TPrincipal.N1PX2P1Click(Sender: TObject);
begin
LimpaJogMenuItems;
TMenuItem(Sender).checked:=true;
  vez:=true;
  COM:=false;
novo_jogo;
end;

end.