unit UDARR;

interface
{-----------------------dimensionable arrays---------------------------------}
const TDMAXARR = 100000000;
type TDArrayCelType = word; { tipo da celula do array }
type TDArrayType = array [0..TDMAXARR] of TDArrayCelType;
type TDArrayTypePtr = ^TDArrayType;
type TDArray = object
     private
     M:TDArrayTypePtr;
     AMEM:longint;{Allocated Memory}
     TAM:longint;{Tamanho do array}
     Inicio:longint;{Inicio do array}
     CelSize:longint;{ Cell Size }
     procedure SetData(Index: longint;Value: TDArrayCelType);
     function GetData(Index: longint): TDArrayCelType;
     public
     property E[Index: longint]: TDArrayCelType read GetData write SetData; default;
     procedure Dim(PInicio,PFIM,PCelSize:longint);{dimensiona}
     procedure Free; {desaloca}
     end;

implementation
procedure TDArray.SetData(Index: longint;Value:TDArrayCelType);
begin
    M^[Index+Inicio]:=Value;
end;

function TDArray.GetData(Index: longint): TDArrayCelType;
begin
GetData:=M^[Index+Inicio];
end;

procedure TDArray.Dim(PInicio,PFIM,PCelSize:longint);{dimensiona}
begin
TAM:=succ(PFIM-PInicio);
Inicio:=PInicio;
AMEM:=TAM*CelSize;
GetMem(M,AMEM);
end;

procedure TDArray.Free; {desaloca}
begin
FreeMem(M,AMEM);
end;

end.
