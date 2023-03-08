unit uresizeutil;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralnetwork;

function CreateResizingNN(pSizeX, pSizeY: integer):TNNet;

implementation

function CreateResizingNN(pSizeX, pSizeY: integer):TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer( TNNetInput.Create(pSizeX, pSizeY,3) );
  Result.AddLayer( TNNetDeAvgPool.Create(2) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,3,1,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(64,1,0,0) );
  Result.AddLayer( TNNetConvolutionReLU.Create(3,1,0,0) );
end;

end.

