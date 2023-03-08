unit usimpleneuralwebserver;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  Buttons, blcksock, sockets, Synautil, neuralnetwork, neuralvolume;

type

  { TFormNeuralWebServer }

  TFormNeuralWebServer = class(TForm)
    BitClose: TBitBtn;
    BitStop: TBitBtn;
    BitBtnStart: TBitBtn;
    ButLoad: TButton;
    CheckLogs: TCheckBox;
    EdBind: TEdit;
    EdPort: TEdit;
    Label1: TLabel;
    LabBind: TLabel;
    OpenDialogNN: TOpenDialog;
    procedure BitBtnStartClick(Sender: TObject);
    procedure BitCloseClick(Sender: TObject);
    procedure BitStopClick(Sender: TObject);
    procedure ButLoadClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private
    { private declarations }
    FListenerSocket, FConnectionSocket: TTCPBlockSocket;
    FWantQuit: boolean;
    FCanClose: boolean;
    FNN: TNNet;
    FInputV, FOutputV: TNNetVolume;

    procedure OutputData(ASocket: TTCPBlockSocket; var OutputDataString: string);
    procedure AttendConnection(ASocket: TTCPBlockSocket);
  public
    { public declarations }
  end;

var
  FormNeuralWebServer: TFormNeuralWebServer;

implementation

{$R *.lfm}

{ TFormNeuralWebServer }

procedure TFormNeuralWebServer.FormCreate(Sender: TObject);
begin
  FListenerSocket := TTCPBlockSocket.Create;
  FConnectionSocket := TTCPBlockSocket.Create;
  FListenerSocket.CreateSocket;
  FListenerSocket.setLinger(true,10);
  FNN := TNNet.Create;
  FInputV := TNNetVolume.Create;
  FOutputV := TNNetVolume.Create;

  FWantQuit := false;
  FCanClose := true;
end;

procedure TFormNeuralWebServer.OutputData(ASocket: TTCPBlockSocket;
  var OutputDataString: string);
begin
  // Write the headers back to the client
  ASocket.SendString('HTTP/1.0 200' + CRLF);
  ASocket.SendString('Content-type: Text/Html' + CRLF);
  ASocket.SendString('Content-length: ' + IntTostr(Length(OutputDataString)) + CRLF);
  ASocket.SendString('Connection: close' + CRLF);
  ASocket.SendString('Date: ' + Rfc822DateTime(now) + CRLF);
  ASocket.SendString('Server: Servidor inspirado no servidor do Felipe usando Synapse' + CRLF);
  ASocket.SendString('' + CRLF);

  // Write the document back to the browser
  ASocket.SendString(OutputDataString);
end;

procedure TFormNeuralWebServer.AttendConnection(ASocket: TTCPBlockSocket);
var
  timeout: integer;
  s: string;
  method, uri, protocol: string;
  OutputDataString: string;
  uriS: TStringList;
  uriCnt: integer;
  pcmd, pdata: string;
begin
  timeout := 120000;

  //read request line
  s := ASocket.RecvString(timeout);
  if CheckLogs.Checked then WriteLn(s);
  method := fetch(s, ' ');
  uri := fetch(s, ' ');
  protocol := fetch(s, ' ');

  //read request headers
  repeat
    s := ASocket.RecvString(Timeout);
    if CheckLogs.Checked then WriteLn(s);
  until s = '';

  if CheckLogs.Checked then WriteLn('uri:',uri);

  uriS := CreateTokenizedStringList(uri, '/');

  // Now write the document to the output stream
  if uriS.Count > 2 then
  begin
    if CheckLogs.Checked then
    begin
      for uriCnt := 0 to uriS.Count - 1 do
      begin
        WriteLn(uriCnt,':',uriS[uriCnt]);
      end;
    end;

    pcmd := uriS[1];
    pdata := uriS[2];

    if pcmd = 'compute' then
    begin
      FInputV.LoadFromString(pdata);
      FNN.Compute(FInputV);
      FNN.GetOutput(FOutputV);
      OutputDataString := FOutputV.SaveToString();
    end
    else if pcmd = 'backprop' then
    begin
      FInputV.LoadFromString(pdata);
      FNN.Backpropagate(FInputV);
      OutputDataString := '1';
    end
    else
    begin
      OutputDataString := 'no recognized command:' + pcmd;
    end;

    if CheckLogs.Checked then WriteLn(OutputDataString);
    // Write the output document to the stream
    // OutputDataString += CRLF;
    OutputData(ASocket, OutputDataString);
  end
  else if uri='/' then
  begin
    OutputDataString :=
      'Neural Server is <b>alive</b>. <br/><br/> '+
      'compute/volume_data runs neural network. <br/>'+
      'backprop/volume_data runs backpropagation. <br/><br/>'+
      'Encode your input data with TVolume.SaveToString. <br/> Example '+
      ' http://127.0.0.1:64001/compute/1;2;1;1;0.3;0.4 <br/><br/>'+
      'Decode output data with TVolume.LoadFromString. <br/> Example '+
      ' http://127.0.0.1:64001/backprop/1;2;1;1;0.8;0.2 <br/>';
    OutputData(ASocket, OutputDataString);
  end
  else
  begin
    ASocket.SendString('HTTP/1.0 404' + CRLF);
  end;

  uriS.Free;
  if CheckLogs.Checked then WriteLn('Finished.');
end;

procedure TFormNeuralWebServer.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FWantQuit := true;
  while not(FCanClose) do Application.ProcessMessages();
  FListenerSocket.Free;
  FConnectionSocket.Free;
  FInputV.Free;
  FOutputV.Free;
  FNN.Free;
end;

procedure TFormNeuralWebServer.BitBtnStartClick(Sender: TObject);
begin
  FCanClose := false;
  FWantQuit := false;
  BitBtnStart.Enabled := false;
  BitStop.Enabled := true;
  BitClose.Enabled := false;

  FListenerSocket.bind(EdBind.Text,EdPort.Text);
  FListenerSocket.listen;

  while not(FWantQuit) do
  begin
    Application.ProcessMessages();
    if FListenerSocket.canread(100) then
    begin
      FConnectionSocket.Socket := FListenerSocket.accept;
      //WriteLn('Attending Connection. Error code (0=Success): ', FConnectionSocket.lasterror);
      AttendConnection(FConnectionSocket);
      FConnectionSocket.CloseSocket;
    end;
  end;
  FCanClose := true;
  BitBtnStart.Enabled := true;
  BitStop.Enabled := false;
  BitClose.Enabled := true;
end;

procedure TFormNeuralWebServer.BitCloseClick(Sender: TObject);
begin
  Close;
end;

procedure TFormNeuralWebServer.BitStopClick(Sender: TObject);
begin
  FWantQuit := true;
  BitStop.Enabled := false;
  BitClose.Enabled := true;
end;

procedure TFormNeuralWebServer.ButLoadClick(Sender: TObject);
begin
  if OpenDialogNN.Execute then
  begin
    FNN.LoadFromFile(OpenDialogNN.FileName);
    BitBtnStart.Enabled := true;
  end;
end;

end.

