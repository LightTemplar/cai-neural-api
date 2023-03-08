unit ueasy_trillion_test_form;
{
 Lazarus / Free Pascal / OpenCL Example
 This code implements a basic parallel evolutionary algorithm that minimizes
 an error function.

 The error function is: abs(x-5000). Evidently, the solution is 5000.

 This code adapts and is inspired on explamples found at:
 fpc\3.0.2\source\packages\opencl\examples

 Users have the option to run either 1 billion or 1 trillion mutations.

 This code has been tested with
 * ATI
 * Intel Graphics
 * NVIDIA

 Count = DATA_SIZE * kernel executions * external opencl loop * number of mutations
 1 Billion  = (12800) * 80 *  100 *   10
 1 Trillion = (12800) * 80 * 1000 * 1000

 The DATA_SIZE above is actually the number of threads.

 The kernel is run 80 times. The output buffer is used as both:
 * output.
 * working memory (threads can get results from other threads).

 Each kernel execution is an implicit global synchronization point.

 This prototype is part of the Conscious Artificial Intelligence project
 located at:
 https://sourceforge.net/projects/cai/
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  neuralopencl, neuralopenclv;

type

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    ButProbe: TButton;
    ComboPlatform: TComboBox;
    ComboDevType: TComboBox;
    LabText2: TLabel;
    LabText1: TLabel;
    LabPlatform: TLabel;
    LabelDevType: TLabel;
    LabKernel: TLabel;
    LabResult: TLabel;
    LabTotal: TLabel;
    RBBillion: TRadioButton;
    RBTrillion: TRadioButton;
    procedure ButProbeClick(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure ComboDevTypeChange(Sender: TObject);
    procedure ComboPlatformChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FOpenCL: TEasyOpenCLCL;
    procedure testOpenCL(trillion:longint);
  public
    { public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.lfm}

uses cl;

procedure TForm1.testOpenCL(trillion:longint);

const
  // Use a static data size for simplicity
  DATA_SIZE = 12800;

type
  single2 = record
    x: single;
    y: single;
  end;

type
  TMyData = single2;

var
  err: integer; // error code returned from api calls
  Data: array [0..DATA_SIZE - 1] of TMyData; // original data set given to device
  results: array [0..DATA_SIZE - 1] of TMyData; // results returned from device

  kernel: cl_kernel;         // compute kernel

  input: cl_mem; // device memory used for the input array
  output: cl_mem; // device memory used for the output array

  i: integer;
  Count: longword;

  ArraySizeBytes: integer;

  startTime, finishTime, totalTimeSeconds: double;
begin
  // Fill our data set with random float values
  Count := DATA_SIZE;

  for i := 0 to Count - 1 do
    Data[i].x := random;

  FOpenCL.CompileProgramFromFile('evolve_easy.cl');

  kernel := FOpenCL.CreateKernel('minimize_error');

  ArraySizeBytes := sizeof(TMyData) * Count;

  input  := FOpenCL.CreateInputBuffer(ArraySizeBytes);
  output := FOpenCL.CreateOutputBuffer(ArraySizeBytes);

  FOpenCL.WriteBuffer(input, ArraySizeBytes, @Data);

// Set the arguments to our compute kernel
// __kernel void minimize_error
//(
//  __global float2* input,
//  __global float2* output,
//  const unsigned int count
//  const int trillion
//)
  err := clSetKernelArg(kernel, 0, sizeof(cl_mem), @input);
  err := err or clSetKernelArg(kernel, 1, sizeof(cl_mem), @output);
  err := err or clSetKernelArg(kernel, 2, sizeof(longword), @Count);
  err := err or clSetKernelArg(kernel, 3, sizeof(longint), @trillion);

  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Failed to set kernel arguments:', err);
    exit;
  end
  else
    writeln('Kernel arguments OK!');

  startTime := now();

  // runs the kernel 80 times with "count" thread count
  for i := 1 to 80 do
  begin
    FOpenCL.RunKernel(kernel,Count);
  end;

  // Wait for the command commands to get serviced before reading back results
  FOpenCL.Finish();

  // Read back the results from the device to verify the output
  err := FOpenCL.ReadBuffer(output, ArraySizeBytes, @results);
  if (err = CL_SUCCESS) then
  begin
    writeln('clEnqueueReadBuffer OK!');
    finishTime := now();

    totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

    writeln('Total run time:', (totalTimeSeconds): 10: 5, ' seconds.');
    Form1.LabResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';

    // Validate our results
    for i := 0 to 29 do
    begin
      writeln('Result at pos i:', i, ' Value: ', results[i].x:10:5, ' Error:', results[i].y:10:5);
    end;
  end
  else
  begin
    Form1.LabResult.Caption := 'ERROR';
  end;

  // Shutdown and cleanup
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseKernel(kernel);
end;

{ TForm1 }
procedure TForm1.Button1Click(Sender: TObject);
begin
  Button1.Enabled := false;
  LabResult.Caption := 'RUNNING - Please Wait';
  Application.ProcessMessages();

  if RBTrillion.Checked then
    testOpenCL(1)
  else
    testOpenCL(0);

  Button1.Enabled := true;
end;

procedure TForm1.ComboDevTypeChange(Sender: TObject);
begin
  FOpenCL.ComboDevTypeChange(ComboDevType);
end;

procedure TForm1.ComboPlatformChange(Sender: TObject);
begin
  FOpenCL.ComboPlatformChange(ComboPlatform, ComboDevType);
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  FOpenCL := TEasyOpenCLCL.Create();
end;

procedure TForm1.FormDestroy(Sender: TObject);
begin
  FOpenCL.Free;
end;

procedure TForm1.ButProbeClick(Sender: TObject);
begin
  FOpenCL.printDevicesInfo();
  if FOpenCL.ProbeClick(ComboPlatform, ComboDevType) then
  begin
    Button1.Enabled := True;
  end;
end;


end.
