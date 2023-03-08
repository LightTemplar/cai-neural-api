unit uopencl_trillion_test;
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
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls;

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
  private
    { private declarations }
  public
    { public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.lfm}

uses ctypes, cl;

type
  TPlatformNamesArr = array of string;

function getPlatformsStrArray(): TPlatformNamesArr;
var
  err: integer; // error code returned from api calls
  platformids: pcl_platform_id;
  platforms: cl_uint;
  i: integer;
  buf: array[0..99999] of char;
  bufwritten: csize_t;
begin
  err := clGetPlatformIDs(0, nil, @platforms);
  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Cannot get number of platforms!');
    exit;
  end;
  getmem(platformids, platforms * sizeof(cl_platform_id));
  err := clGetPlatformIDs(platforms, platformids, nil);
  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Cannot get platforms!');
    freemem(platformids);
    exit;
  end;

  SetLength(Result, platforms);
  writeln(platforms, ' platform(s) found');
  for i := 0 to platforms - 1 do
  begin
    err := clGetPlatformInfo(platformids[i], CL_PLATFORM_NAME, sizeof(buf), @buf, bufwritten);
    Result[i] := buf;
  end;
  freemem(platformids);
end;

procedure printDevicesInfo();
const
  platform_str_info: array[1..5] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: CL_PLATFORM_PROFILE; Name: 'PROFILE'),
    (id: CL_PLATFORM_VERSION; Name: 'VERSION'),
    (id: CL_PLATFORM_NAME; Name: 'NAME'),
    (id: CL_PLATFORM_VENDOR; Name: 'VENDOR'),
    (id: CL_PLATFORM_EXTENSIONS; Name: 'EXTENSIONS')
    );

  device_str_info: array[1..5] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: CL_DEVICE_NAME; Name: 'DEVICE NAME'),
    (id: CL_DEVICE_VENDOR; Name: 'DEVICE VENDOR'),
    (id: CL_DEVICE_VERSION; Name: 'DEVICE VERSION'),
    (id: CL_DEVICE_PROFILE; Name: 'DEVICE PROFILE'),
    (id: CL_DEVICE_EXTENSIONS; Name: 'DEVICE EXTENSIONS')
    );

  device_word_info: array[1..8] of record
      id: dword;
      Name: PChar
    end
  =
    (
    (id: CL_DEVICE_TYPE_INFO; Name: 'DEVICE TYPE'),
    (id: CL_DEVICE_MAX_WORK_GROUP_SIZE; Name: 'DEVICE MAX WORK GROUP SIZE'),
    (id: CL_DEVICE_MAX_COMPUTE_UNITS; Name: 'DEVICE MAX COMPUTE UNITS'),
    (id: CL_DEVICE_IMAGE3D_MAX_WIDTH; Name: 'DEVICE IMAGE3D MAX WIDTH'),
    (id: CL_DEVICE_IMAGE3D_MAX_HEIGHT; Name: 'DEVICE IMAGE3D MAX HEIGHT'),
    (id: CL_DEVICE_GLOBAL_MEM_SIZE; Name: 'DEVICE GLOBAL MEM SIZE'),
    (id: CL_DEVICE_LOCAL_MEM_SIZE; Name: 'DEVICE LOCAL MEM SIZE'),
    (id: CL_DEVICE_COMPILER_AVAILABLE; Name: 'DEVICE COMPILER AVAILABLE')
    );

var
  err: integer; // error code returned from api calls
  platformids: pcl_platform_id;
  platforms: cl_uint;
  devices: cl_uint;
  deviceids: pcl_device_id;
  i, j, k: integer;
  buf: array[0..99999] of char;
  bufwritten: csize_t;
begin
  err := clGetPlatformIDs(0, nil, @platforms);
  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Cannot get number of platforms!');
    exit;
  end;
  getmem(platformids, platforms * sizeof(cl_platform_id));
  err := clGetPlatformIDs(platforms, platformids, nil);
  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Cannot get platforms!');
    freemem(platformids);
    exit;
  end;
  writeln(platforms, ' platform(s) found');
  for i := 0 to platforms - 1 do
  begin
    writeln('Platform info: ', i, ' ----------------------------------------------');
    for k := low(device_str_info) to high(device_str_info) do
    begin
      err := clGetPlatformInfo(platformids[i], platform_str_info[k].id,
        sizeof(buf), @buf, bufwritten);
      writeln(platform_str_info[k].Name, ': ', buf);
    end;

    err := clGetDeviceIDs(platformids[i], CL_DEVICE_TYPE_ALL, 0, nil, @devices);
    if (err <> CL_SUCCESS) then
    begin
      writeln('ERROR: Cannot get number of devices for platform.');
      break;
    end;
    writeln(devices, ' device(s) found');
    getmem(deviceids, devices * sizeof(cl_device_id));
    err := clGetDeviceIDs(platformids[i], CL_DEVICE_TYPE_ALL, devices, deviceids, nil);
    for j := 0 to devices - 1 do
    begin
      writeln('Device info: ', j, ' ------------');
      for k := low(device_str_info) to high(device_str_info) do
      begin
        err := clGetDeviceInfo(deviceids[j], device_str_info[k].id,
          sizeof(buf), @buf, bufwritten);
        writeln(device_str_info[k].Name, ': ', buf);
      end;

      for k := low(device_word_info) to high(device_word_info) do
      begin
        err := clGetDeviceInfo(deviceids[j], device_word_info[k].id,
          sizeof(buf), @buf, bufwritten);
        writeln(device_word_info[k].Name, ': ', pdword(@buf)^);
      end;
    end;
  end;
  freemem(platformids);
end;

procedure testOpenCL(filename: string; selectedPlatformID: integer; gpu: cl_device_type);

const
  // Use a static data size for simplicity
  DATA_SIZE = 12800;

  compilerOptins: PChar = '-cl-opt-disable';

var
  KernelSource, errorlog: PChar;

  errorlogstr: string[255];
  loglen: csize_t;

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
  correct: longword; // number of correct results returned

  global: csize_t; // global domain size for our calculation
  local: csize_t; // local domain size for our calculation

  device_id: cl_device_id;      // compute device id
  context: cl_context;        // compute context
  commands: cl_command_queue;  // compute command queue
  prog: cl_program;        // compute program
  kernel: cl_kernel;         // compute kernel

  input: cl_mem; // device memory used for the input array
  output: cl_mem; // device memory used for the output array

  i: integer;
  Count: integer;

  platformids: Pcl_platform_id;
  num_platforms: cl_uint;

  loadedOpenClCode: TStrings;

  startTime, finishTime, totalTimeSeconds: double;
begin
  // Fill our data set with random float values
  Count := DATA_SIZE;

  for i := 0 to Count - 1 do
    Data[i].x := random;

  loadedOpenClCode := TStringList.Create();
  loadedOpenClCode.LoadFromFile(filename);

  KernelSource := loadedOpenClCode.GetText;

  err := clGetPlatformIDs(0, nil, @num_platforms);
  Writeln('Number of platforms: ', num_platforms);
  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Cannot get number of platforms!');
    exit;
  end;

  getmem(platformids, num_platforms * sizeof(cl_platform_id));

  err := clGetPlatformIDs(num_platforms, platformids, nil);

  if (err <> CL_SUCCESS) then
  begin
    Writeln('Error: Failed to get platforms IDs');
    exit;
  end;

  device_id := nil;
  err := clGetDeviceIDs(platformids[selectedPlatformID], gpu, 1, @device_id, nil);
  if (err <> CL_SUCCESS) then
  begin
    Writeln('Error: Failed to create a device group:', err);
    exit;
  end
  else
    writeln('clGetDeviceIDs OK!');

  // Create a compute context
  context := clCreateContext(nil, 1, @device_id, nil, nil, err);

  if context = nil then
  begin
    Writeln('Error: Failed to create a compute context:', err);
    exit;
  end
  else
    writeln('clCreateContext OK!');

  // Create a command commands
  commands := clCreateCommandQueue(context, device_id, 0, err);
  if commands = nil then
  begin
    Writeln('Error: Failed to create a command commands:', err);
    exit;
  end
  else
    writeln('clCreateCommandQueue OK!');

  // Create the compute program from the source buffer
  prog := clCreateProgramWithSource(context, 1, PPChar(@KernelSource), nil, err);
  if prog = nil then
  begin
    writeln(KernelSource);
    writeln('Error: Failed to create compute program:', err);
    exit;
  end
  else
    writeln('clCreateProgramWithSource OK!');

  // Build the program executable
  err := clBuildProgram(prog, 0, nil, compilerOptins, nil, nil);

  if (err <> CL_SUCCESS) then
  begin
    errorlog := @errorlogstr[1];
    loglen := 255;
    clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, 255, errorlog, loglen);
    writeln('Error: Failed to build program executable:', err);
    writeln(errorlog);
    exit;
  end
  else
    writeln('clBuildProgram OK!');

  // Create the compute kernel in the program we wish to run
  kernel := clCreateKernel(prog, 'minimize_error', err);
  if (kernel = nil) or (err <> CL_SUCCESS) then
  begin
    writeln('Error: Failed to create compute kernel!');
    exit;
  end
  else
    writeln('clCreateKernel OK!');

  // Create the input and output arrays in device memory for our calculation
  input := clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(TMyData) *
    Count, nil, err);
  writeln('clCreateBuffer - input buffer:', err, ' Size:', sizeof(TMyData) *
    Count, ' bytes');
  output := clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(TMyData) *
    Count, nil, err);
  writeln('clCreateBuffer - output buffer:', err, ' Size:', sizeof(TMyData) *
    Count, ' bytes');
  if (input = nil) or (output = nil) then
  begin
    writeln('Error: Failed to allocate device memory!');
    exit;
  end
  else
    writeln('clCreateBuffer OK!');

  // Write our data set into the input array in device memory
  err := clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(TMyData) *
    Count, @Data, 0, nil, nil);

  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Failed to write to source array:', err);
    exit;
  end
  else
    writeln('clEnqueueWriteBuffer OK!');

  // Set the arguments to our compute kernel
(*
__kernel void minimize_error
(
  __global float2* input,
  __global float2* output,
  const unsigned int count
)
*)
  err := 0;
  err := clSetKernelArg(kernel, 0, sizeof(cl_mem), @input);
  //writeln('clSetKernelArg input ', err);
  err := err or clSetKernelArg(kernel, 1, sizeof(cl_mem), @output);
  //writeln('clSetKernelArg output ', err);
  err := err or clSetKernelArg(kernel, 2, sizeof(longword), @Count);
  //writeln('clSetKernelArg ', err);

  if (err <> CL_SUCCESS) then
  begin
    writeln('Error: Failed to set kernel arguments:', err);
    exit;
  end
  else
    writeln('Kernel arguments OK!');

  startTime := now();

  global := Count;

  // runs the kernel 80 times
  for i := 1 to 80 do
  begin
    // Execute the kernel over the entire range of our 1d input data set
    // let the API to decide the work group size (local is nil)
    err := clEnqueueNDRangeKernel(commands, kernel, 1, nil, @global, nil, 0, nil, nil);
  end;

  if (err <> 0) then
  begin
    if (err = CL_INVALID_WORK_GROUP_SIZE) then
      writeln('ERROR: Invalid work group size.');
    writeln('Error: Failed to execute kernel. Error:', err);
  end
  else
    writeln('clEnqueueNDRangeKernel OK!', ' local: ', local, ' global: ',
      global, ' count:', Count);

  // Wait for the command commands to get serviced before reading back results
  err := clFinish(commands);

  if err = 0 then
    writeln('clFinish OK!')
  else
    writeln('Error at clFinish:', err);

  if err = CL_INVALID_COMMAND_QUEUE then
    writeln('ERROR while running OpenCL code.');

  // Read back the results from the device to verify the output
  err := clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(TMyData) *
    Count, @results, 0, nil, nil);
  if (err <> CL_SUCCESS) then
  begin
    if (err = CL_OUT_OF_RESOURCES) then
      writeln('ERROR: Out of computing resources - probably out of memory.');
    writeln('Error: Failed to read output array! ', err);
  end
  else
  begin
    writeln('clEnqueueReadBuffer OK!');
    finishTime := now();

    totalTimeSeconds := (finishTime - startTime) * 24 * 60 * 60;

    writeln('Total run time:', (totalTimeSeconds): 10: 5, ' seconds.');
    Form1.LabResult.Caption := FloatToStr(round(totalTimeSeconds * 100) / 100) + ' seconds.';

    // Validate our results
    correct := 0;
    for i := 0 to 29 do
    begin
      writeln('Result at pos i:', i, ' Value: ', results[i].x:10:5, ' Error:', results[i].y:10:5);
    end;
  end;

  // Shutdown and cleanup
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(prog);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
end;

{ TForm1 }

procedure TForm1.Button1Click(Sender: TObject);
var
  DeviceType: cl_device_type;
begin
  Button1.Enabled := false;
  LabResult.Caption := 'RUNNING - Please Wait';
  Application.ProcessMessages();
  if ComboDevType.ItemIndex < 4 then
    DeviceType := 1 shl ComboDevType.ItemIndex
  else
    DeviceType := CL_DEVICE_TYPE_ALL;

  if RBBillion.Checked then
    testOpenCL('evolve_billion.cl', ComboPlatform.ItemIndex, DeviceType)
  else
    testOpenCL('evolve_trillion.cl', ComboPlatform.ItemIndex, DeviceType);

  Button1.Enabled := true;
end;

procedure TForm1.ButProbeClick(Sender: TObject);
var
  Platforms: TPlatformNamesArr;
  i: integer;
begin
  printDevicesInfo();
  ComboPlatform.Items.Clear();
  Platforms := getPlatformsStrArray();
  if Length(Platforms) > 0 then
  begin
    for i := low(Platforms) to high(Platforms) do
    begin
      ComboPlatform.Items.Add(Platforms[i]);
    end;
    ComboPlatform.ItemIndex := 0;

    ComboPlatform.Enabled := True;
    ComboDevType.Enabled := True;
    Button1.Enabled := True;
  end;
end;


end.
