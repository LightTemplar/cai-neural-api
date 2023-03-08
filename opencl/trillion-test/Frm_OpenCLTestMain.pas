unit Frm_OpenCLTestMain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls;

type

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    LabResult: TLabel;
    LabTotal: TLabel;
    OpenDialog1: TOpenDialog;
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

// Use a static data size for simplicity
uses ctypes, cl;

resourcestring
  rsClGetDeviceI = 'clGetDeviceIDs %d';
  rsClGetPlatformID = 'clGetPlatformIDs %d';
  rsErrFailedToPlatfor = 'Error: Failed to platforms!';
  rsErrCannotGetNumOfPl = 'Error: Cannot get number of platforms!';
  rsErrorFailedToAllocateDeviceMemory = 'Error: Failed to allocate device '
    +'memory!';
  rsErrorFailedToBuildProgramExecutable = 'Error: Failed to build program '
    +'executable!';
  rsErrorFailedToCreateACommandCommands = 'Error: Failed to create a command '
    +'commands!';
  rsErrorFailedToCreateAComputeContext = 'Error: Failed to create a compute '
    +'context!';
  rsErrorFailedToCreateADeviceGroup = 'Error: Failed to create a device group!';
  rsErrorFailedToCreateComputeKernel =
    'Error: Failed to create compute kernel!';
  rsErrorFailedToCreateComputeProgram = 'Error: Failed to create compute '
    +'program! ';
  rsErrorFailedToExecuteKernel = 'Error: Failed to execute kernel!';
  rsErrorFailedToReadOutputArray = 'Error: Failed to read output array! ';
  rsErrorFailedToRetrieveKernelWorkGroupInfo = 'Error: Failed to retrieve '
    +'kernel work group info!';
  rsErrorFailedToRetrieveKernelWorkGroupInfo2 = 'Error: Failed to retrieve '
    +'kernel work group info!';
  rsErrorFailedToSetKernelArguments = 'Error: Failed to set kernel arguments! ';
  rsErrorFailedToWriteToSourceArray = 'Error: Failed to write to source array!';
  rsSeconds = '%f seconds.';
  rsTimeToRunSeconds = 'Time to run: %f seconds.';
  SOpenClFiles = 'OpenCL-File';
  SAllFiles = 'All Files';

procedure testOpenCL(Filename:String);

const
  DATA_SIZE = 12800*8;

  compilerOptins : PChar = '-cl-opt-disable';

// Simple compute kernel which computes the square of an input array
var
  KernelSource, errorlog : PChar;

  errorlogstr: string[255];
  loglen: csize_t;

type single2 = record
  x: single;
  y: single;
end;

type TMyData = single2;

var
  err     : Integer; // error code returned from api calls
  data    : array [0..DATA_SIZE-1] of TMyData; // original data set given to device
  results : array [0..DATA_SIZE-1] of TMyData; // results returned from device
  correct : LongWord; // number of correct results returned

  global  : csize_t; // global domain size for our calculation
  local   : csize_t; // local domain size for our calculation

  device_id : cl_device_id;      // compute device id
  context   : cl_context;        // compute context
  commands  : cl_command_queue;  // compute command queue
  prog      : cl_program;        // compute program
  kernel    : cl_kernel;         // compute kernel

  input   : cl_mem; // device memory used for the input array
  output  : cl_mem; // device memory used for the output array

  i     : Integer;
  count : Integer;
  gpu   : cl_device_type;

  platformids   : Pcl_platform_id;
  num_platforms : cl_uint;

  tmpd  : single;

  loadedOpenClCode: TStrings;

  startTime, finishTime, totalTimeSeconds: double;
begin
  // Fill our data set with random float values
  count := DATA_SIZE;

  for i:=0 to count - 1 do
    data[i].x:= random;

  loadedOpenClCode := TStringList.Create();


  loadedOpenClCode.LoadFromFile(Filename);
  KernelSource := loadedOpenClCode.GetText;

  writeln(KernelSource);

  err:=clGetPlatformIDs(0,nil,@num_platforms);
  Writeln(Format(rsClGetPlatformID, [num_platforms]));
  if (err <> CL_SUCCESS) then
  begin
      writeln(rsErrCannotGetNumOfPl);
      exit;
  end;

  getmem(platformids,num_platforms*sizeof(cl_platform_id));

  err := clGetPlatformIDs(num_platforms, platformids, nil);

  if (err <> CL_SUCCESS) then begin
      Writeln(rsErrFailedToPlatfor);
      exit;
  end;

  // Connect to a compute device
  // change CL_DEVICE_TYPE_CPU to CL_DEVICE_TYPE_GPU is you have powerful video (GeForce 8800/8600M or higher)
  gpu := CL_DEVICE_TYPE_CPU;

  device_id:=nil;
  err := clGetDeviceIDs(platformids[0], gpu, 1, @device_id, nil);
  writeln(Format(rsClGetDeviceI, [ err]));
  if (err <> CL_SUCCESS) then begin
    Writeln(rsErrorFailedToCreateADeviceGroup);
    exit;
  end;

  // Create a compute context
  context := clCreateContext(nil, 1, @device_id, nil, nil, err);
  writeln('clCreateContext ', err);
  if context=nil then begin
    Writeln(rsErrorFailedToCreateAComputeContext);
    exit;
  end;

  // Create a command commands
  commands := clCreateCommandQueue(context, device_id, 0, err);
  writeln('clCreateCommandQueue ', err);
  if commands=nil then begin
    Writeln(rsErrorFailedToCreateACommandCommands);
    exit;
  end;

  // Create the compute program from the source buffer
  prog:= clCreateProgramWithSource(context, 1, PPChar(@KernelSource), nil, err);
  writeln('clCreateProgramWithSource ', err);
  if prog=nil then begin
    writeln(rsErrorFailedToCreateComputeProgram);
    exit;
  end;

  // Build the program executable
  err := clBuildProgram(prog, 0, nil, compilerOptins, nil, nil);
  writeln('clBuildProgram ', err);
  if (err <> CL_SUCCESS) then
  begin
    errorlog := @errorlogstr[1];
    loglen := 255;
    clGetProgramBuildInfo(prog,device_id,CL_PROGRAM_BUILD_LOG,255,errorlog,loglen);
    writeln(rsErrorFailedToBuildProgramExecutable);
    writeln(errorlog);
    exit;
  end;

  // Create the compute kernel in the program we wish to run
  kernel := clCreateKernel(prog, 'square', err);
  writeln('clCreateKernel ', err);
  if (kernel=nil) or (err <> CL_SUCCESS) then begin
    writeln(rsErrorFailedToCreateComputeKernel);
    exit;
  end;

  err := clGetKernelWorkGroupInfo(kernel, device_id,  CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), @local, nil);
  writeln('clGetKernelWorkGroupInfo ', err);
  if (err<>CL_SUCCESS) then begin
    writeln(rsErrorFailedToRetrieveKernelWorkGroupInfo);
    exit;
  end;

  // Create the input and output arrays in device memory for our calculation
  input := clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(TMyData) * count, nil, err);
  writeln('clCreateBuffer ', err);
  output := clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(TMyData) * count, nil, err);
  writeln('clCreateBuffer ', err);
  if (input=nil) or (output=nil) then begin
    writeln(rsErrorFailedToAllocateDeviceMemory);
    exit;
  end;

  // Write our data set into the input array in device memory
  err := clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(TMyData) * count, @data, 0, nil, nil);
  writeln('clEnqueueWriteBuffer ', err);
  if (err <> CL_SUCCESS) then begin
    writeln(rsErrorFailedToWriteToSourceArray);
    exit;
  end;

  // Set the arguments to our compute kernel
  err := 0;
  err := clSetKernelArg(kernel, 0, sizeof(cl_mem), @input);
  writeln('clSetKernelArg ', err);
  err := err or clSetKernelArg(kernel, 1, sizeof(cl_mem), @output);
  writeln('clSetKernelArg ', err);
  err := err or clSetKernelArg(kernel, 2, sizeof(longword), @count);
  writeln('clSetKernelArg ', err);
  if (err <> CL_SUCCESS) then begin
    writeln(rsErrorFailedToSetKernelArguments);
    exit;
  end;

  // Get the maximum work group size for executing the kernel on the device
  err := clGetKernelWorkGroupInfo(kernel, device_id,  CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), @local, nil);
  writeln('clGetKernelWorkGroupInfo ', err);
  if (err<>CL_SUCCESS) then begin
    writeln(rsErrorFailedToRetrieveKernelWorkGroupInfo2);
    exit;
  end;

  startTime := now();

  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  global := count;
  err := clEnqueueNDRangeKernel(commands, kernel, 1, nil, @global, @local, 0, nil, nil);
  writeln('clEnqueueNDRangeKernel ',err);
  if (err<>0) then begin
    writeln(rsErrorFailedToExecuteKernel);
    exit;
  end;

  // Wait for the command commands to get serviced before reading back results
  err:=clFinish(commands);
  writeln('clFinish ',err);

  // Read back the results from the device to verify the output
  err := clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(TMyData) * count, @results, 0, nil, nil);
  writeln('clEnqueueReadBuffer ',err);
  if (err <> CL_SUCCESS) then begin
    writeln(rsErrorFailedToReadOutputArray, err);
    exit;
  end;

  finishTime := now();

  totalTimeSeconds := (finishTime - startTime)*24*60*60;

  writeln(Format(rsTimeToRunSeconds, [totalTimeSeconds]));
  Form1.LabResult.caption := Format(rsSeconds, [round(
    totalTimeSeconds*100)/100]);

  // Validate our results
  correct := 0;
  for i:= 0 to count - 1 do
  begin
    //writeln('i:', i, ':', results[i].x:10:5, ':', results[i].y:10:5);
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
  Filename, FDataPath: String;
  i: Integer;
begin
  Filename:='evolve1.cl';
  FDataPath := 'Data';
  for i := 0 to 2 do
    if not DirectoryExists(FDataPath) then
      FDataPath:='..'+DirectorySeparator+ FDataPath
    else
      break;

   if not Fileexists(FdataPath+DirectorySeparator+'OpenCL'+DirectorySeparator+Filename) then
    begin
      Opendialog1.Filename := FdataPath+DirectorySeparator+'OpenCL'+DirectorySeparator+Filename;
      OpenDialog1.Filter:='*.cl|'+SOpenClFiles+'[*.cl]|*.*|'+SAllfiles+'[*.*]';
      OpenDialog1.DefaultExt:='.cl';
      if OpenDialog1.Execute then
         begin
           Filename:=OpenDialog1.FileName;
         end
    end
  else
    filename:= FdataPath+DirectorySeparator+'OpenCL'+DirectorySeparator+Filename;
  testOpenCL(Filename);
end;


end.

