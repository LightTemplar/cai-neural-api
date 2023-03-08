program testcnnalgo;
(*
 Coded by Joao Paulo Schwarz Schuler.
 // https://sourceforge.net/p/cai
 This command line tool runs the CAI Convolutional Neural Network with
 CIFAR10 files.

 In the case that your processor supports AVX instructions, uncomment
 {$DEFINE AVX} or {$DEFINE AVX2} defines. Also have a look at AVX512 define.

 Look at TTestCNNAlgo.WriteHelp; for more info.
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  CustApp,
  neuralnetwork,
  neuralvolume,
  Math,
  neuraldatasets,
  neuralfit;

const csLogEvery =
{$IFDEF Release}
  1000
{$ELSE}
  100
{$ENDIF}
;

type
  { TTestCNNAlgo }
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
    procedure RunAlgo(iAlgo: integer; fLearningRate, fInertia, fTarget: single);
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
    procedure WriteHelp; virtual;
  end;

  { testcnnalgo }
  procedure TTestCNNAlgo.DoRun;
  var
    Algo, LearningRate, Inertia, Target: string;
    iAlgo: integer;
    fLearningRate, fInertia, fTarget: single;
  begin
    // quick check parameters
    // parse parameters
    if HasOption('h', 'help') then
    begin
      WriteHelp;
    end;

    fLearningRate := 0.001;
    if HasOption('l', 'learningrate') then
    begin
      LearningRate := GetOptionValue('l', 'learningrate');
      fLearningRate := StrToFloat(LearningRate);
    end;

    fInertia := 0.9;
    if HasOption('i', 'inertia') then
    begin
      Inertia := GetOptionValue('i', 'inertia');
      fInertia := StrToFloat(Inertia);
    end;

    fTarget := 0.8;
    if HasOption('t', 'target') then
    begin
      Target := GetOptionValue('t', 'target');
      fTarget := StrToFloat(Target);
    end;

    if HasOption('a', 'algo') then
    begin
      Algo := GetOptionValue('a', 'algo');
      Writeln('Running algorithm:[',Algo,']');
      iAlgo := StrToInt(Algo);

      if (iAlgo > 0) and (iAlgo < 14) then
      begin
        RunAlgo(iAlgo, fLearningRate, fInertia, fTarget);
      end
      else
      begin
        WriteLn('Bad algorithm number:',iAlgo);
      end;
    end
    else
    begin
      {$IFDEF Release}
      WriteHelp;
      Write('Press ENTER to quit.');
      ReadLn();
      {$ELSE}
      iAlgo := 3;
      RunAlgo(iAlgo, fLearningRate, fInertia, fTarget);
      {$ENDIF}
    end;

    Terminate;
    Exit;
  end;

  procedure TTestCNNAlgo.RunAlgo(iAlgo: integer; fLearningRate, fInertia, fTarget: single);
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    I: integer;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    Volume: TNNetVolume;
    NumClasses: integer;
    fileNameBase: string;
    TA,TB,TC: TNNetLayer;
  begin
    if not CheckCIFARFile() then exit;
    WriteLn('Creating Neural Network...');
    NumClasses  := 10;
    NN := TNNet.Create();
    fileNameBase := 'autosave-neuralnetwork_a'+IntToStr(iAlgo);

    case iAlgo of
      1:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
      2:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
      3:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddMovingNorm(false, 0, 0);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));
        NN.AddMovingNorm(false, 0, 0);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
      4:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 2));
        NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(32, 3, 1, 2));
        NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 1));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
      5:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 4));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
        fLearningRate := Min(fLearningRate, 0.001);
      end;
      6:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 1));
        NN.AddLayer(TNNetMaxPool.Create(4));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
        fLearningRate := Min(fLearningRate, 0.001);
      end;
      7:
      begin
        //RELU + LOCAL CONNECT TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 1));
        NN.AddLayer(TNNetMaxPool.Create(4));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 1, 1));
        NN.AddLayer(TNNetLocalConnectReLU.Create(10, 1, 0, 1));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
        fLearningRate := Min(fLearningRate, 0.001);
      end;
      8:
      begin
        //RELU TESTING
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(20, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(20, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(160));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
      9:
      begin
        // Hiperbolic Tangent TESTING I
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolution.Create(64, 3, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(10));
        NN.AddLayer(TNNetConvolution.Create(64, 3, 0, 0));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(256));
        NN.AddLayer(TNNetLayerFullConnect.Create(NumClasses));
      end;
      10:
      begin
        // Hiperbolic Tangent TESTING II
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolution.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolution.Create(32, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolution.Create(32, 5, 0, 0));
        NN.AddLayer(TNNetLayerFullConnect.Create(32));
        NN.AddLayer(TNNetLayerFullConnect.Create(NumClasses));
      end;
      11:
      begin
        // Hiperbolic Tangent TESTING III
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolution.Create(16, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolution.Create(20, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolution.Create(20, 5, 2, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLayerFullConnect.Create(NumClasses));
      end;
      12:
      begin
        //Reshape, Identity, LocalConnect testing
        NN.AddLayer(TNNetInput.Create(32, 32, 3));
        NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetLocalConnectReLU.Create(128, 5, 0, 0));
        NN.AddLayer(TNNetReshape.Create(128,1,1));
        NN.AddLayer(TNNetIdentity.Create());
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());

        // This algorith requires special inertia
        fInertia := Max(fInertia, 0.999);
      end;
      13:
      begin
        // Network that splits into 2 branches and then later is concatenated
        TA := NN.AddLayer(TNNetInput.Create(32, 32, 3));

        // First branch starting from TA (5x5 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 5, 0, 0),TA);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        TB := NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));

        // Another branch starting from TA (3x3 features)
        NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 3, 0, 0),TA);
        NN.AddLayer(TNNetMaxPool.Create(2));
        NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));
        NN.AddLayer(TNNetMaxPool.Create(2));
        TC := NN.AddLayer(TNNetConvolutionReLU.Create(64, 6, 0, 0));

        // Concats both branches so the NN has only one end.
        NN.AddLayer(TNNetConcat.Create([TB,TC]));
        NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));
        NN.AddLayer(TNNetFullConnectLinear.Create(NumClasses));
        NN.AddLayer(TNNetSoftMax.Create());
      end;
    end;
    WriteLn('Learning rate set to: [',fLearningRate:7:5,']');
    WriteLn('Inertia set to: [',fInertia:7:5,']');
    WriteLn('Target set to: [',fTarget:7:5,']');

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons:', NN.CountNeurons());
    WriteLn(' Weights:' ,NN.CountWeights());
    NN.DebugWeights();
    NN.DebugStructure();

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := fileNameBase;
    NeuralFit.InitialLearningRate := fLearningRate;
    NeuralFit.Inertia := fInertia;
    NeuralFit.TargetAccuracy := fTarget;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, NumClasses, {batchsize=}128, {epochs=}100);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
  end;

  constructor TTestCNNAlgo.Create(TheOwner: TComponent);
  begin
    inherited Create(TheOwner);
    StopOnException := True;
  end;

  destructor TTestCNNAlgo.Destroy;
  begin
    inherited Destroy;
  end;

  procedure TTestCNNAlgo.WriteHelp;
  begin
    WriteLn
    (
      'CIFAR-10 Classification Example by Joao Paulo Schwarz Schuler',sLineBreak,
      'Command Line Example: cifar10 -a 1', sLineBreak,
      ' -h : displays this help. ', sLineBreak,
      ' -l : defines learing rate. Default is -l 0.01. ', sLineBreak,
      ' -i : defines inertia. Default is -i 0.9.', sLineBreak,
      ' -a : defines the algorithm or neural network structure. Follows examples:', sLineBreak,
      ' -a 1 means:', sLineBreak,
      '   NN.AddLayer(TNNetInput.Create(32, 32, 3)); ', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0)); ', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2)); ', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0)); ', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2)); ', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(128, 5, 0, 0)); ', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(64)); ', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(NumClasses)); ', sLineBreak,
      ' -a 2 means:', sLineBreak,
      '   NN.AddLayer(TNNetInput.Create(32, 32, 3));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(64, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(64));', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(NumClasses));', sLineBreak,
      ' -a 3 means:', sLineBreak,
      '   NN.AddLayer(TNNetInput.Create(32, 32, 3));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(16, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetMaxPool.Create(2));', sLineBreak,
      '   NN.AddLayer(TNNetConvolutionReLU.Create(32, 5, 0, 0));', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(32));', sLineBreak,
      '   NN.AddLayer(TNNetLayerFullConnectReLU.Create(NumClasses));', sLineBreak,
      ' Algorithms 1,2 and 3 use ReLU. Algorithms ',
      ' 9,10 and 11 use hiperbolic tangents.',sLineBreak,
      ' ReLU implementations are a lot faster to run.',sLineBreak,
      ' If you are unsure about what algorithm to run, try typing "cifar10 -a 1"',sLineBreak,
      ' You can find other algoriths looking at testcnnalgo.lpr source code:',sLineBreak,
      ' https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/testcnnalgo/testcnnalgo.lpr',sLineBreak,
      ' More info at:',sLineBreak,
      '   https://sourceforge.net/projects/cai/'
    );
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
