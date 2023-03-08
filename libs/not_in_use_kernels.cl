// these kernels aren't currently in use.
// A vectors (A1, A2, A3, ...) are operated with a number of
// B vectors (B1, B2, B3, ...) via dot product. There is a resulting vector
// R with all dot products A1.B1 .. A1.B2 .. A2.B1 .. AN.BN .
// A vectors are sometimes interleaved.
__kernel void cai_dot_product2
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8  = FSize -  8;
    const int FSizeMinus32 = FSize - 32;

    const int a0 =  0*FNumAs;
    const int a1 =  1*FNumAs;
    const int a2 =  2*FNumAs;
    const int a3 =  3*FNumAs;
    const int a4 =  4*FNumAs;
    const int a5 =  5*FNumAs;
    const int a6 =  6*FNumAs;
    const int a7 =  7*FNumAs;
    const int a8 =  8*FNumAs;
    const int a9 =  9*FNumAs;
    const int a10 = 10*FNumAs;
    const int a11 = 11*FNumAs;
    const int a12 = 12*FNumAs;
    const int a13 = 13*FNumAs;
    const int a14 = 14*FNumAs;
    const int a15 = 15*FNumAs;
    const int a16 = 16*FNumAs;
    const int a17 = 17*FNumAs;
    const int a18 = 18*FNumAs;
    const int a19 = 19*FNumAs;
    const int a20 = 20*FNumAs;
    const int a21 = 21*FNumAs;
    const int a22 = 22*FNumAs;
    const int a23 = 23*FNumAs;
    const int a24 = 24*FNumAs;
    const int a25 = 25*FNumAs;
    const int a26 = 26*FNumAs;
    const int a27 = 27*FNumAs;
    const int a28 = 28*FNumAs;
    const int a29 = 29*FNumAs;
    const int a30 = 30*FNumAs;
    const int a31 = 31*FNumAs;

    while (i < FSizeMinus32)
    {
      const int startBPos = i + VectBPos;

      //a_id + (i+31)*FNumAs -> a_id + i*FNumAs + FNumAs * 31 -> ai + a31

      const int ai =  a_id + i*FNumAs;

      DotProductResult =
        mad(FInputBufferAs[ai +  a0], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[ai +  a1], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[ai +  a2], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[ai +  a3], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[ai +  a4], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[ai +  a5], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[ai +  a6], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[ai +  a7], FInputBufferBs[startBPos +  7],
        mad(FInputBufferAs[ai +  a8], FInputBufferBs[startBPos +  8],
        mad(FInputBufferAs[ai +  a9], FInputBufferBs[startBPos +  9],
        mad(FInputBufferAs[ai + a10], FInputBufferBs[startBPos + 10],
        mad(FInputBufferAs[ai + a11], FInputBufferBs[startBPos + 11],
        mad(FInputBufferAs[ai + a12], FInputBufferBs[startBPos + 12],
        mad(FInputBufferAs[ai + a13], FInputBufferBs[startBPos + 13],
        mad(FInputBufferAs[ai + a14], FInputBufferBs[startBPos + 14],
        mad(FInputBufferAs[ai + a15], FInputBufferBs[startBPos + 15],
        mad(FInputBufferAs[ai + a16], FInputBufferBs[startBPos + 16],
        mad(FInputBufferAs[ai + a17], FInputBufferBs[startBPos + 17],
        mad(FInputBufferAs[ai + a18], FInputBufferBs[startBPos + 18],
        mad(FInputBufferAs[ai + a19], FInputBufferBs[startBPos + 19],
        mad(FInputBufferAs[ai + a20], FInputBufferBs[startBPos + 20],
        mad(FInputBufferAs[ai + a21], FInputBufferBs[startBPos + 21],
        mad(FInputBufferAs[ai + a22], FInputBufferBs[startBPos + 22],
        mad(FInputBufferAs[ai + a23], FInputBufferBs[startBPos + 23],
        mad(FInputBufferAs[ai + a24], FInputBufferBs[startBPos + 24],
        mad(FInputBufferAs[ai + a25], FInputBufferBs[startBPos + 25],
        mad(FInputBufferAs[ai + a26], FInputBufferBs[startBPos + 26],
        mad(FInputBufferAs[ai + a27], FInputBufferBs[startBPos + 27],
        mad(FInputBufferAs[ai + a28], FInputBufferBs[startBPos + 28],
        mad(FInputBufferAs[ai + a29], FInputBufferBs[startBPos + 29],
        mad(FInputBufferAs[ai + a30], FInputBufferBs[startBPos + 30],
        mad(FInputBufferAs[ai + a31], FInputBufferBs[startBPos + 31],
        DotProductResult
        ))))))))
        ))))))))
        ))))))))
        ))))))));
      i += 32;
    }

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;
      const int ai =  a_id + i*FNumAs;

      DotProductResult =
        mad(FInputBufferAs[ai +  a0], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[ai +  a1], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[ai +  a2], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[ai +  a3], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[ai +  a4], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[ai +  a5], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[ai +  a6], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[ai +  a7], FInputBufferBs[startBPos +  7],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }

    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

#define TS 16 // The tile-size
__kernel void simpleGEMMT(
  const int FThreadCount,
  const int M, const int N, const int K,
  int ActFN,
  __global __read_only float* A,
  __global __read_only float* B,
  __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // 0..M
    const int globalCol = TS*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];

    // Initialise the accumulation registers
    float acc;
    acc = 0.0f;

    // Loop over all tiles
    int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        int tiledIndex = TS*t;
        //int indexA = globalRow*K + tiledIndex + col; // not interleaved: a_id*K + i // a_id = globalRow; i = tileIndex + col
        int indexA = globalRow + (tiledIndex + col)*M; // interleaved: a_id + (i+0)*FNumAs
        int indexB = globalCol*K + tiledIndex + row;
        Asub[row][col] = A[indexA];
        Bsub[row][col] = B[indexB];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
          acc += Asub[row][k] * Bsub[k][col];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalCol * M + globalRow] = acc;
}

__kernel void cai_dot_product_div
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSizeTotal,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int global_a_id  = get_global_id(0);
  const int global_b_id  = get_global_id(1);

  if ( (global_a_id < FNumAs) && (global_b_id < FNumBs) )
  {
    const int c_id         = get_global_id(2);
    const int b_id         = global_b_id * 16 + c_id;
    const int a_id         = global_a_id * 16 + c_id;
    const int FSize        = FSizeTotal / 16;
    const int FSizeMinus8  = FSize -  8;
    const int FSizeMinus32 = FSize - 32;
    const int VectBPos     = b_id * FSize;

    float DotProductResult = 0.0f;
    int i = 0;

    while (i < FSizeMinus32)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(FInputBufferAs[a_id + (i+ 0)*FNumAs], FInputBufferBs[startBPos +  0],
        mad(FInputBufferAs[a_id + (i+ 1)*FNumAs], FInputBufferBs[startBPos +  1],
        mad(FInputBufferAs[a_id + (i+ 2)*FNumAs], FInputBufferBs[startBPos +  2],
        mad(FInputBufferAs[a_id + (i+ 3)*FNumAs], FInputBufferBs[startBPos +  3],
        mad(FInputBufferAs[a_id + (i+ 4)*FNumAs], FInputBufferBs[startBPos +  4],
        mad(FInputBufferAs[a_id + (i+ 5)*FNumAs], FInputBufferBs[startBPos +  5],
        mad(FInputBufferAs[a_id + (i+ 6)*FNumAs], FInputBufferBs[startBPos +  6],
        mad(FInputBufferAs[a_id + (i+ 7)*FNumAs], FInputBufferBs[startBPos +  7],
        mad(FInputBufferAs[a_id + (i+ 8)*FNumAs], FInputBufferBs[startBPos +  8],
        mad(FInputBufferAs[a_id + (i+ 9)*FNumAs], FInputBufferBs[startBPos +  9],
        mad(FInputBufferAs[a_id + (i+10)*FNumAs], FInputBufferBs[startBPos + 10],
        mad(FInputBufferAs[a_id + (i+11)*FNumAs], FInputBufferBs[startBPos + 11],
        mad(FInputBufferAs[a_id + (i+12)*FNumAs], FInputBufferBs[startBPos + 12],
        mad(FInputBufferAs[a_id + (i+13)*FNumAs], FInputBufferBs[startBPos + 13],
        mad(FInputBufferAs[a_id + (i+14)*FNumAs], FInputBufferBs[startBPos + 14],
        mad(FInputBufferAs[a_id + (i+15)*FNumAs], FInputBufferBs[startBPos + 15],
        mad(FInputBufferAs[a_id + (i+16)*FNumAs], FInputBufferBs[startBPos + 16],
        mad(FInputBufferAs[a_id + (i+17)*FNumAs], FInputBufferBs[startBPos + 17],
        mad(FInputBufferAs[a_id + (i+18)*FNumAs], FInputBufferBs[startBPos + 18],
        mad(FInputBufferAs[a_id + (i+19)*FNumAs], FInputBufferBs[startBPos + 19],
        mad(FInputBufferAs[a_id + (i+20)*FNumAs], FInputBufferBs[startBPos + 20],
        mad(FInputBufferAs[a_id + (i+21)*FNumAs], FInputBufferBs[startBPos + 21],
        mad(FInputBufferAs[a_id + (i+22)*FNumAs], FInputBufferBs[startBPos + 22],
        mad(FInputBufferAs[a_id + (i+23)*FNumAs], FInputBufferBs[startBPos + 23],
        mad(FInputBufferAs[a_id + (i+24)*FNumAs], FInputBufferBs[startBPos + 24],
        mad(FInputBufferAs[a_id + (i+25)*FNumAs], FInputBufferBs[startBPos + 25],
        mad(FInputBufferAs[a_id + (i+26)*FNumAs], FInputBufferBs[startBPos + 26],
        mad(FInputBufferAs[a_id + (i+27)*FNumAs], FInputBufferBs[startBPos + 27],
        mad(FInputBufferAs[a_id + (i+28)*FNumAs], FInputBufferBs[startBPos + 28],
        mad(FInputBufferAs[a_id + (i+29)*FNumAs], FInputBufferBs[startBPos + 29],
        mad(FInputBufferAs[a_id + (i+30)*FNumAs], FInputBufferBs[startBPos + 30],
        mad(FInputBufferAs[a_id + (i+31)*FNumAs], FInputBufferBs[startBPos + 31],
        DotProductResult
        ))))))))
        ))))))))
        ))))))))
        ))))))));

      i += 32;
    }

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;

      DotProductResult =
        mad(FInputBufferAs[a_id + (i+0)*FNumAs], FInputBufferBs[startBPos + 0],
        mad(FInputBufferAs[a_id + (i+1)*FNumAs], FInputBufferBs[startBPos + 1],
        mad(FInputBufferAs[a_id + (i+2)*FNumAs], FInputBufferBs[startBPos + 2],
        mad(FInputBufferAs[a_id + (i+3)*FNumAs], FInputBufferBs[startBPos + 3],
        mad(FInputBufferAs[a_id + (i+4)*FNumAs], FInputBufferBs[startBPos + 4],
        mad(FInputBufferAs[a_id + (i+5)*FNumAs], FInputBufferBs[startBPos + 5],
        mad(FInputBufferAs[a_id + (i+6)*FNumAs], FInputBufferBs[startBPos + 6],
        mad(FInputBufferAs[a_id + (i+7)*FNumAs], FInputBufferBs[startBPos + 7],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }

    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }

    FResultBuffer[global_b_id * FNumAs + global_a_id] = DotProductResult;
  }
} // end of kernel

__kernel void cai_dot_product_simple16
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float16* FInputBufferAs,
  __global __read_only float16* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int a_id = get_global_id(0);
  const int b_id = get_global_id(1);
  const int local_a_id = get_local_id(0);
  const int local_b_id = get_local_id(1);
  const int group_a_id = 16*get_group_id(0);
  const int group_b_id = 16*get_group_id(1);
  const int SizeDiv16 = FSize / 16;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectAPos = a_id * SizeDiv16;
    const int VectBPos = b_id * SizeDiv16;

    float16 DotProductResult = 0.0f;
    int i = 0;

    while (i < SizeDiv16)
    {
      __local float16 Asub[16];
      __local float16 Bsub[16];

      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_b_id == 0)
      {
        Asub[local_a_id] = FInputBufferAs[(group_a_id + local_a_id)*SizeDiv16 + i];
        Bsub[local_a_id] = FInputBufferBs[(group_b_id + local_a_id)*SizeDiv16 + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      DotProductResult =
        mad(Asub[local_a_id], Bsub[local_b_id], DotProductResult);
      i += 1;
    }

    float8 Final8 = DotProductResult.lo + DotProductResult.hi;
    float4 Final4 = Final8.lo + Final8.hi;
    float2 Final2 = Final4.lo + Final4.hi;
    float FinalResult   = Final2.lo + Final2.hi;

    if (ActFN == 1)
    {
      if (FinalResult < 0.0f) { FinalResult = 0.0f; }
    }

    FResultBuffer[b_id * FNumAs + a_id] = FinalResult;//b_id + a_id;
  }
} // end of kernel

__kernel void cai_dot_product8ii
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int local_a_id = get_local_id(0);
  const int local_b_id = get_local_id(1);
  const int group_a_id = 8*get_group_id(0);
  const int group_b_id = 8*get_group_id(1);
  const int a_id = group_a_id + local_a_id;
  const int b_id = group_b_id + local_b_id;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8 = FSize - 8;

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;

      __local float Asub[8][8];
      __local float Bsub[8][8];

      barrier(CLK_LOCAL_MEM_FENCE);
      Asub[local_a_id][local_b_id] = FInputBufferAs[a_id + (i+local_b_id)*FNumAs];
      Bsub[local_a_id][local_b_id] = FInputBufferBs[b_id + (i+local_a_id)*FNumBs];
      barrier(CLK_LOCAL_MEM_FENCE);

      DotProductResult =
        mad(Asub[local_a_id][0], Bsub[0][local_b_id],
        mad(Asub[local_a_id][1], Bsub[1][local_b_id],
        mad(Asub[local_a_id][2], Bsub[2][local_b_id],
        mad(Asub[local_a_id][3], Bsub[3][local_b_id],
        mad(Asub[local_a_id][4], Bsub[4][local_b_id],
        mad(Asub[local_a_id][5], Bsub[5][local_b_id],
        mad(Asub[local_a_id][6], Bsub[6][local_b_id],
        mad(Asub[local_a_id][7], Bsub[7][local_b_id],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[b_id + i*FNumBs], DotProductResult);
        i += 1;
    }
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

__kernel void cai_dot_product8i
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int local_a_id = get_local_id(0);
  const int local_b_id = get_local_id(1);
  const int group_a_id = 8*get_group_id(0);
  const int group_b_id = 8*get_group_id(1);
  const int a_id = group_a_id + local_a_id;
  const int b_id = group_b_id + local_b_id;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8 = FSize - 8;

    while (i < FSizeMinus8)
    {
      const int startBPos = i + VectBPos;

      __local float Asub[8][8];
      __local float Bsub[8][8];

      barrier(CLK_LOCAL_MEM_FENCE);
      Asub[local_a_id][local_b_id] = FInputBufferAs[a_id + (i+local_b_id)*FNumAs];
      Bsub[local_b_id][local_a_id] = FInputBufferBs[startBPos + local_a_id];
      barrier(CLK_LOCAL_MEM_FENCE);

      DotProductResult +=
        Asub[local_a_id][0] * Bsub[local_b_id][0] +
        Asub[local_a_id][1] * Bsub[local_b_id][1] +
        Asub[local_a_id][2] * Bsub[local_b_id][2] +
        Asub[local_a_id][3] * Bsub[local_b_id][3] +
        Asub[local_a_id][4] * Bsub[local_b_id][4] +
        Asub[local_a_id][5] * Bsub[local_b_id][5] +
        Asub[local_a_id][6] * Bsub[local_b_id][6] +
        Asub[local_a_id][7] * Bsub[local_b_id][7];
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[a_id + i*FNumAs], FInputBufferBs[i + VectBPos], DotProductResult);
        i += 1;
    }
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

__kernel void cai_dot_product8
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int local_a_id = get_local_id(0);
  const int local_b_id = get_local_id(1);
  const int group_a_id = 8*get_group_id(0);
  const int group_b_id = 8*get_group_id(1);
  const int a_id = group_a_id + local_a_id;
  const int b_id = group_b_id + local_b_id;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectAPos = a_id * FSize;
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus8 = FSize - 8;

    while (i < FSizeMinus8)
    {
      const int startAPos = i + VectAPos;
      const int startBPos = i + VectBPos;

      __local float Asub[8][8];
      __local float Bsub[8][8];

      barrier(CLK_LOCAL_MEM_FENCE);
      Asub[local_a_id][local_b_id] = FInputBufferAs[startAPos + local_b_id];
      Bsub[local_a_id][local_b_id] = FInputBufferBs[startBPos + local_a_id];
      barrier(CLK_LOCAL_MEM_FENCE);

      DotProductResult =
        mad(Asub[local_a_id][0], Bsub[0][local_b_id],
        mad(Asub[local_a_id][1], Bsub[1][local_b_id],
        mad(Asub[local_a_id][2], Bsub[2][local_b_id],
        mad(Asub[local_a_id][3], Bsub[3][local_b_id],
        mad(Asub[local_a_id][4], Bsub[4][local_b_id],
        mad(Asub[local_a_id][5], Bsub[5][local_b_id],
        mad(Asub[local_a_id][6], Bsub[6][local_b_id],
        mad(Asub[local_a_id][7], Bsub[7][local_b_id],
        DotProductResult))))))));
      i += 8;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[VectAPos + i], FInputBufferBs[VectBPos + i], DotProductResult);
        i += 1;
    }
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel

__kernel void cai_dot_product16
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int local_a_id = get_local_id(0);
  const int local_b_id = get_local_id(1);
  const int group_a_id = 16*get_group_id(0);
  const int group_b_id = 16*get_group_id(1);
  const int a_id = group_a_id + local_a_id;
  const int b_id = group_b_id + local_b_id;

  if ( (a_id < FNumAs) && (b_id < FNumBs) )
  {
    const int VectAPos = a_id * FSize;
    const int VectBPos = b_id * FSize;

    float DotProductResult = 0;
    int i = 0;

    const int FSizeMinus16 = FSize - 16;

    while (i < FSizeMinus16)
    {
      const int startAPos = i + VectAPos;
      const int startBPos = i + VectBPos;

      __local float Asub[16][16];
      __local float Bsub[16][16];

      barrier(CLK_LOCAL_MEM_FENCE);
      Asub[local_a_id][local_b_id] = FInputBufferAs[startAPos + local_b_id];
      Bsub[local_a_id][local_b_id] = FInputBufferBs[startBPos + local_a_id];
      barrier(CLK_LOCAL_MEM_FENCE);

      DotProductResult +=
        Asub[local_a_id][0]  * Bsub[local_b_id][0] +
        Asub[local_a_id][1]  * Bsub[local_b_id][1] +
        Asub[local_a_id][2]  * Bsub[local_b_id][2] +
        Asub[local_a_id][3]  * Bsub[local_b_id][3] +
        Asub[local_a_id][4]  * Bsub[local_b_id][4] +
        Asub[local_a_id][5]  * Bsub[local_b_id][5] +
        Asub[local_a_id][6]  * Bsub[local_b_id][6] +
        Asub[local_a_id][7]  * Bsub[local_b_id][7] +
        Asub[local_a_id][8]  * Bsub[local_b_id][8] +
        Asub[local_a_id][9]  * Bsub[local_b_id][9] +
        Asub[local_a_id][10] * Bsub[local_b_id][10]+
        Asub[local_a_id][11] * Bsub[local_b_id][11]+
        Asub[local_a_id][12] * Bsub[local_b_id][12]+
        Asub[local_a_id][13] * Bsub[local_b_id][13]+
        Asub[local_a_id][14] * Bsub[local_b_id][14]+
        Asub[local_a_id][15] * Bsub[local_b_id][15];
      i += 16;
    }

    while (i < FSize)
    {
      DotProductResult =
        mad(FInputBufferAs[VectAPos + i], FInputBufferBs[VectBPos + i], DotProductResult);
        i += 1;
    }
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    FResultBuffer[b_id * FNumAs + a_id] = DotProductResult;
  }
} // end of kernel


// This kernel assumes only one B vector.
__kernel void cai_dot_product_simple_one_vector
(
  const int FThreadCount,
  const int FNumAs,
  const int FNumBs,
  const int FSize,
  int ActFN,
  __global __read_only float* FInputBufferAs,
  __global __read_only float* FInputBufferBs,
  __global   float* FResultBuffer
)
{
  const int GlobalId = get_global_id(0);

  if (GlobalId < FThreadCount)
  {
    float DotProductResult = 0;
    int i = 0;

    while (i < FSize)
    {
        DotProductResult =
          mad(FInputBufferAs[GlobalId + i*FNumAs], FInputBufferBs[i], DotProductResult);
        i += 1;
    }
    if (ActFN == 1)
    {
      if (DotProductResult < 0.0f) { DotProductResult = 0.0f; }
    }
    FResultBuffer[GlobalId] = DotProductResult;
  }
} // end of kernel
