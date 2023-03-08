// This OpenCL program implements error minimization via evolutionary algorithm.
// The error function is: abs(x-5000)

// MWC64X random generator got from the link below
// http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state)
{
    enum { A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}

float randomFloat(uint2 *state, float size)
{
  const uint rndValue = MWC64X( state );
  return ( ( (convert_float(rndValue & 65535) )/65536)*size);
}

float eval(float x)
{
  return fabs(x-5000.0f);
}

float mutate(uint2 *state, float x)
{
  return x+randomFloat(state, 10.0f)-5.0f;
}

float2 getBestOf(uint2 *state, float2 pInput, uint cnt)
{
  float2 result = pInput;

  for ( uint i = 0; i<cnt; i++ )
  {
    float x = mutate(state, pInput.x);
    float y = eval(x);
    if (y < result.y)
    {
     result.x = x;
     result.y = y;
    }
  }

  return result;
}

__kernel void minimize_error
(
  __global float2* input,
  __global float2* output,
  const unsigned int count
)
{
  uint i = get_global_id(0);
  uint j;
  uint2 randomState = {i+1,0};

  if (i < count)
  {
    float x = input[i].x;

    float2 result = {x , eval(x)};
    float outputeval = eval(output[i].x);

    // is the output already better than the input?
    if ( outputeval < result.y )
    {
      result = output[i];
    }

    for( j=0; j<100; j++ )
    {
      result = getBestOf(&randomState, result, 10);
      //result.x = randomFloat(&randomState, 1.0f);
      //result.y = mutate(&randomState, 0.0f);
      //result.y = randomFloat(&randomState, 1.0f);
      output[i] = result;

      if (i < count - 1)
      {
        // is the result worse than the next parallel result?
        if ( (output[i].y > output[i+1].y) && (output[i+1].y>0) )
        {
          output[i] = output[i+1];
          result = output[i];
        }
      }

      if (i > 1)
      {
        // is the result worse than the previous parallel result?
        if ( (output[i].y > output[i-1].y) && (output[i-1].y>0) )
        {
          output[i] = output[i-1];
          result = output[i];
        }
      }

      if (i < count - 33)
      {
        // is the result worse than the next parallel result?
        if ( (output[i].y > output[i+33].y) && (output[i+33].y>0) )
        {
          output[i] = output[i+33];
          result = output[i];
        }
      }

      if (i > 33)
      {
        // is the result worse than the previous parallel result?
        if ( (output[i].y > output[i-33].y) && (output[i-33].y>0) )
        {
          output[i] = output[i-33];
          result = output[i];
        }
      }

    } // end of for
  } // end of if
} // end of kernel
