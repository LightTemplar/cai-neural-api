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

