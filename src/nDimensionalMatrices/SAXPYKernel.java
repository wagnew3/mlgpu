package nDimensionalMatrices;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2011 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.driver.JCudaDriver.*;

import java.io.*;
import java.util.Random;

import jcuda.*;
import jcuda.driver.*;

/**
 * Example of a reduction. It is based on the NVIDIA 'reduction' sample, 
 * and uses an adopted version of one of the kernels presented in 
 * this sample. 
 */
public class SAXPYKernel
{
    /**
     * The CUDA context created by this sample
     */
    private static CUcontext context;
    
    /**
     * The module which is loaded in form of a PTX file
     */
    private static CUmodule module;
    
    /**
     * The actual kernel function from the module
     */
    private static CUfunction function;
    
    /**
     * Temporary memory for the device output
     */
    private static CUdeviceptr deviceBuffer;
    
    /**
     * Entry point of this sample
     *
     * @param args Not used
     */
    public static void main(String args[])
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        init();
        boolean passed = true;
        for (int n = 100000; n <= 25600000; n *= 2)
        {
        	float a=1.0f;
        	float[] x=createRandomArray(n);
            float[] y=createRandomArray(n);

            long time0 = 0;
            long time1 = 0;

            // Copy the input data to the device
            time0 = System.nanoTime();
            
            CUdeviceptr xPtr=new CUdeviceptr();
            cuMemAlloc(xPtr, x.length * Sizeof.FLOAT);
            cuMemcpyHtoD(xPtr, Pointer.to(x), 
                x.length * Sizeof.FLOAT);
            
            CUdeviceptr yPtr=new CUdeviceptr();
            cuMemAlloc(yPtr, y.length * Sizeof.FLOAT);
            cuMemcpyHtoD(yPtr, Pointer.to(y), 
                y.length * Sizeof.FLOAT);
            
            time1 = System.nanoTime();
            long durationCopy = time1 - time0;

            // Execute the reduction with CUDA
            time0 = System.nanoTime();
            float[] resultJCuda = saxpy(a, xPtr, yPtr, x.length);
            time1 = System.nanoTime();
            long durationComp = time1 - time0;

            cuMemFree(xPtr);
            cuMemFree(yPtr);

            System.out.println("Reduction of " + n + " elements");
            System.out.printf(
                "  JCuda: %5.3fms " +
                "(copy: %5.3fms, comp: %5.3fms)\n",
                (durationCopy + durationComp) / 1e6, 
                durationCopy / 1e6, durationComp / 1e6);
        }

        shutdown();
    }  
    
    public static float[] saxpy(float a, Pointer xPtr, Pointer yPtr, int length)
    {
    	int maxThreads=128;
    	int maxBlocks=64;
    	int numBlocks = getNumBlocks(length, maxBlocks, maxThreads);
        int numThreads = getNumThreads(length, maxBlocks, maxThreads);
        
        int sharedMemSize = numThreads * Sizeof.FLOAT;
        if (numThreads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
        	Pointer.to(new float[]{a}),
            Pointer.to(xPtr),
            Pointer.to(yPtr),
            Pointer.to(deviceBuffer),
            Pointer.to(new int[]{length})
        );

        // Call the kernel function.
        cuLaunchKernel(function,
            numBlocks,  1, 1,         // Grid dimension
            numThreads, 1, 1,         // Block dimension
            sharedMemSize, null,   // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        
        float[] result=new float[length];
        cuMemcpyDtoH(Pointer.to(result), deviceBuffer, length);     
        return result;
    }
    
    /**
     * Initialize the driver API and create a context for the first
     * device, and then call {@link #prepare()}
     */
    private static void init()
    {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        prepare();
    }
    
    /**
     * Prepare everything for calling the reduction kernel function.
     * This method assumes that a context already has been created
     * and is current!
     */
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxFileName = null;
        try
        {
            ptxFileName = preparePtxFile("C:\\Users\\C\\workspace\\mlGPU\\src\\nDimensionalMatrices\\saxpy2.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "reduce" function.
        function = new CUfunction();
        cuModuleGetFunction(function, module, "saxpy2");
        
        // Allocate a chunk of temporary memory (must be at least
        // numberOfBlocks * Sizeof.FLOAT)
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, 25600000 * Sizeof.FLOAT);
        
    }
    
    /**
     * Release all resources allocated by this class
     */
    public static void shutdown()
    {
        cuModuleUnload(module);
        cuMemFree(deviceBuffer);
        if (context != null)
        {
            cuCtxDestroy(context);
        }
    }
    
    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }
    
    /**
     * Returns the power of 2 that is equal to or greater than x
     * 
     * @param x The input
     * @return The next power of 2
     */
    private static int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    
    /**
     * Create an array of the given size, with random data
     * 
     * @param size The array size
     * @return The array
     */
    private static float[] createRandomArray(int size)
    {
        Random random = new Random();
        float array[] = new float[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = random.nextFloat();
        }
        return array;
    }
    

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }


}