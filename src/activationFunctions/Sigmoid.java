package activationFunctions;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class Sigmoid extends ActivationFunction
{
	
    public static CUcontext context;
    private static CUmodule module;
    private static CUfunction activation;
    private static CUfunction activationDerivative;
    private static CUdeviceptr deviceBuffer;
    
    static
    {
    	JCudaDriver.setExceptionsEnabled(true);
        //init();
    }

	@Override
	public float applyActivationFunction(float input) 
	{
		return (float) (1/(1+Math.exp(-input)));
	}

	@Override
	public float getDerivative(float input) 
	{
		return (float) (applyActivationFunction(input)*(1.0-applyActivationFunction(input)));
	}
	
	/*
	@Override
	public Matrix applyActivationFunction(Matrix input)
	{
		input=new FDMatrix(new float[1][10]);
		((FDMatrix)input).sendToGPU();
		
		int maxThreads=1;
    	int maxBlocks=1;
    	int numBlocks = getNumBlocks(input.getLen(), maxBlocks, maxThreads);
        int numThreads = getNumThreads(input.getLen(), maxBlocks, maxThreads);
        
        int sharedMemSize = numThreads * Sizeof.FLOAT;
        if (numThreads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(((FDMatrix)input).gpuPointer),
            Pointer.to(new int[]{input.getLen()})
        );

        try
        {
        // Call the kernel function.
        cuLaunchKernel(activation,
            numBlocks,  1, 1,         // Grid dimension
            numThreads, 1, 1,         // Block dimension
            sharedMemSize, null,   // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        }
        catch(Exception e)
        {
        	//e.printStackTrace();
        	System.out.println("Sigmoid cuda error");
        }
        //cuCtxSynchronize();
            
        return input;
	}
	
	@Override
	public Matrix getDerivatives(Matrix input)
	{
		((FDMatrix)input).sendToGPU();
		
		int maxThreads=128;
    	int maxBlocks=64;
    	int numBlocks = getNumBlocks(input.getLen(), maxBlocks, maxThreads);
        int numThreads = getNumThreads(input.getLen(), maxBlocks, maxThreads);
        
        int sharedMemSize = numThreads * Sizeof.FLOAT;
        if (numThreads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(((FDMatrix)input).gpuPointer),
            Pointer.to(new int[]{input.getLen()})
        );

        try
        {
        // Call the kernel function.
        cuLaunchKernel(activationDerivative,
            numBlocks,  1, 1,         // Grid dimension
            numThreads, 1, 1,         // Block dimension
            sharedMemSize, null,   // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
		}
	    catch(Exception e)
	    {
	    	//e.printStackTrace();
	    	System.out.println("Sigmoid deriv cuda error");
	    }
        cuCtxSynchronize();
            
        return input;
	}
    
    private static void init()
    {
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        prepare();
    }
    
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxSigmoidFileName = null;
        try
        {
        	ptxSigmoidFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/activationFunctions/sigmoid.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        String ptxSigmoidDerivativeFileName = null;
        try
        {
        	ptxSigmoidDerivativeFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/activationFunctions/sigmoidDerivative.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module=new CUmodule();
        
        // Obtain a function pointer to the "reduce" function.
        cuModuleLoad(module, ptxSigmoidFileName);
        activation=new CUfunction();
        cuModuleGetFunction(activation, module, "sigmoid");
        
        cuModuleLoad(module, ptxSigmoidDerivativeFileName);
        activationDerivative=new CUfunction();
        cuModuleGetFunction(activationDerivative, module, "sigmoidDerivative");
    }
    
    public static void shutdown()
    {
        cuModuleUnload(module);
        if(deviceBuffer!=null)
        {
        	cuMemFree(deviceBuffer);
        }
        if (context != null)
        {
            cuCtxDestroy(context);
        }
    }
    
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }
    
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
            "/usr/local/cuda/bin/nvcc " + modelString + " -ptx "+
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
*/
	
}
