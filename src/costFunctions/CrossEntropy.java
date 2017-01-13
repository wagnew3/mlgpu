package costFunctions;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import activationFunctions.Sigmoid;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import nDimensionalMatrices.*;

public class CrossEntropy extends CostFunction
{
	
	public static CUcontext context;
    private static CUmodule module;
    private static CUfunction crossEntropyCost;
    private static CUfunction crossEntropyCostDerivative;
    private static CUdeviceptr deviceBuffer;

    static
    {
    	if(FDMatrix.GPU)
    	{
	    	JCudaDriver.setExceptionsEnabled(true);
	        init();
    	}
    }
    
    /*
	@Override
	public float getCost(Matrix input, Matrix networkOutput, Matrix desiredOutput) 
	{
		double cost=0.0;
		for(int entryInd=0; entryInd<networkOutput.getRows(); entryInd++)
		{
				cost+=((FDMatrix)desiredOutput).data[entryInd]*Math.log(((FDMatrix)networkOutput).data[entryInd])
						+(1-((FDMatrix)desiredOutput).data[entryInd])*Math.log(1-((FDMatrix)networkOutput).data[entryInd]);
		}
		cost/=networkOutput.getRows();
		cost=-cost;
		return (float)cost;
	}

	@Override
	public Matrix getCostDerivative(Matrix input, Matrix networkOutput, Matrix desiredOutput) 
	{
		return networkOutput.msub(desiredOutput);
	}
	*/
    
    @Override
    public float getCost(Matrix[] inputs, Matrix[] networkOutput, Matrix[] desiredOutput) 
	{
    	if(FDMatrix.GPU)
    	{
	    	float cost=0.0f;
			for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
			{
				((FDMatrix)desiredOutput[outputInd]).sendToGPU();
				((FDMatrix)networkOutput[outputInd]).sendToGPU();
				
				int maxThreads=128;
		    	int maxBlocks=64;
		    	int numBlocks = getNumBlocks(desiredOutput[outputInd].getLen(), maxBlocks, maxThreads);
		        int numThreads = getNumThreads(desiredOutput[outputInd].getLen(), maxBlocks, maxThreads);
		        
		        int sharedMemSize = numThreads * Sizeof.FLOAT;
		        if (numThreads <= 32) 
		        {
		            sharedMemSize *= 2;
		        }
		        
		        Pointer kernelParameters = Pointer.to(
		            Pointer.to(((FDMatrix)desiredOutput[outputInd]).gpuPointer),
		            Pointer.to(new int[]{desiredOutput[outputInd].getLen()}),
		            Pointer.to(((FDMatrix)networkOutput[outputInd]).gpuPointer),
		            Pointer.to(((FDMatrix)networkOutput[outputInd]).gpuPointer)
		        );
	
		        // Call the kernel function.
		        cuLaunchKernel(crossEntropyCost,
		            numBlocks,  1, 1,         // Grid dimension
		            numThreads, 1, 1,         // Block dimension
		            sharedMemSize, null,   // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
		        cost+=((FDMatrix)networkOutput[outputInd]).getSum();
			}
			return -cost;
    	}
    	else
    	{
	    	float cost=0.0f;
	    	for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
			{
				for(int costInd=0; costInd<networkOutput[outputInd].getLen(); costInd++)
				{
					cost+=desiredOutput[outputInd].get(costInd, 0)*Math.log(networkOutput[outputInd].get(costInd, 0)+0.0001f)
							+(1.0001f-desiredOutput[outputInd].get(costInd, 0))*Math.log(1.0001f-networkOutput[outputInd].get(costInd, 0));
				}
			}
			return -cost;
    	}
	}
	
	@Override
	public Matrix[] getCostDerivative(Matrix[] inputs, Matrix[] networkOutput,
			Matrix[] desiredOutput, Matrix[] results) 
	{		
		if(FDMatrix.GPU)
		{
			for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
			{
				((FDMatrix)desiredOutput[outputInd]).sendToGPU();
				((FDMatrix)networkOutput[outputInd]).sendToGPU();
				
				int maxThreads=128;
		    	int maxBlocks=64;
		    	int numBlocks = getNumBlocks(desiredOutput[outputInd].getLen(), maxBlocks, maxThreads);
		        int numThreads = getNumThreads(desiredOutput[outputInd].getLen(), maxBlocks, maxThreads);
		        
		        int sharedMemSize = numThreads * Sizeof.FLOAT;
		        if (numThreads <= 32) 
		        {
		            sharedMemSize *= 2;
		        }
		        
		        Pointer kernelParameters = Pointer.to(
		            Pointer.to(((FDMatrix)desiredOutput[outputInd]).gpuPointer),
		            Pointer.to(new int[]{desiredOutput[outputInd].getLen()}),
		            Pointer.to(((FDMatrix)networkOutput[outputInd]).gpuPointer),
		            Pointer.to(((FDMatrix)results[outputInd]).gpuPointer)
		        );
	
		        // Call the kernel function.
		        cuLaunchKernel(crossEntropyCostDerivative,
		            numBlocks,  1, 1,         // Grid dimension
		            numThreads, 1, 1,         // Block dimension
		            sharedMemSize, null,   // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
			}
			return results;
		}
		else
		{
			for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
			{
				for(int costInd=0; costInd<networkOutput[outputInd].getLen(); costInd++)
				{
					results[outputInd].set(costInd, 0, 
							-((0.00001f+desiredOutput[outputInd].get(costInd, 0))/((networkOutput[outputInd].get(costInd, 0))+0.00001f)
							-(1.00001f-desiredOutput[outputInd].get(costInd, 0))/(1.00001f-networkOutput[outputInd].get(costInd, 0))));
					if(Float.isNaN(-((0.00001f+desiredOutput[outputInd].get(costInd, 0))/((networkOutput[outputInd].get(costInd, 0))+0.00001f)
							-(1.00001f-desiredOutput[outputInd].get(costInd, 0))/(1.00001f-networkOutput[outputInd].get(costInd, 0)))))
					{
						int u=0;
					}
				}
			}
			return results;
		}
		/*
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
        //cuCtxSynchronize();
            
        return input;
        */
	}
    
    private static void init()
    {
        context = Sigmoid.context;
        prepare();
    }
    
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxSigmoidFileName = null;
        try
        {
        	ptxSigmoidFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/costFunctions/crossEntropyCost.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        String ptxSigmoidDerivativeFileName = null;
        try
        {
        	ptxSigmoidDerivativeFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/costFunctions/crossEntropyCostDerivative.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module=new CUmodule();
        
        // Obtain a function pointer to the "reduce" function.
        cuModuleLoad(module, ptxSigmoidFileName);
        crossEntropyCost=new CUfunction();
        cuModuleGetFunction(crossEntropyCost, module, "crossEntropyCost");
        
        cuModuleLoad(module, ptxSigmoidDerivativeFileName);
        crossEntropyCostDerivative=new CUfunction();
        cuModuleGetFunction(crossEntropyCostDerivative, module, "crossEntropyCostDerivative");
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

}
