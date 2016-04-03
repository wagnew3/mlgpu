package nDimensionalMatrices;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasPointerMode.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;
import jcuda.jcublas.*;
import jcuda.runtime.JCuda;

public class Test 
{
	
	public static void main(String[] args)
	{
		//testFDMatrixMult(1000);
		//testFDMatrixSub(4);
		//testFDMatrixVecScale(2);
		testDotProduct();
	}
	
	public static void testFDMatrixMult(int matSize)
	{
		float[][] floatA=new float[matSize][matSize];
		fillFloatssWithRandoms(floatA);
		
		float[][] floatB=new float[matSize][matSize];
		fillFloatssWithRandoms(floatB);
		
		Matrix matA=new FDMatrix(floatA);
		Matrix matB=new FDMatrix(floatB);
		
		long time=System.nanoTime();
		Matrix matC=matA.mmult(matB);
		time=System.nanoTime()-time;
		System.out.println(time);
		int u=0;
	}
	
	private static void fillFloatssWithRandoms(float[][] floatss)
	{
		for(int rowInd=0; rowInd<floatss.length; rowInd++)
		{
			for(int colInd=0; colInd<floatss[rowInd].length; colInd++)
			{
				floatss[rowInd][colInd]=(float)Math.random();
			}
		}
	}
	
	public static void testFDMatrixSub(int matSize)
	{
		float[][] floatA=new float[1][matSize];
		fillFloatssWithRandoms(floatA);
		
		float[][] floatB=new float[1][matSize];
		fillFloatssWithRandoms(floatB);
		
		Matrix matA=new FDMatrix(floatA);
		Matrix matB=new FDMatrix(floatB);
		
		long time=System.nanoTime();
		Matrix matC=matA.msub(matB);
		time=System.nanoTime()-time;
		System.out.println(time);
		int u=0;
	}
	
	public static void testFDMatrixVecScale(int matSize)
	{
		float[][] floatA=new float[matSize][matSize];
		fillFloatssWithRandoms(floatA);
		
		float[][] floatB=new float[matSize][1];
		fillFloatssWithRandoms(floatB);
		
		Matrix matA=new FDMatrix(floatA);
		Matrix matB=new FDMatrix(floatB);
		
		long time=System.nanoTime();
		Matrix matC=matA.matVecMultScale(matA, matB, 2.0f);
		time=System.nanoTime()-time;
		System.out.println(time);
		int u=0;
	}

	public static void testDotProduct()
	{
		// Enable exceptions and omit subsequent error checks
        JCublas2.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);

        // Create the input data: A vector containing the
        // value 1.0 exactly n times.
        int n = 10;
        float hostData[] = new float[n];
        for (int i=0; i<n; i++)
        {
            hostData[i] = 0.5f;
        }

        // Allocate device memory, and copy the input data to the device
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);
        cudaMemcpy(deviceData, Pointer.to(hostData), 
            n * Sizeof.FLOAT, cudaMemcpyHostToDevice);

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);
        
        // Execute the 'dot' function in DEVICE pointer mode:
        // The result will be written to a pointer that
        // points to device memory.

        // Set the pointer mode to DEVICE
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

        // Prepare the pointer for the result in DEVICE memory
        Pointer deviceResultPointer = new Pointer();
        cudaMalloc(deviceResultPointer, Sizeof.FLOAT);

        // Execute the 'dot' function 
        long beforeDeviceCall = System.nanoTime();
        cublasSdot(handle, n, deviceData, 1, deviceData, 1, deviceResultPointer);
        long afterDeviceCall = System.nanoTime();

        // Synchronize in order to wait for the result to
        // be available (note that this is done implicitly
        // when cudaMemcpy is called)
        cudaDeviceSynchronize();
        long afterDeviceSync = System.nanoTime();

        // Copy the result from the device to the host
        float deviceResult[] = { -1.0f };
        cudaMemcpy(Pointer.to(deviceResult), deviceResultPointer, 
            Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        
        // Print the result and timing information
        double deviceCallDuration = (afterDeviceCall-beforeDeviceCall)/1e6;
        double deviceFullDuration = (afterDeviceSync-beforeDeviceCall)/1e6;
        System.out.println("Device call duration: "+deviceCallDuration+" ms");
        System.out.println("Device full duration: "+deviceFullDuration+" ms");
        System.out.println("Result: "+deviceResult[0]);
        
        
        // Clean up
        cudaFree(deviceData);
        cublasDestroy(handle);
	}
}
