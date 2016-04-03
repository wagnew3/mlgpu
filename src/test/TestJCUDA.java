package test;

/*
 * JCublas - Java bindings for CUBLAS, the NVIDIA CUDA BLAS library,
 * to be used with JCuda <br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2011 Marco Hutter - http://www.jcuda.org
 */

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.runtime.JCuda.*;

import java.util.Random;

import jcuda.*;
import jcuda.jcublas.cublasHandle;
import jcuda.jcusparse.JCusparse;
import jcuda.runtime.JCuda;
import nDimensionalMatrices.SparseFMatrix;

/**
 * This is a sample class demonstrating the application of JCublas2 for
 * performing a BLAS 'sgemm' operation, i.e. for computing the matrix <br />
 * C = alpha * A * B + beta * C <br />
 * for single-precision floating point values alpha and beta, and matrices A, B
 * and C of size 1000x1000.
 */
public class TestJCUDA
{
    
	public static void main(String args[])
    {
		JCuda.setExceptionsEnabled(true);
        JCusparse.setExceptionsEnabled(true);
        
        //testSgemm(1000);
        testSparse();
    }
	
	public static void testSparse()
	{
		float testData[] = createRandomFloatData(1000);
		for(int i=0; i<750; i++)
		{
			int ind=(int)Math.floor(testData.length*Math.random());
			testData[ind]=0.0f;
		}
		
		SparseFMatrix sfMatrix=new SparseFMatrix(new float[][]{testData});
		sfMatrix.sendToGPU();
		int u=0;
	}

    /**
     * Test the JCublas sgemm operation for matrices of size n x x
     * 
     * @param n The matrix size
     */
    public static void testSgemm(int n)
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        int nn = n * n;

        System.out.println("Creating input data...");
        float h_A[] = createRandomFloatData(nn);
        //h_A=new float[]{1.0f, 0.0f, 0.0f, 1.0f};
        float h_B[] = createRandomFloatData(nn);
        //h_B=new float[]{1.0f, 0.0f, 0.0f, 1.0f};
        float h_C[] = createRandomFloatData(nn);
        //h_C=new float[1];
        float h_C_ref[] = h_C.clone();

        System.out.println("Performing Sgemm with Java...");
        long time=System.nanoTime();
        sgemmJava(n, alpha, h_A, h_B, beta, h_C_ref);
        time=System.nanoTime()-time;
        System.out.println(time);

        System.out.println("Performing Sgemm with JCublas...");
        time=System.nanoTime();
        sgemmJCublas(n, alpha, h_A, h_B, beta, h_C);
        time=System.nanoTime()-time;
        System.out.println(time);

        boolean passed = isCorrectResult(h_C, h_C_ref);
        System.out.println("testSgemm "+(passed?"PASSED":"FAILED")); 
    }

    /**
     * Implementation of sgemm using JCublas
     */
    private static void sgemmJCublas(int n, float alpha, float A[], float B[],
                    float beta, float C[])
    {
        int nn = n * n;

        // Create a CUBLAS handle
        cublasHandle handle = new cublasHandle();
        cublasCreate(handle);

        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cudaMalloc(d_A, nn * Sizeof.FLOAT);
        cudaMalloc(d_B, nn * Sizeof.FLOAT);
        cudaMalloc(d_C, nn * Sizeof.FLOAT);
        
        // Copy the memory from the host to the device
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
        cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

        // Execute sgemm
        Pointer pAlpha = Pointer.to(new float[]{alpha});
        Pointer pBeta = Pointer.to(new float[]{beta});
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
            pAlpha, d_A, n, d_B, n, pBeta, d_C, n);

        // Copy the result from the device to the host
        cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    /**
     * Simple implementation of sgemm, using plain Java
     */
    private static void sgemmJava(int n, float alpha, float A[], float B[],
                    float beta, float C[])
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float prod = 0;
                for (int k = 0; k < n; ++k)
                {
                    prod += A[k * n + i] * B[j * n + k];
                }
                C[j * n + i] = alpha * prod + beta * C[j * n + i];
            }
        }
    }


    /**
     * Creates an array of the specified size, containing some random data
     */
    private static float[] createRandomFloatData(int n)
    {
        Random random = new Random();
        float x[] = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = random.nextFloat();
        }
        return x;
    }

    /**
     * Compares the given result against a reference, and returns whether the
     * error norm is below a small epsilon threshold
     */
    private static boolean isCorrectResult(float result[], float reference[])
    {
        float errorNorm = 0;
        float refNorm = 0;
        for (int i = 0; i < result.length; ++i)
        {
            float diff = reference[i] - result[i];
            errorNorm += diff * diff;
            refNorm += reference[i] * result[i];
        }
        errorNorm = (float) Math.sqrt(errorNorm);
        refNorm = (float) Math.sqrt(refNorm);
        if (Math.abs(refNorm) < 1e-6)
        {
            return false;
        }
        return (errorNorm / refNorm < 1e-6f);
    }
    
}