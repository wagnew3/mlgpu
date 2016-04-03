package nDimensionalMatrices;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.*;
import static jcuda.jcublas.cublasPointerMode.*;
import static jcuda.runtime.cudaError.cudaErrorMemoryAllocation;
import static jcuda.runtime.JCuda.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasHandle;
import learningRule.MPBackPropGradientDescent;

public class FDMatrix extends Matrix
{
	
	protected int rows;
	protected int columns;
	public float[] data;
	protected boolean inGPU;
	public Pointer gpuPointer;
	protected static cublasHandle handle;
	
	static
	{
		handle=new cublasHandle();
        cublasCreate(handle);
        JCublas.cublasInit();
	}
	
	public FDMatrix(float[] newData, int rows, int columns)
	{
		super(newData);
		this.rows=rows;
		this.columns=columns;
		this.data=newData;
		inGPU=false;
		gpuPointer=null;
	}
	
	public FDMatrix(float[][] newData)
	{
		super(newData);
		
		rows=newData.length;
		columns=newData[0].length;
		
		this.data=new float[newData.length*newData[0].length];
		
		for(int colInd=0; colInd<newData[0].length; colInd++)
		{
			for(int rowInd=0; rowInd<newData.length; rowInd++)
			{
				data[colInd*newData.length+rowInd]=newData[rowInd][colInd];
			}
		}
		inGPU=false;
		gpuPointer=null;
	}
	
	public FDMatrix(int rows, int columns)
	{
		this.rows=rows;
		this.columns=columns;
		data=null;
		inGPU=false;
		sendToGPU();
	}
	
	public void sendToGPU()
	{
		if(!inGPU)
		{
			gpuPointer=GPUMemoryManager.alloc(getLen()*Sizeof.FLOAT);
			inGPU=true;
	        if(data!=null)
	        {
	        	cublasSetVector(getLen(), Sizeof.FLOAT, Pointer.to(data), 1, gpuPointer, 1);
	        }
	        else
	        {
	        	cudaMemset(gpuPointer, 0, getLen()*Sizeof.FLOAT);
	        	//cublasSetVector(getLen(), Sizeof.FLOAT, Pointer.to(new float[getLen()]), 1, gpuPointer, 1);
	        	//omscal(0.0f);
	        }
	        data=null;
		}
	}
	
	public void getFromGPU()
	{
		if(inGPU)
		{
			data=new float[rows*columns];
			cublasGetVector(rows*columns, Sizeof.FLOAT, gpuPointer, 1, Pointer.to(data), 1);
			GPUMemoryManager.free(gpuPointer);
	        gpuPointer=null;
	        inGPU=false;
		}
	}
	
	protected void setGPUPointer(Pointer gpuDataPointer)
	{
		inGPU=true;
		if(gpuPointer!=null
				&& !gpuPointer.equals(gpuDataPointer))
		{
			GPUMemoryManager.free(gpuPointer);
			GPUMemoryManager.incUsages(gpuDataPointer);
		}		
		gpuPointer=gpuDataPointer;
        data=null;
	}
	
	@Override
	public void clear()
	{
		if(inGPU)
		{
			GPUMemoryManager.free(gpuPointer);
	        gpuPointer=null;
	        inGPU=false;
		}
	}
	
	@Override
	public void finalize()
	{
		if(inGPU)
		{
			GPUMemoryManager.free(gpuPointer);
	        gpuPointer=null;
	        inGPU=false;
		}
	}
	
	@Override
	public int getRows()
	{
		return rows;
	}
	
	@Override
	public int getCols()
	{
		return columns;
	}
	
	@Override
	public int getLen()
	{
		return rows*columns;
	}
	
	@Override
	public float get(int row, int col) 
	{
		getFromGPU();
		return data[col*rows+row];
	}

	@Override
	public void set(int row, int col, float val) 
	{
		getFromGPU();
		data[col*rows+row]=val;
	}
	
	@Override
	public  float[] getData()
	{
		getFromGPU();
		return data;
	}
	
	@Override
	public Matrix getSubVector(int offset, int length)
	{
		getFromGPU();
		float[] subVectorData=new float[length];
		System.arraycopy(data, offset, subVectorData, 0, subVectorData.length);
		if(rows==1)
		{
			return new FDMatrix(subVectorData, 1, subVectorData.length);
		}
		else
		{
			return new FDMatrix(subVectorData, subVectorData.length, 1);
		}
	}
	
	@Override
	public Matrix otrans() 
	{
		if(!(rows==1 || columns==1))
		{
			System.out.println("tanspose not yet implemented for non-vectors!");
		}
		int tempColumns=columns;
		columns=rows;
		rows=tempColumns;
		return this;
	}
	
	@Override
	public Matrix append(Matrix toAppend)
	{
		getFromGPU();
		float[] newData=new float[data.length+((FDMatrix)toAppend).data.length];
		System.arraycopy(data, 0, newData, 0, data.length);
		System.arraycopy(((FDMatrix)toAppend).data, 0, newData, data.length, ((FDMatrix)toAppend).data.length);
		return new FDMatrix(newData, newData.length, 1);
	}

	@Override
	public Matrix mmult(Matrix toMultiplyBy) 
	{
		Matrix matC=new FDMatrix(rows, ((FDMatrix)toMultiplyBy).columns);
		return sgemm(false, false, toMultiplyBy, 1.0f, 0.0f, matC, matC);
	}

	@Override
	public Matrix ommult(Matrix toMultiplyBy) 
	{
		return sgemm(false, false, toMultiplyBy, 1.0f, 0.0f, this, this);
	}
	
	@Override
	public Matrix oebemult(Matrix multVec)
	{
		return sbmv('u', 0, 1.0f, this, multVec, 0.0f, this);
		
		/*
		getFromGPU();
		float[] resultData=new float[data.length];
		if(multVec.getCols()==1)
		{
			for(int rowInd=0; rowInd<multVec.getRows(); rowInd++)
			{
				resultData[rowInd]=get(rowInd, 0)*multVec.get(rowInd, 0);
			}
			return new FDMatrix(resultData, resultData.length, 1);
		}
		else
		{
			for(int colInd=0; colInd<multVec.getCols(); colInd++)
			{
				resultData[colInd]=get(0, colInd)*multVec.get(0, colInd);
			}
			return new FDMatrix(resultData, 1, resultData.length);
		}
		*/
	}

	@Override
	public Matrix mscal(float toScaleBy)
	{
		Matrix matC=new FDMatrix(rows, columns);
		return null;//scal(toScaleBy, 1, Matrix result);
	}

	@Override
	public Matrix omscal(float toScaleBy) 
	{
		return scal(toScaleBy, 1, this);
	}

	@Override
	public Matrix mad(Matrix toAddTo) 
	{
		Matrix resMat=new FDMatrix(rows, columns);
		return saxpy(1.0f, toAddTo, 1, this, 1, resMat);
	}

	@Override
	public Matrix omad(Matrix toAddTo)
	{
		return saxpy(1.0f, toAddTo, 1, this, 1, this);
	}
	
	@Override
	public Matrix madScale(Matrix toAddTo, float scaleAddBy)
	{
		Matrix resMat=new FDMatrix(rows, columns);
		return saxpy(scaleAddBy, toAddTo, 1, this, 1, resMat);
	}
	
	@Override
	public Matrix omadScale(Matrix toAddTo, float scaleAddBy)
	{
		return saxpy(scaleAddBy, toAddTo, 1, this, 1, this);
	}

	@Override
	public Matrix msub(Matrix toSubtractBy, Matrix result) 
	{
		this.sendToGPU();
		((FDMatrix)result).sendToGPU();
		cublasScopy(handle, getLen(), gpuPointer, 1, ((FDMatrix)result).gpuPointer, 1);
		return saxpy(-1.0f, toSubtractBy, 1, result, 1, result);
	}

	@Override
	public Matrix omsub(Matrix toSubtractBy)
	{
		return saxpy(-1.0f, toSubtractBy, 1, this, 1, this);
	}
	
	@Override
	public Matrix msubScale(Matrix toSubtractBy, float scaleSubBy) 
	{
		Matrix resMat=new FDMatrix(rows, columns);
		return saxpy(-scaleSubBy, toSubtractBy, 1, this, 1, resMat);
	}

	@Override
	public Matrix omsubScale(Matrix toSubtractBy, float scaleSubBy)
	{
		return saxpy(-scaleSubBy, toSubtractBy, 1, this, 1, this);
	}
	
	@Override
	public float dot(Matrix toDotWith) 
	{
		return sdot(this, 1, toDotWith, 1);
	}
	
	@Override
	public Matrix matVecMultScale(Matrix mat, Matrix vec, float scaleSubBy)
	{
		Matrix result=new FDMatrix(mat.getRows(), 1);
		return sgemv(false, scaleSubBy, mat, vec, 1, 0.0f, result, 1, false, result);
	}
	
	@Override
	public Matrix omatVecMultScale(Matrix mat, Matrix vec, float scaleSubBy)
	{
		return sgemv(false, scaleSubBy, mat, vec, 1, 0.0f, vec, 1, false, vec);
	}
	
	@Override
	public Matrix matVecMultScaleAdd(Matrix mat, Matrix vecMult, float scaleMultBy, Matrix vecAdd)
	{
		((FDMatrix)vecAdd).sendToGPU();
		Matrix result=new FDMatrix(mat.getRows(), 1);
		sgemv(false, scaleMultBy, mat, vecMult, 1, 0.0f, result, 1, true, result);
		return saxpy(1.0f, vecAdd, 1, result, 1, result);
		//return sgemv(false, scaleSubBy, mat, vecMult, 1, 1.0f, vecAdd, 1, true, result);
	}
	
	@Override
	public Matrix omatVecMultScaleAdd(Matrix mat, Matrix vecMult, float scaleMultBy, Matrix vecAdd, Matrix result, float scaleResultBy)
	{
		sgemv(false, scaleMultBy, mat, vecMult, 1, scaleResultBy, result, 1, true, result);
		return saxpy(1.0f, vecAdd, 1, result, 1, result);
		
		//return sgemv(false, scaleSubBy, mat, vec, 1, 1.0f, vecAdd, 1, true, result);
	}
	
	@Override
	public Matrix matVecMultScaleAddScale(Matrix mat, Matrix vecMult, float scaleSubBy, Matrix vecAdd, float scaleAddBy)
	{
		((FDMatrix)vecAdd).sendToGPU();
		Matrix result=new FDMatrix(mat.getRows(), 1);
		sgemv(false, scaleSubBy, mat, vecMult, 1, 0.0f, result, 1, true, result);
		return saxpy(scaleAddBy, vecAdd, 1, result, 1, result);
		//return sgemv(false, scaleSubBy, mat, vecMult, 1, 1.0f, vecAdd, 1, true, result);
	}
	
	@Override
	public Matrix outProd(Matrix vecA, Matrix vecB, Matrix result)
	{
		//return sger(1.0f, vecA, 1, vecB, 1, result);
		return vecA.sgemm(false, true,
				vecB, 1.0f, 0.0f, result, result);
	}

	@Override
	public Matrix sgemm(boolean transposeA, boolean transposeB,
			Matrix toMultiplyBy, float alpha, float beta, Matrix toAdd, Matrix result) 
	{
		//System.out.println("sgemm_start");
		sendToGPU();
		((FDMatrix)toMultiplyBy).sendToGPU();
		((FDMatrix)toAdd).sendToGPU();
		
		/*
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        cudaMalloc(d_A, data.length*Sizeof.FLOAT);
        cudaMalloc(d_B, ((FDMatrix)toMultiplyBy).data.length*Sizeof.FLOAT);
        
        
        cublasSetVector(data.length, Sizeof.FLOAT, Pointer.to(data), 1, d_A, 1);
        cublasSetVector(((FDMatrix)toMultiplyBy).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)toMultiplyBy).data), 1, d_B, 1);
        
        cudaMalloc(d_C, resultLocation.length*Sizeof.FLOAT);
        cublasSetVector(((FDMatrix)toAdd).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)toAdd).data), 1, d_C, 1);
        */
		
        Pointer pAlpha = Pointer.to(new float[]{alpha});
        Pointer pBeta = Pointer.to(new float[]{beta});
        
        if(!transposeA && !transposeB)
        {
	        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, result.getCols(), result.getRows(), columns, 
    	            pAlpha, gpuPointer, rows, ((FDMatrix)toMultiplyBy).gpuPointer, ((FDMatrix)toMultiplyBy).rows, pBeta, ((FDMatrix)toAdd).gpuPointer, ((FDMatrix)toAdd).rows);
        }
        else if(!transposeA && transposeB)
        {
        	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, result.getRows(), result.getCols(), columns, 
    	            pAlpha, gpuPointer, rows, ((FDMatrix)toMultiplyBy).gpuPointer, ((FDMatrix)toMultiplyBy).rows, pBeta, ((FDMatrix)toAdd).gpuPointer, ((FDMatrix)toAdd).rows);
        }
        else if(transposeA && !transposeB)
        {
        	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, result.getRows(), result.getCols(), rows, 
    	            pAlpha, gpuPointer, rows, ((FDMatrix)toMultiplyBy).gpuPointer, ((FDMatrix)toMultiplyBy).rows, pBeta, ((FDMatrix)toAdd).gpuPointer, ((FDMatrix)toAdd).rows);
        }
        else
        {
        	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, result.getRows(), result.getCols(), rows, 
    	            pAlpha, gpuPointer, rows, ((FDMatrix)toMultiplyBy).gpuPointer, ((FDMatrix)toMultiplyBy).rows, pBeta, ((FDMatrix)toAdd).gpuPointer, ((FDMatrix)toAdd).rows);
        }

        /*
        cublasGetVector(resultLocation.length, Sizeof.FLOAT, ((FDMatrix)toAdd).gpuPointer, 1, Pointer.to(resultLocation), 1);
        
        if(!transposeA && !transposeB)
        {
        	return new FDMatrix(resultLocation, rows, ((FDMatrix)toMultiplyBy).columns);
        }
        else if(!transposeA && transposeB)
        {
        	return new FDMatrix(resultLocation, rows, ((FDMatrix)toMultiplyBy).rows);
        }
        else if(transposeA && !transposeB)
        {
        	return new FDMatrix(resultLocation, columns, ((FDMatrix)toMultiplyBy).columns);
        }
        else
        {
        	return new FDMatrix(resultLocation, columns, ((FDMatrix)toMultiplyBy).rows);
        }
        */
        
        ((FDMatrix)result).setGPUPointer(((FDMatrix)toAdd).gpuPointer);
        
        //System.out.println("sgemm_end");
        return result;
	}
	
	public Matrix sbmv(char uplo, int k, float alpha, Matrix matA, Matrix vecX, float beta, 
			Matrix vecY)
	{
		((FDMatrix)matA).sendToGPU();
		((FDMatrix)vecX).sendToGPU();
		((FDMatrix)vecY).sendToGPU();
		
		cublasSsbmv(handle, 0, matA.getRows(), k, Pointer.to(new float[]{alpha}), ((FDMatrix)matA).gpuPointer,
				1, ((FDMatrix)vecX).gpuPointer, 1, Pointer.to(new float[]{beta}),
	               ((FDMatrix)vecY).gpuPointer, 1);
		
		return vecY;
	}
	
	public Matrix sgemv(boolean transpose, float scaleFactor, Matrix matA, 
			Matrix vecX, int incx, float beta, Matrix vecY, int incy, boolean addY, Matrix result)
	{
		//System.out.println("sgemv_start");
		Pointer d_C=null;
		((FDMatrix)matA).sendToGPU();
		((FDMatrix)vecX).sendToGPU();
		if(addY)
		{
			((FDMatrix)vecY).sendToGPU();
			d_C=((FDMatrix)vecY).gpuPointer;
			((FDMatrix)result).setGPUPointer(d_C);
		}
		else
		{
			d_C=((FDMatrix)result).gpuPointer;
		}
		
		/*
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        
        cudaMalloc(d_A, ((FDMatrix)matA).data.length*Sizeof.FLOAT);
        cudaMalloc(d_B, ((FDMatrix)vecX).data.length*Sizeof.FLOAT);
        
        cublasSetVector(((FDMatrix)matA).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)matA).data), 1, d_A, 1);
        cublasSetVector(((FDMatrix)vecX).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)vecX).data), 1, d_B, 1);
		
        if(addY)
        {
        	cublasSetVector(((FDMatrix)vecY).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)vecY).data), 1, d_C, 1);
        }
        else
        {
        	beta=0.0f;
        }
        */
		
		if(!addY)
		{
			beta=0.0f;
		}
        
        Pointer pAlpha = Pointer.to(new float[]{scaleFactor});
        Pointer pBeta = Pointer.to(new float[]{beta});
        
        if(transpose)
        {
        	cublasSgemv(handle, CUBLAS_OP_T, ((FDMatrix)matA).rows, ((FDMatrix)matA).columns, 
        			pAlpha, ((FDMatrix)matA).gpuPointer, ((FDMatrix)matA).rows,
        			((FDMatrix)vecX).gpuPointer, incx, pBeta, d_C, incy);
        }
        else
        {
        	cublasSgemv(handle, CUBLAS_OP_N, ((FDMatrix)matA).rows, ((FDMatrix)matA).columns, 
        			pAlpha, ((FDMatrix)matA).gpuPointer, ((FDMatrix)matA).rows,
        			((FDMatrix)vecX).gpuPointer, incx, pBeta, d_C, incy);
        }
        
        /*
        cublasGetVector(((FDMatrix)result).data.length, Sizeof.FLOAT, d_C, 1, Pointer.to(((FDMatrix)result).data), 1);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
		*/
        
        //System.out.println("sgemv_end");
        
        return result;
	}
	
	public Matrix sger(float alpha, Matrix vecX, int incx, Matrix vecY, int incy, Matrix matA) 
	{
		//System.out.println("sger_start");
		((FDMatrix)vecX).sendToGPU();
		((FDMatrix)vecY).sendToGPU();
		((FDMatrix)matA).sendToGPU();
		
		/*
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_C = new Pointer();
        
        cudaMalloc(d_A, ((FDMatrix)vecX).data.length*Sizeof.FLOAT);
        cudaMalloc(d_B, ((FDMatrix)vecY).data.length*Sizeof.FLOAT);
        cudaMalloc(d_C, ((FDMatrix)matA).data.length*Sizeof.FLOAT);
        
        cublasSetVector(((FDMatrix)vecX).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)vecX).data), 1, d_A, 1);
        cublasSetVector(((FDMatrix)vecY).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)vecY).data), 1, d_B, 1);
        cublasSetVector(((FDMatrix)matA).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)matA).data), 1, d_C, 1);
        */
		
        Pointer pAlpha = Pointer.to(new float[]{alpha});

        cublasSger(handle, matA.getRows(), matA.getCols(), pAlpha, ((FDMatrix)vecX).gpuPointer, 
        		1, ((FDMatrix)vecY).gpuPointer, 1, ((FDMatrix)matA).gpuPointer, matA.getRows());
        
        ((FDMatrix)matA).setGPUPointer(((FDMatrix)matA).gpuPointer);
        
        /*
        cublasGetVector(((FDMatrix)matA).data.length, Sizeof.FLOAT, d_C, 1, 
        		Pointer.to(((FDMatrix)matA).data), 1);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
		*/
        
        //System.out.println("sger_end");
        
        return matA;
	}
	
	public Matrix saxpy(float alpha, Matrix xVec, int incx, Matrix yVec, int incY, Matrix result) 
	{
		//float ans=alpha*xVec.get(0, 0)+yVec.get(0, 0);
		//System.out.println("saxpy_start: "+alpha+"*"+xVec.get(0, 0)+"+"+yVec.get(0, 0)+"="+ans);
		((FDMatrix)xVec).sendToGPU();
		((FDMatrix)yVec).sendToGPU();
		
		/*
        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        cudaMalloc(d_A, ((FDMatrix)xVec).data.length*Sizeof.FLOAT);
        cudaMalloc(d_B, ((FDMatrix)yVec).data.length*Sizeof.FLOAT);
        
        // Copy the memory from the host to the device
        cublasSetVector(((FDMatrix)xVec).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)xVec).data), 1, d_A, 1);
        cublasSetVector(((FDMatrix)yVec).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)yVec).data), 1, d_B, 1);
		*/
		
        Pointer pAlpha = Pointer.to(new float[]{alpha});
        cublasSaxpy(handle, ((FDMatrix)xVec).getLen(), pAlpha, ((FDMatrix)xVec).gpuPointer, 
        		incx, ((FDMatrix)yVec).gpuPointer, incY);

        ((FDMatrix)result).setGPUPointer(((FDMatrix)yVec).gpuPointer);
        /*
        // Copy the result from the device to the host
        cublasGetVector(((FDMatrix)xVec).data.length, Sizeof.FLOAT, d_B, 1, Pointer.to(((FDMatrix)result).data), 1);

        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        */
        /*
        System.out.println("saxpy_end");
        
        if(result.get(0, 0)!=ans)
        {
        	int u=0;
        }
        */
        return result;
	}
	
	public float sdot(Matrix xVec, int incx, Matrix yVec, int incy)
	{
		//System.out.println("sdot_start");
		
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
        
        ((FDMatrix)xVec).sendToGPU();
		((FDMatrix)yVec).sendToGPU();
		
        /*
        // Allocate memory on the device
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        */
        
        Pointer d_C = new Pointer();
        
        /*
        cudaMalloc(d_A, ((FDMatrix)xVec).data.length*Sizeof.FLOAT);
        cudaMalloc(d_B, ((FDMatrix)yVec).data.length*Sizeof.FLOAT);
        */
        
        int res=cudaMalloc(d_C, Sizeof.FLOAT);
        if(res==cudaErrorMemoryAllocation)
        {
        	System.out.println("Out of memory!");
        }
        
        /*
        // Copy the memory from the host to the device
        cublasSetVector(((FDMatrix)xVec).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)xVec).data), 1, d_A, 1);
        cublasSetVector(((FDMatrix)yVec).data.length, Sizeof.FLOAT, Pointer.to(((FDMatrix)yVec).data), 1, d_B, 1);
		*/
        
        cublasSdot(handle, ((FDMatrix)xVec).getLen(), ((FDMatrix)xVec).gpuPointer, incx, 
        		((FDMatrix)yVec).gpuPointer, incy, d_C);

        // Copy the result from the device to the host
        float[] result=new float[1];
        cublasGetVector(1, Sizeof.FLOAT, d_C, 1, Pointer.to(result), 1);

        /*
        // Clean up
        cudaFree(d_A);
        cudaFree(d_B);
        */
        
        cudaFree(d_C);
        
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        
        //System.out.println("sdot_end");
        
        return result[0];
	}
	
	@Override
	public Matrix scal(float alpha, int incx, Matrix result) 
	{
		//System.out.println("scal_start");
		sendToGPU();
		((FDMatrix)result).sendToGPU();

		if(!gpuPointer.equals(((FDMatrix)result).gpuPointer))
		{
			cublasScopy(handle, getLen(), gpuPointer, 1, ((FDMatrix)result).gpuPointer, 1);
		}

		/*
        // Allocate memory on the device
        Pointer d_A = new Pointer();
        
		
        Pointer d_B = new Pointer();
        */
        /*
        cudaMalloc(d_A, data.length*Sizeof.FLOAT);
        */
        /*
        int res=cudaMalloc(d_B, rows*columns*Sizeof.FLOAT);
        if(res==cudaErrorMemoryAllocation)
        {
        	System.out.println("Out of memory!");
        }
        */
        /*
        // Copy the memory from the host to the device
        cublasSetVector(data.length, Sizeof.FLOAT, Pointer.to(data), 1, d_A, 1);
		*/
        
        // Execute sgemm
        Pointer pAlpha = Pointer.to(new float[]{alpha});
    	cublasSscal(handle, rows*columns, pAlpha, ((FDMatrix)result).gpuPointer, incx);

    	/*
        // Copy the result from the device to the host
        cublasGetVector(data.length, Sizeof.FLOAT, d_B, 1, Pointer.to(((FDMatrix)result).data), 1);
		*/
    	
    	/*
        // Clean up
        cudaFree(d_A);
        */
        
       // cudaFree(d_B);
        
        //System.out.printl("scal_end");
        
        return result;
	}
	
	public float getSum()
	{
		float sum=0.0f;
		for(int rowInd=0; rowInd<getRows(); rowInd++)
		{
			for(int colInd=0; colInd<getCols(); colInd++)
			{
				sum+=get(rowInd, colInd);
			}
		}
		return sum;
	}
	
	private static CUcontext context;
    private static CUmodule module;
    private static CUfunction activation;
    private static CUfunction activationDerivative;
    private static CUdeviceptr deviceBuffer;
	
    /*
	static
    {
    	JCudaDriver.setExceptionsEnabled(true);
        init();
    }
	*/
    
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
        String ptxSigmoidFileName = null;
        try
        {
        	ptxSigmoidFileName=preparePtxFile("C:\\Users\\C\\workspace\\mlGPU\\src\\activationFunctions\\sigmoid.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        String ptxSigmoidDerivativeFileName = null;
        try
        {
        	ptxSigmoidDerivativeFileName=preparePtxFile("C:\\Users\\C\\workspace\\mlGPU\\src\\activationFunctions\\sigmoidDerivative.cu");
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
    
    private void writeObject(ObjectOutputStream oos) throws IOException 
    {
    	getFromGPU();
	    oos.defaultWriteObject();
	    oos.writeObject(rows);
	    oos.writeObject(columns);
	    oos.writeObject(data);
	    oos.writeObject(inGPU);
    }

	private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException 
	{
	    ois.defaultReadObject();
	    rows=(int)ois.readObject();
	    columns=(int)ois.readObject();
	    data=(float[])ois.readObject();
	    inGPU=(boolean)ois.readObject();
	}
	
	@Override
	public String toString()
	{
		if(inGPU)
		{
			data=new float[rows*columns];
			cublasGetVector(rows*columns, Sizeof.FLOAT, gpuPointer, 1, Pointer.to(data), 1);
		}
		return Arrays.toString(data);
	}



}
