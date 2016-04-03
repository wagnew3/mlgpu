package layer;

import java.util.Arrays;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class SparseArrayRealVector extends ArrayRealVector
{
	
	public double[] nonZeroEntries;
	public double[] sparseData;
	protected int dimension;
	
	public SparseArrayRealVector(double[] data)
	{
		dimension=data.length;
		int numberZeros=0;
		for(int dataInd=0; dataInd<data.length; dataInd++)
		{
			if(data[dataInd]!=0)
			{
				numberZeros++;
			}
		}
		
		int ind=0;
		sparseData=new double[numberZeros];
		nonZeroEntries=new double[numberZeros];
		for(int dataInd=0; dataInd<data.length; dataInd++)
		{
			if(data[dataInd]!=0)
			{
				sparseData[ind]=data[dataInd];
				nonZeroEntries[ind]=dataInd;
				ind++;
			}
		}
	}
	
	public int getNumberZeroEntries()
	{
		return getDataRef().length-nonZeroEntries.length;
	}
	
	public double getSparseData(int ind)
	{
		return sparseData[ind];
	}
	
	@Override
    public RealVector getSubVector(int index, int n)
	{
		int startLoc=Arrays.binarySearch(nonZeroEntries, index);
		if(startLoc<0)
		{
			startLoc=-(startLoc+1);
		}
		
		int endLoc=Arrays.binarySearch(nonZeroEntries, index+n);
		if(endLoc<0)
		{
			endLoc=-(endLoc+1);
		}
		
		
		
		double[] newSparseData=new double[endLoc-startLoc];
		System.arraycopy(sparseData, startLoc, newSparseData, 0, newSparseData.length);
		double[] newNonZeroEntries=new double[endLoc-startLoc];
		System.arraycopy(nonZeroEntries, startLoc, newNonZeroEntries, 0, newNonZeroEntries.length);
		
		for(int entryInd=0; entryInd<newNonZeroEntries.length; entryInd++)
		{
			newNonZeroEntries[entryInd]-=index;
		}
		
		SparseArrayRealVector newVector=new SparseArrayRealVector(new double[0]);
		newVector.nonZeroEntries=newNonZeroEntries;
		newVector.sparseData=newSparseData;
		newVector.dimension=n;
		return newVector;
	}
	
	public static ArrayRealVector add(ArrayRealVector vectorA, SparseArrayRealVector vectorB)
	{
		for(int entryInd=0; entryInd<vectorB.nonZeroEntries.length; entryInd++)
		{
			vectorA.setEntry((int)vectorB.nonZeroEntries[entryInd],
					vectorA.getEntry((int)vectorB.nonZeroEntries[entryInd])+vectorB.getSparseData(entryInd));
		}
		return vectorA;
	}
	
	public static BlockRealMatrix add(BlockRealMatrix vectorA, RealMatrix sparseMatB)
	{
		for(int entryInd=0; entryInd<sparseMatB.getColumnDimension(); entryInd++)
		{
			vectorA.setEntry(0, (int)sparseMatB.getEntry(0, entryInd),
					vectorA.getEntry(0, (int)sparseMatB.getEntry(0, entryInd))+sparseMatB.getEntry(1, entryInd));
		}
		return vectorA;
	}
	
	@Override
	public ArrayRealVector append(ArrayRealVector v) 
	{
        SparseArrayRealVector sparseV=(SparseArrayRealVector)v;
        double[] newEntriesIndices=new double[nonZeroEntries.length+sparseV.nonZeroEntries.length];
        System.arraycopy(nonZeroEntries, 0, newEntriesIndices, 0, nonZeroEntries.length);
        System.arraycopy(sparseV.nonZeroEntries, 0, newEntriesIndices, nonZeroEntries.length, sparseV.nonZeroEntries.length);
        for(int newEntryInd=nonZeroEntries.length; newEntryInd<newEntriesIndices.length; newEntryInd++)
        {
        	newEntriesIndices[newEntryInd]+=dimension;
        }
        double[] newSparseData=new double[sparseData.length+sparseV.sparseData.length]; 
        System.arraycopy(sparseData, 0, newSparseData, 0, sparseData.length);
        System.arraycopy(sparseV.sparseData, 0, newSparseData, sparseData.length, sparseV.sparseData.length);
        
        nonZeroEntries=newEntriesIndices;
        sparseData=newSparseData;
        dimension=dimension+sparseV.dimension;
        
        return this;
    }
	
	@Override
    public int getDimension() 
	{
        return dimension;
    }

}
