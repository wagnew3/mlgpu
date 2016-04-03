package filters;

import java.io.Serializable;

import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;

public class ScaleFilter implements Serializable
{
	
	double offset=0.0;
	double scaleFactor=1.0;
	
	public ScaleFilter()
	{
		
	}
	
	public Matrix[] scaleData(Matrix[] data)
	{
		Matrix[] scaledData=new Matrix[data.length];
		for(int dataInd=0; dataInd<data.length; dataInd++)
		{
			Matrix scaledDataPart=new FDMatrix(new float[data[dataInd].getRows()], data[dataInd].getRows(), 1);
			for(int rowInd=0; rowInd<data[dataInd].getRows(); rowInd++)
			{
				scaledDataPart.set(rowInd, 0, (float)((data[dataInd].get(rowInd, 0)+offset)*scaleFactor));
			}
			scaledData[dataInd]=scaledDataPart;
		}
		return scaledData;
	}
	
	public Matrix[] unScaleData(Matrix[] data)
	{
		Matrix[] unScaledData=new Matrix[data.length];
		for(int dataInd=0; dataInd<data.length; dataInd++)
		{
			Matrix unScaledDataPart=new FDMatrix(new float[data[dataInd].getRows()], data[dataInd].getRows(), 1);
			for(int rowInd=0; rowInd<data[dataInd].getRows(); rowInd++)
			{
				unScaledDataPart.set(rowInd, 0, (float)((data[dataInd].get(rowInd, 0)/scaleFactor)-offset));
			}
			unScaledData[dataInd]=unScaledDataPart;
		}
		return unScaledData;
	}
	
	public Matrix[] scaleData(Matrix[] data, boolean setScale)
	{
		return scaleData(new Matrix[][]{data}, setScale)[0];
	}
	
	public Matrix[][] scaleData(Matrix[][] data, boolean setScale)
	{
		double maximum=Double.MIN_VALUE;
		double minimum=Double.MAX_VALUE;
		
		for(Matrix[] dataGroup: data)
		{
			for(Matrix vector: dataGroup)
			{
				for(int entryInd=0; entryInd<vector.getRows(); entryInd++)
				{
					if(vector.get(entryInd, 0)>maximum)
					{
						maximum=vector.get(entryInd, 0);
					}
					if(vector.get(entryInd, 0)<minimum)
					{
						minimum=vector.get(entryInd, 0);
					}
				}
			}
		}
		
		if(maximum>minimum)
		{
			if(setScale)
			{
				offset=-minimum;
				scaleFactor=1.0/(maximum-minimum);
			}
			
			Matrix[][] scaledData=new Matrix[data.length][];
			for(int vectorInd=0; vectorInd<scaledData.length; vectorInd++)
			{
				scaledData[vectorInd]=scaleData(data[vectorInd]);
			}
			return scaledData;
		}
		else
		{
			if(setScale)
			{
				offset=0.0;
				scaleFactor=1.0;
			}
		}
		return data;
	}

}
