package filters;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;

import nDimensionalMatrices.FDMatrix;
import nDimensionalMatrices.Matrix;
import network.SplitNetwork;

public class ScaleFilter implements Serializable
{
	
	double offset=0.0;
	public double scaleFactor=1.0;
	
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
	
	public Matrix[][] scaleData(Matrix[][] data)
	{
		for(int matInd=0; matInd<data.length; matInd++)
		{
			data[matInd]=scaleData(data[matInd]);
		}
		return data;
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
	
	public static void saveFilter(File saveLocation, ScaleFilter filter)
	{
		ByteArrayOutputStream bOut=new ByteArrayOutputStream();
        ObjectOutputStream oOut;
		try 
		{
			oOut=new ObjectOutputStream(bOut);
			oOut.writeUnshared(filter);
	        oOut.close();
	        Files.write(saveLocation.toPath(), bOut.toByteArray());
	        bOut.close();
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	public static ScaleFilter loadFilter(File saveLocation)
	{
		
		try 
		{
			byte[] networkBytes=Files.readAllBytes(saveLocation.toPath());
			ByteArrayInputStream bIn=new ByteArrayInputStream(networkBytes);
	        ObjectInputStream oIn=new ObjectInputStream(bIn);       
	        ScaleFilter filter=(ScaleFilter)oIn.readObject();
	        oIn.close();
	        bIn.close();
	        return filter;
		} 
		catch (IOException | ClassNotFoundException e) 
		{
			e.printStackTrace();
		}
		return null;
	}

}
