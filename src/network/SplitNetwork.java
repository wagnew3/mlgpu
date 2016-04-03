package network;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.util.HashMap;

import layer.BLayer;
import layer.Layer;
import nDimensionalMatrices.Matrix;

public abstract class SplitNetwork implements Serializable
{
	
	public BLayer[][] layers;
	
	public SplitNetwork(BLayer[][] layers)
	{
		this.layers=layers;
	}
	
	public BLayer[][] getLayers()
	{
		return layers;
	}
	
	public abstract Matrix[] getOutput(Matrix[] inputs);
	
	public abstract HashMap<BLayer, Matrix>[] getOutputs(Matrix[] inputs, HashMap<BLayer, Matrix>[] outputs);
	
	public abstract Network clone();
	
	public static void saveNetwork(File saveLocation, SplitNetwork network)
	{
		ByteArrayOutputStream bOut=new ByteArrayOutputStream();
        ObjectOutputStream oOut;
		try 
		{
			oOut = new ObjectOutputStream(bOut);
			oOut.writeUnshared(network);
	        oOut.close();
	        Files.write(saveLocation.toPath(), bOut.toByteArray());
	        bOut.close();
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	public static SplitNetwork loadNetwork(File saveLocation)
	{
		
		try 
		{
			byte[] networkBytes=Files.readAllBytes(saveLocation.toPath());
			ByteArrayInputStream bIn=new ByteArrayInputStream(networkBytes);
	        ObjectInputStream oIn=new ObjectInputStream(bIn);       
	        SplitNetwork network=(SplitNetwork)oIn.readObject();
	        oIn.close();
	        bIn.close();
	        return network;
		} 
		catch (IOException | ClassNotFoundException e) 
		{
			e.printStackTrace();
		}
		return null;
	}

}
