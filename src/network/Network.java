package network;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealVector;

import activationFunctions.ActivationFunction;
import filters.ScaleFilter;
import layer.Layer;

public abstract class Network implements Serializable
{
	
	protected Layer[] layers;
	
	public Network(Layer[] layers)
	{
		this.layers=layers;
	}
	
	public Layer[] getLayers()
	{
		return layers;
	}
	
	public abstract ArrayRealVector getOutput(ArrayRealVector input);
	
	public abstract Network clone();
	
	public static void saveNetwork(File saveLocation, Network network)
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
	
	public static Network loadNetwork(File saveLocation)
	{
		
		try 
		{
			byte[] networkBytes=Files.readAllBytes(saveLocation.toPath());
			ByteArrayInputStream bIn=new ByteArrayInputStream(networkBytes);
	        ObjectInputStream oIn=new ObjectInputStream(bIn);       
	        Network network=(Network)oIn.readObject();
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
