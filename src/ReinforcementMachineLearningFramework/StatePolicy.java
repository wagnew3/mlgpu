package ReinforcementMachineLearningFramework;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.util.Set;
import java.util.Map.Entry;

import network.Network;

public abstract class StatePolicy implements Serializable
{
	public abstract Action getAction(State state, LimitedActionsEnvironment environment);
	
	public abstract void setStateValue(State state, double value);
	
	public abstract double getStateValue(State state);
	
	public abstract Set<Entry<State, Double>> getStateValues();
	
	public abstract Set<State> getStates();
	
	public static void savePolicy(File saveLocation, StatePolicy policy)
	{
		ByteArrayOutputStream bOut=new ByteArrayOutputStream();
        ObjectOutputStream oOut;
		try 
		{
			oOut=new ObjectOutputStream(bOut);
			oOut.writeUnshared(policy);
	        oOut.close();
	        Files.write(saveLocation.toPath(), bOut.toByteArray());
	        bOut.close();
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	public static StatePolicy loadPolicy(File saveLocation)
	{
		
		try 
		{
			byte[] networkBytes=Files.readAllBytes(saveLocation.toPath());
			ByteArrayInputStream bIn=new ByteArrayInputStream(networkBytes);
	        ObjectInputStream oIn=new ObjectInputStream(bIn);       
	        StatePolicy policy=(StatePolicy)oIn.readObject();
	        oIn.close();
	        bIn.close();
	        return policy;
		} 
		catch (IOException | ClassNotFoundException e) 
		{
			e.printStackTrace();
		}
		return null;
	}
	
}
