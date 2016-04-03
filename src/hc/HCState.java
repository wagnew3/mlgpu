package hc;

import java.util.Arrays;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.State;

public class HCState extends State
{
	
	int[] position;
	int numberMustBeZero;
	
	public HCState(int[] position, int numberMustBeZero)
	{
		this.position=position;
		this.numberMustBeZero=numberMustBeZero;
	}

	@Override
	public int hashCode() 
	{
		return Arrays.hashCode(position);
	}

	@Override
	public boolean equals(Object other) 
	{
		if(!(other instanceof HCState))
		{
			return false;
		}
		else
		{
			return Arrays.equals(position, ((HCState)other).position);
		}
	}

	@Override
	public boolean isEndState() 
	{
		for(int ind=0; ind<numberMustBeZero; ind++)
		{
			if(position[ind]>0)
			{
				return false;
			}
		}
		return true;
	}
	
	@Override
	public double[] getValue() 
	{
		double[] value=new double[position.length];
		for(int ind=0; ind<value.length; ind++)
		{
			value[ind]=position[ind];
		}
		return value;
	}
	
	public int[] getValueInt() 
	{
		return position;
	}

	@Override
	public State getFromNNValue(double[] value) 
	{
		int[] newPosition=new int[value.length];
		for(int ind=0; ind<value.length; ind++)
		{
			newPosition[ind]=(int)Math.abs(Math.round(value[ind]));
		}
		return new HCState(newPosition, numberMustBeZero);
	}

}
