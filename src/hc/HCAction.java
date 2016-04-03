package hc;

import java.util.Arrays;

import ReinforcementMachineLearningFramework.Action;

public class HCAction extends Action
{

	int[] move;
	
	public HCAction(int[] move)
	{
		this.move=move;
	}
	
	@Override
	public int hashCode() 
	{
		return Arrays.hashCode(move);
	}

	@Override
	public boolean equals(Object other) 
	{
		if(!(other instanceof HCAction))
		{
			return false;
		}
		else
		{
			return Arrays.equals(move, ((HCAction)other).move);
		}
	}

	@Override
	public double[] getValue() 
	{
		double[] value=new double[move.length];
		for(int ind=0; ind<value.length; ind++)
		{
			value[ind]=move[ind];
		}
		return value;
	}

	@Override
	public Action getFromNNValue(double[] value) 
	{
		int[] newMove=new int[value.length];
		for(int ind=0; ind<value.length; ind++)
		{
			newMove[ind]=(int)Math.abs(Math.round(value[ind]));
		}
		return new HCAction(newMove);
	}

}
