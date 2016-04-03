package boardGame;

import java.util.Arrays;

import ReinforcementMachineLearningFramework.Action;
import utils.UnsecureHash;

public class BoardAction extends Action
{

	public byte[][] boardChanges;
	
	public BoardAction(byte[][] boardChanges)
	{
		this.boardChanges=boardChanges;
	}
	
	@Override
	public int hashCode() 
	{
	    	int hash=0;
		for(byte[] col: boardChanges)
		{
			hash+=UnsecureHash.hash(col, 97);
		}
		return hash;
	}

	@Override
	public boolean equals(Object other) 
	{
		for(int rowInd=0; rowInd<boardChanges.length; rowInd++)
		{
			if(!Arrays.equals(boardChanges[rowInd], ((BoardAction)other).boardChanges[rowInd]))
			{
				return false;
			}
		}
		return true;
	}

	@Override
	public double[] getValue() 
	{
		double[] value=new double[boardChanges.length*boardChanges[0].length];
		for(int rowInd=0; rowInd<boardChanges.length; rowInd++)
		{
			for(int colInd=0; colInd<boardChanges[0].length; colInd++)
			{
				value[rowInd*boardChanges.length+colInd]=boardChanges[rowInd][colInd];
			}
		}
		return value;
	}

	@Override
	public BoardAction getFromNNValue(double[] value) 
	{
	    	byte[][] boardChanges=new byte[this.boardChanges.length][this.boardChanges[0].length];
		for(int rowInd=0; rowInd<boardChanges.length; rowInd++)
		{
			for(int colInd=0; colInd<boardChanges[0].length; colInd++)
			{
				boardChanges[rowInd][colInd]=(byte)Math.round(value[rowInd*boardChanges.length+colInd]);
			}
		}
		return new BoardAction(boardChanges);
	}

}
