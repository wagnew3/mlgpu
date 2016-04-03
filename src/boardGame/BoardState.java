package boardGame;

import java.util.Arrays;

import ReinforcementMachineLearningFramework.Action;
import ReinforcementMachineLearningFramework.State;
import utils.UnsecureHash;

public class BoardState extends State
{
	
	protected byte[][] boardState;
	
	public BoardState(byte[][] boardState)
	{
		this.boardState=boardState;
	}

	@Override
	public int hashCode() 
	{
		int hash=0;
		for(byte[] col: boardState)
		{
			hash+=UnsecureHash.hash(col, 97);
		}
		return hash;
	}

	@Override
	public boolean equals(Object other) 
	{
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			if(!Arrays.equals(boardState[rowInd], ((BoardState)other).boardState[rowInd]))
			{
				return false;
			}
		}
		return true;
	}

	@Override
	public boolean isEndState() 
	{
		System.out.println("Calling generic isEndState()!");
		return false;
	}

	@Override
	public double[] getValue() 
	{
		double[] value=new double[boardState.length*boardState[0].length];
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				value[rowInd*boardState.length+colInd]=boardState[rowInd][colInd];
			}
		}
		return value;
	}
	
	@Override
	public double[] getNNValue() 
	{
		double[] value=new double[boardState.length*boardState[0].length];
		for(int rowInd=0; rowInd<boardState.length; rowInd++)
		{
			for(int colInd=0; colInd<boardState[rowInd].length; colInd++)
			{
				value[rowInd*boardState.length+colInd]=boardState[rowInd][colInd];
			}
		}
		return value;
	}

	@Override
	public BoardState getFromNNValue(double[] value) 
	{
		byte[][] boardChanges=new byte[this.boardState.length][this.boardState[0].length];
		for(int rowInd=0; rowInd<boardChanges.length; rowInd++)
		{
			for(int colInd=0; colInd<boardChanges[rowInd].length; colInd++)
			{
				boardChanges[rowInd][colInd]=(byte)Math.round(value[rowInd*boardChanges.length+colInd]);
			}
		}
		return new BoardState(boardChanges);
	}
	
	public int[][] getBoardImage()
	{
		return null;
	}
	
	public byte[][] getBoardState()
	{
		return boardState;
	}
	

}
