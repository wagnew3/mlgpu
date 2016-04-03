package regularization;

import org.apache.commons.math3.linear.BlockRealMatrix;

import layer.BLayer;
import layer.Layer;
import network.SplitNetwork;

public class L2Regularization extends Regularization
{
	
	protected double numberOutputs;
	protected double lambda;
	protected double mu;
	
	
	public L2Regularization(double numberOutputs, double lambda, double mu)
	{
		this.numberOutputs=numberOutputs;
		this.lambda=lambda;
		this.mu=mu;
	}

	@Override
	public SplitNetwork regularize(SplitNetwork network) 
	{
		for(BLayer[] layer: network.getLayers())
		{
			for(BLayer net: layer)
			{
				if(net.getWeights()!=null)
				{
					for(BlockRealMatrix weights: net.getWeights())
					{
						for(int row=0; row<weights.getRowDimension(); row++)
						{
							for(int col=0; col<weights.getColumnDimension(); col++)
							{
								weights.setEntry(row, col, (1.0-lambda*mu/numberOutputs)*weights.getEntry(row, col));
							}
						}
					}
				}
				if(net.getBiases()!=null)
				{
					for(int entryInd=0; entryInd<net.getBiases().getDimension(); entryInd++)
					{
						net.getBiases().setEntry(entryInd, (1.0-lambda*mu/numberOutputs)*net.getBiases().getEntry(entryInd));
					}
				}
			}
		}
		return network;
	}

}
