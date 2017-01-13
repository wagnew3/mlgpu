package filters;

import nDimensionalMatrices.Matrix;

public class SumToOneFilter 
{
	
	public SumToOneFilter()
	{
		
	}
	
	public Matrix[][] scaleData(Matrix[][] data)
	{
		for(int dataInd=0; dataInd<data.length; dataInd++)
		{
			float sum=0;
			for(int dataPartInd=0; dataPartInd<data[dataInd].length; dataPartInd++)
			{
				for(int dataPartPartInd=0; 
						dataPartPartInd<data[dataInd][dataPartInd].getLen();
						dataPartPartInd++)
				{
					sum+=data[dataInd][dataPartInd].get(dataPartPartInd, 0);
				}
			}
			
			if(sum>0)
			{
				for(int dataPartInd=0; dataPartInd<data[dataInd].length; dataPartInd++)
				{
					for(int dataPartPartInd=0; 
							dataPartPartInd<data[dataInd][dataPartInd].getLen();
							dataPartPartInd++)
					{
						data[dataInd][dataPartInd].set(dataPartPartInd,
								0,
								data[dataInd][dataPartInd].get(dataPartPartInd, 0)/sum);
					}
				}
			}
		}
		return data;
	}

}
