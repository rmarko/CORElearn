#include <float.h>

#include "general.h"
#include "error.h"
#include "contain.h"
#include "utils.h"
#include "estimatorReg.h"
#include "binpart.h"



// ************************************************************
//
//                       binarizeGeneral
//                       ----------------
//
//    creates binary split of attribute values according to
//   the selected estimator; search is exhaustive, greedy or random depending
//   on the number of values of attribute
//
// ************************************************************
void estimationReg::binarizeGeneral(int selectedEstimator, constructReg &nodeConstruct, double &bestEstimation, int firstFreeDiscSlot)
{
   if (firstFreeDiscSlot == 0)
	   firstFreeDiscSlot = noDiscrete ;

   int NoValues = nodeConstruct.noValues ;
   nodeConstruct.leftValues.create(NoValues+1,mFALSE) ;
   
   if (NoValues < 2) 
   {
	  bestEstimation = -DBL_MAX ;
      return ;
   }

   booleanT binaryEvaluationBefore = eopt.binaryEvaluation ;
   eopt.binaryEvaluation = mFALSE ;
 
   attributeCount bestType ;
   int i ;

   if (NoValues == 2) // already binary, but we estimate it
   {
       adjustTables(0, firstFreeDiscSlot + 1) ;
       for (i=0 ; i < TrainSize ; i++)
          DiscValues.Set(i, firstFreeDiscSlot, nodeConstruct.discreteValue(DiscValues,NumValues,i)) ;
          
       prepareDiscAttr(firstFreeDiscSlot, 2) ; 

	   i = estimate(eopt.selectionEstimatorReg, 1, 1, firstFreeDiscSlot, firstFreeDiscSlot+1, bestType) ;
       nodeConstruct.leftValues[1] = mTRUE ;
       bestEstimation =  DiscEstimation[firstFreeDiscSlot] ;
   }
  

   char attrValue ;
   int bestIdx ;
   bestEstimation = -DBL_MAX ;
   if (NoValues > eopt.maxValues4Greedy) {
	    // random binarization
		marray<int>  valueCount(NoValues, 0) ;
		for (i=0 ; i < TrainSize ; i++)	{
			attrValue = nodeConstruct.discreteValue(DiscValues, NumValues, i) ;
			valueCount[attrValue] ++ ;
		}
		int validValues = TrainSize - valueCount[NAdisc] ;
		if ( validValues <= eopt.minNodeWeightEst/2.0) { // split will be invalid anyway
			nodeConstruct.leftValues.init(mFALSE) ;
		}
		double targetLeftVals = randBetween(eopt.minNodeWeightEst, validValues/2.0) ;
		int leftVal = 0 ;
		// prepare the random order of values
		marray<int> order(NoValues+1) ;
		for (i = 0 ; i <= NoValues; ++i)
			order[i] = i ;
		for (i = 1 ; i < NoValues; ++i) { // shuffle all expect 0th
			swap(order[i], order[randBetween(i, NoValues)]) ;
		}
		for (int idx=0 ; idx <= NoValues ; ++idx) 	{
			leftVal += valueCount[order[idx]] ;
            if (leftVal == validValues)  // do not allow all relevant values on the left
            	break ;
			nodeConstruct.leftValues[ order[idx] ] = mTRUE ;
			if (leftVal >= targetLeftVals)
				break ;
		}
		// now estimate the quality of the split
	    adjustTables(0, firstFreeDiscSlot + 1) ;
	    for (i=0 ; i < TrainSize ; i++)
	       {
	          attrValue = nodeConstruct.discreteValue(DiscValues,NumValues,i) ;
	          if (attrValue == NAdisc)
	            DiscValues.Set(i, firstFreeDiscSlot, NAdisc) ;
	          else
	            if (nodeConstruct.leftValues[attrValue])
	              DiscValues.Set(i, firstFreeDiscSlot, 1) ;
	            else
	              DiscValues.Set(i, firstFreeDiscSlot, 2) ;
	       }
	    prepareDiscAttr(firstFreeDiscSlot, 2) ;
		i = estimate(eopt.selectionEstimatorReg, 1, 1, firstFreeDiscSlot, firstFreeDiscSlot+1, bestType) ;
	    bestEstimation =  DiscEstimation[firstFreeDiscSlot] ;
	}
	else if ( NoValues <= eopt.maxValues4Exhaustive) { // &&(exhaustivePositions * 0.8 <= greedyPositions ||exhaustivePositions < eopt.discretizationSample))
     // exhaustive search
     binPartition Generator(NoValues) ;
     double exhaustivePositions = Generator.noPositions() ;
     adjustTables(0,  int(firstFreeDiscSlot + exhaustivePositions)) ;
     marray<marray<booleanT> >  leftValues( (int)exhaustivePositions) ;
     int noIncrements = 0 ;
     while (Generator.increment() )
     {
       // save partition
       leftValues[noIncrements] = Generator.leftPartition ;
       // compute data column
       for (i=0 ; i < TrainSize ; i++)
       {
          attrValue = nodeConstruct.discreteValue(DiscValues,NumValues,i) ;
          if (attrValue == NAdisc)
            DiscValues.Set(i, firstFreeDiscSlot + noIncrements, NAdisc) ;
          else
            if (leftValues[noIncrements][attrValue])
              DiscValues.Set(i, firstFreeDiscSlot + noIncrements, 1) ;
            else
              DiscValues.Set(i, firstFreeDiscSlot + noIncrements, 2) ;  
       }
       prepareDiscAttr(firstFreeDiscSlot + noIncrements, 2) ; 
       noIncrements++ ;
     }
  
     // estimate and select best
     bestIdx = estimate(selectedEstimator, 1, 1,
                               firstFreeDiscSlot, firstFreeDiscSlot+noIncrements, bestType) ;
     nodeConstruct.leftValues = leftValues[bestIdx - firstFreeDiscSlot] ; 
     bestEstimation = DiscEstimation[bestIdx] ;
   }
   else   {  // greedy search
     adjustTables(0, firstFreeDiscSlot + NoValues) ;
     marray<marray<booleanT> >  leftValues(NoValues) ;
     marray<booleanT> currentBest(NoValues+1, mFALSE) ;
     int j, added ;
     for (int filled=1 ; filled < NoValues ; filled++)
     {
        added = 0 ;
        for (j=1 ; j <= NoValues ; j++)
          if (currentBest[j] == mFALSE)
          {
            currentBest[j] = mTRUE ;
            leftValues[added] = currentBest ;
    
            // compute data column
            for (i=0 ; i < TrainSize ; i++)
            {
               attrValue = nodeConstruct.discreteValue(DiscValues,NumValues,i) ;
               if (attrValue == NAdisc)
                  DiscValues.Set(i, firstFreeDiscSlot + added, NAdisc) ;
               else
                 if (leftValues[added][attrValue])
                   DiscValues.Set(i, firstFreeDiscSlot + added, 1) ;
                 else
                   DiscValues.Set(i, firstFreeDiscSlot + added, 2) ;  
            }
            prepareDiscAttr(firstFreeDiscSlot + added, 2) ;
            
            currentBest[j] = mFALSE ;
            added ++ ;
          }
        bestIdx = estimate(selectedEstimator, 1, 1,
                               firstFreeDiscSlot, firstFreeDiscSlot + added, bestType) ;
        currentBest = leftValues[bestIdx - firstFreeDiscSlot] ; 
        if (DiscEstimation[bestIdx] > bestEstimation)
        {
          bestEstimation = DiscEstimation[bestIdx] ;
          nodeConstruct.leftValues =  currentBest ;
        }
     }
   }
   eopt.binaryEvaluation = binaryEvaluationBefore ;

}


//************************************************************
//
//                       binarizeBreiman
//                       ---------------
//
//    creates binary split of attribute values according to
//   optimal procedure described in Breiman et all, 1984,
//         (for numeric prediction value)
//
//************************************************************
void estimationReg::binarizeBreiman(constructReg &nodeConstruct, double &bestEstimation)
{
   nodeConstruct.leftValues.init(mFALSE) ;
   int NoValues = nodeConstruct.noValues ;
   marray<double> valueClass(NoValues+1, 0.0) ;
   marray<double> valueWeight(NoValues+1, 0.0) ;
   marray<double> squaredValues(NoValues+1, 0.0) ;
   marray<sortRec> sortedMean(NoValues) ;
   int idx, j ;
   double value ;
   // estimationReg of discrete attributtes

   for (j=0 ; j < TrainSize ; j++)
   {
      idx = nodeConstruct.discreteValue(DiscValues,NumValues,j) ;
      value = NumValues(j, 0) ;
      valueClass[idx] += weight[j]* value ;
      valueWeight[idx] += weight[j] ;
      squaredValues[idx] += weight[j] * sqr(value) ;
   }
   double RightWeight = 0.0, RightSquares = 0.0, RightValues = 0.0 ;
   int OKvalues = 0 ;
   for (j=1 ; j <= NoValues ; j++)
   {
      if (valueWeight[j] > epsilon)
      {
        sortedMean[OKvalues].key = valueClass[j] / valueWeight[j] ;
        sortedMean[OKvalues].value = j ;
        OKvalues ++ ;
  
        RightWeight += valueWeight[j] ;
        RightSquares +=  squaredValues[j] ;
        RightValues += valueClass[j] ;
      }
   }
   double totalWeight = RightWeight ;
   sortedMean.setFilled(OKvalues) ;
   sortedMean.qsortAsc() ;
   double estimate, pLeft, variance ;
   bestEstimation = DBL_MAX ;
   int bestIdx = -1 ;
   double LeftWeight = 0.0, LeftSquares = 0.0, LeftValues = 0.0 ;
   int upper = OKvalues - 1 ;
   for (j=0 ; j < upper ; j++)
   {
       idx = sortedMean[j].value ;
       LeftSquares += squaredValues[idx] ;
       LeftValues += valueClass[idx] ;
       LeftWeight += valueWeight[idx] ;
       RightSquares -= squaredValues[idx] ;
       RightValues -= valueClass[idx] ;
       RightWeight -= valueWeight[idx] ;
       pLeft = LeftWeight/totalWeight ;
       variance = LeftSquares/LeftWeight - sqr(LeftValues/LeftWeight) ;
       if (LeftWeight > epsilon && variance > 0.0)
         estimate = pLeft *sqrt(variance) ;
       else
         estimate = 0.0 ;
       variance = RightSquares/RightWeight -sqr(RightValues/RightWeight) ;
       if (RightWeight > epsilon && variance > 0.0 )
          estimate +=  (double(1.0) - pLeft)*sqrt(variance) ;
       if (estimate < bestEstimation)
       {
           bestEstimation = estimate ;
           bestIdx = j ;
       }
   }
   nodeConstruct.leftValues.init(mFALSE) ;

   for (j=0 ; j <= bestIdx ; j++)
      nodeConstruct.leftValues[sortedMean[j].value] = mTRUE ;

   #if defined(DEBUG)
     if ( bestIdx < 0)
        merror("regressionTree::binarizeBreiman","invalid split selected") ;
   #endif

}



//************************************************************
//
//                        bestSplitGeneral
//                        -----------------
//
//            finds best split for numeric attribute with selected estimator
//
//************************************************************
double estimationReg::bestSplitGeneral(int selectedEstimator, constructReg &nodeConstruct, double &bestEstimation, int firstFreeDiscSlot)
{
   if (firstFreeDiscSlot == 0)
	   firstFreeDiscSlot = noDiscrete ;

   marray<sortRec> sortedAttr(TrainSize) ;
   int i, j ;
   int OKvalues = 0 ;
   double attrValue ;
   for (j=0 ; j < TrainSize ; j++)
   {
      attrValue = nodeConstruct.continuousValue(DiscValues,NumValues,j) ;
      if (isNAcont(attrValue))
        continue ;
      sortedAttr[OKvalues].key = attrValue ;
      sortedAttr[OKvalues].value = j ;
      OKvalues ++ ;
   }
   if (OKvalues <= 1)    // all the cases have missing value of the attribute or only one OK
   {
      bestEstimation = - DBL_MAX ;
      return - DBL_MAX ; // smaller than any value, so all examples will go into one branch
   }
   sortedAttr.setFilled(OKvalues) ;
   sortedAttr.qsortAsc() ;
   
   int lastUnique = 0 ;
   for (i=1 ; i < OKvalues ; i++)
   {
      if (sortedAttr[i].key != sortedAttr[lastUnique].key)
      {
         lastUnique ++ ;
         sortedAttr[lastUnique] = sortedAttr[i] ;
      }
   }
   OKvalues = lastUnique+1 ;
    if (OKvalues <= 1)    
   {
      bestEstimation = - DBL_MAX ;
      return - DBL_MAX ; // smaller than any value, so all examples will go into one branch
   }

   int sampleSize ; 
   if (eopt.discretizationSample==0)
     sampleSize = OKvalues -1;
   else
     sampleSize = Mmin(eopt.discretizationSample, OKvalues-1) ;
   marray<int> splits(sampleSize) ;
   randomizedSample(splits, sampleSize, OKvalues-1) ;

   attributeCount bestType ;

   adjustTables(0, firstFreeDiscSlot + sampleSize) ;
   for (j=0 ; j < sampleSize ; j++)
   { 
       // compute data column
     for (i=0 ; i < TrainSize ; i++)
     {
       attrValue = nodeConstruct.continuousValue(DiscValues,NumValues,i) ;
       if (isNAcont(attrValue))
         DiscValues.Set(i, firstFreeDiscSlot + j, NAdisc) ;
       else
         if ( attrValue <= sortedAttr[splits[j]].key )
           DiscValues.Set(i, firstFreeDiscSlot + j, 1) ;
         else
           DiscValues.Set(i, firstFreeDiscSlot + j, 2) ;  
     }
     prepareDiscAttr(firstFreeDiscSlot + j, 2) ; 
   }
 
   booleanT binaryEvaluationBefore = eopt.binaryEvaluation ;
   eopt.binaryEvaluation = mFALSE ;

   // estimate and select best
   int bestIdx = estimate(selectedEstimator, 1, 1,
                            firstFreeDiscSlot, firstFreeDiscSlot+sampleSize, bestType) ;
   bestEstimation = DiscEstimation[bestIdx] ;
     
   eopt.binaryEvaluation = binaryEvaluationBefore ;

   return (sortedAttr[splits[bestIdx-firstFreeDiscSlot]].key + sortedAttr[splits[bestIdx-firstFreeDiscSlot]+1].key)/2.0 ;
}





//************************************************************
//
//                        bestMSEsplit
//                        ------------
//
//            finds best split for continuous attribute with standard
//                         deviation
//
//************************************************************
double estimationReg::bestMSEsplit(constructReg &nodeConstruct, double &bestEstimation)
{
  // continuous values
   double dVal, attrValue ;
   marray<sortRec> sortedAttr(TrainSize) ;
   double LeftWeight = 0.0, LeftSquares = 0.0, LeftValues = 0.0 ;
   double RightWeight = 0.0, RightSquares = 0.0, RightValues = 0.0 ;
   int j, idx ;
   int OKvalues = 0 ;
   for (j=0 ; j < TrainSize ; j++)
   {
      attrValue = nodeConstruct.continuousValue(DiscValues,NumValues,j) ;
      if (isNAcont(attrValue))
        continue ;
      sortedAttr[OKvalues].key = attrValue ;
      sortedAttr[OKvalues].value = j ;
      RightWeight += weight[j] ;
      dVal = weight[j] * NumValues(j,0) ;
      RightValues += dVal ;
      dVal *=  NumValues(j,0) ;
      RightSquares += dVal  ;
      OKvalues ++ ;
   }

   double totalWeight = RightWeight ;
   sortedAttr.setFilled(OKvalues) ;
   sortedAttr.qsortAsc() ;
   bestEstimation = DBL_MAX ;
   int bestIdx = -1 ;
   double estimate, pLeft, variance ;

   // int upper = OKvalues  ;
   j=0 ;
   while (j < OKvalues)
   {
      // collect cases with the same value of the attribute - we cannot split between them
      do {
         idx = sortedAttr[j].value ;
         dVal = weight[idx] * NumValues(idx, 0) ;
         LeftValues += dVal ;
         RightValues -= dVal ;
         dVal *= NumValues(idx, 0) ;
         LeftSquares += dVal ;
         RightSquares -= dVal ;
         LeftWeight += weight[idx] ;
         RightWeight -= weight[idx] ;
         j ++ ;
      } while (j < OKvalues && sortedAttr[j].key == sortedAttr[j-1].key)  ;

      // we cannot split with the biggest value
      if (j == OKvalues)
         break ;
      pLeft = LeftWeight/totalWeight ;
      variance = LeftSquares/LeftWeight - sqr(LeftValues/LeftWeight) ;

      if (LeftWeight > epsilon && variance > 0.0)
         estimate = pLeft * sqrt(variance) ;
      else
         estimate = 0.0 ;

      variance = RightSquares/RightWeight -sqr(RightValues/RightWeight) ;
      if (RightWeight > epsilon && variance > 0.0 )
         estimate += (double(1.0) - pLeft) * sqrt(variance) ;
      if (estimate < bestEstimation)
      {
         bestEstimation = estimate ;
         bestIdx = j ;
      }
   }

   if ( bestIdx < 0 )
   {
       // degenerated case
       if (OKvalues > 0)   // all the values are the same
          return sortedAttr[0].key - double(1.0) ;  // smaller then minimum: split will put all the cases
                                           // into one subtree and node will become a leaf
       else  // all the cases have missing value of the attribute
          return - DBL_MAX ;
   }
   else
     return (sortedAttr[bestIdx].key + sortedAttr[bestIdx-1].key)/double(2.0) ;

}



//************************************************************
//
//                        estBinarized
//                        ------------
//
//       estimate attribute as if they were binarized
//
//************************************************************
void estimationReg::estBinarized(int selectedEstimator, int contAttrFrom, int contAttrTo, 
                         int discAttrFrom, int discAttrTo, int firstFreeDiscSlot)
{
   if (firstFreeDiscSlot == 0)
	   firstFreeDiscSlot = noDiscrete ;

   booleanT binaryEvaluationBefore = eopt.binaryEvaluation ;
   eopt.binaryEvaluation = mFALSE ;

   attributeCount bestType ;
   int addedAttr = 0, i, j, NoValues, noPartitions, iDisc, iCont, estIdx ;
   int NoDiscEstimated = discAttrTo - discAttrFrom ;
   int NoContEstimated = contAttrTo - contAttrFrom ;
   marray<int> discFrom(NoDiscEstimated), discTo(NoDiscEstimated), contFrom(NoContEstimated), contTo(NoContEstimated) ;
   int discAttrValue ;

   // estimated size
   adjustTables(0, firstFreeDiscSlot + NoDiscEstimated* 4 + NoContEstimated * eopt.discretizationSample) ;


   for (iDisc=discAttrFrom ; iDisc < discAttrTo; iDisc++)
   {
       NoValues = discNoValues[iDisc] ;
	   estIdx = iDisc - discAttrFrom ; 

       if (NoValues < 2) 
	   {
		  discFrom[estIdx] = discTo[estIdx] = -1 ;
	   }
       else  if (NoValues == 2) // already binary, we estimate it
	   {
		   adjustTables(0, firstFreeDiscSlot + addedAttr + 1) ;
		   for (i=0 ; i < TrainSize ; i++)
			  DiscValues.Set(i, firstFreeDiscSlot + addedAttr, DiscValues(i,iDisc)) ;
          
		   prepareDiscAttr(firstFreeDiscSlot+addedAttr, 2) ; 
           discFrom[estIdx] = firstFreeDiscSlot + addedAttr ;
           discTo[estIdx] = firstFreeDiscSlot + addedAttr + 1 ;
           addedAttr ++ ;
		   continue ;
	   }
	   else {
  
		   binPartition Generator(NoValues) ;
           noPartitions = 0 ;
		   adjustTables(0,  firstFreeDiscSlot + addedAttr + int(Mmin(Generator.noPositions(), (double)(eopt.discretizationSample)))) ;
           discFrom[estIdx] = firstFreeDiscSlot + addedAttr ;
 		   while (Generator.increment() )
		   {
			 // compute data column
			 for (i=0 ; i < TrainSize ; i++)
			 {
			   discAttrValue = DiscValues(i, iDisc) ;
			   if (discAttrValue == NAdisc)
				 DiscValues.Set(i, firstFreeDiscSlot + addedAttr, NAdisc) ;
			   else
				 if (Generator.leftPartition[discAttrValue])
					DiscValues.Set(i, firstFreeDiscSlot + addedAttr, 1) ;
				 else
					DiscValues.Set(i, firstFreeDiscSlot + addedAttr, 2) ;  
			  }
			  prepareDiscAttr(firstFreeDiscSlot + addedAttr, 2) ; 
			  addedAttr++ ;
              noPartitions++ ;
			  if (noPartitions >= eopt.discretizationSample)
				  break ;
			}
            discTo[estIdx] = firstFreeDiscSlot + addedAttr ;

	   }
   }

   marray<sortRec> sortedAttr(TrainSize) ;
   int OKvalues  ;
   double contAttrValue ;
   int sampleSize ; 
   marray<int> splits(TrainSize), sortedCopy(TrainSize) ;

   for (iCont=contAttrFrom ; iCont < contAttrTo; iCont++)
   {

	   estIdx = iCont - contAttrFrom ;
	   contFrom[estIdx] = firstFreeDiscSlot + addedAttr ;
       OKvalues = 0 ;
       
	   for (j=0 ; j < TrainSize ; j++)
	   {
		  contAttrValue = NumValues(j, iCont) ;
		  if (isNAcont(contAttrValue))
			continue ;
		  sortedAttr[OKvalues].key = contAttrValue ;
		  sortedAttr[OKvalues].value = j ;
		  OKvalues ++ ;
	   }
	   if (OKvalues <= 1)    // all the cases have missing value of the attribute or only one OK
	   {
		  contTo[estIdx] = -1 ;
		  continue ;
	   }
	   sortedAttr.setFilled(OKvalues) ;
	   sortedAttr.qsortAsc() ;
   
	   int lastUnique = 0 ;
	   for (i=1 ; i < OKvalues ; i++)
	   {
		  if (sortedAttr[i].key != sortedAttr[lastUnique].key)
		  {
			 lastUnique ++ ;
			 sortedAttr[lastUnique] = sortedAttr[i] ;
		  }
	   }
	   OKvalues = lastUnique+1 ;
	   if (OKvalues <= 1)    
	   {
		  contTo[estIdx] = -1 ;
		  continue ;
	   }


	   if (eopt.discretizationSample==0)
		 sampleSize = OKvalues -1;
	   else
		 sampleSize = Mmin(eopt.discretizationSample, OKvalues-1) ;

      randomizedSample(splits, sampleSize, OKvalues-1) ;

//	   if (OKvalues-1 > sampleSize)  
//	   {
//		   // do sampling
//		   for (i=0 ; i < OKvalues ; i++)
//			 sortedCopy[i] = i ;
//        
//		   int upper = OKvalues - 1 ;
//		   int selected ;
//		   for (i=0 ; i < sampleSize ; i++)
//		   {
//			  selected = randBetween(0, upper) ;
//			  splits[i] = sortedCopy[selected] ;
//			  upper -- ;
//			  sortedCopy[selected] = sortedCopy[upper] ;
//		   }
//	   }
//	   else
//		 for (i=0 ; i < sampleSize ; i++)
//			splits[i] = i ;


	   adjustTables(0, firstFreeDiscSlot + addedAttr+ sampleSize) ;
	   for (j=0 ; j < sampleSize ; j++)
	   { 
		   // compute data column
		 for (i=0 ; i < TrainSize ; i++)
		 {
		   contAttrValue = NumValues(i,iCont) ;
		   if (isNAcont(contAttrValue))
			 DiscValues.Set(i, firstFreeDiscSlot + addedAttr, NAdisc) ;
		   else
			 if ( contAttrValue <= sortedAttr[splits[j]].key )
			   DiscValues.Set(i, firstFreeDiscSlot + addedAttr, 1) ;
			 else
			   DiscValues.Set(i, firstFreeDiscSlot + addedAttr, 2) ;  
		 }
		 prepareDiscAttr(firstFreeDiscSlot + addedAttr, 2) ;
		 addedAttr ++ ;
	   }
   
	   contTo[estIdx] = firstFreeDiscSlot + addedAttr ;

   }
   
   estimate(selectedEstimator, 1, 1, firstFreeDiscSlot, firstFreeDiscSlot + addedAttr, bestType) ;
   int iBin ;
   for (iDisc=discAttrFrom ; iDisc < discAttrTo; iDisc++)
   {
	  estIdx = iDisc - discAttrFrom ;
      DiscEstimation[iDisc] = -DBL_MAX ;
      for (iBin=discFrom[estIdx] ; iBin < discTo[estIdx] ; iBin++)
		  if (DiscEstimation[iBin] > DiscEstimation[iDisc])
			  DiscEstimation[iDisc] = DiscEstimation[iBin] ;
   }

   for (iCont=contAttrFrom ; iCont < contAttrTo; iCont++)
   {
	  estIdx = iCont - contAttrFrom ;
      NumEstimation[iCont] = -DBL_MAX ;
      for (iBin=contFrom[estIdx] ; iBin < contTo[estIdx] ; iBin++)
		  if (DiscEstimation[iBin] > NumEstimation[iCont])
			  NumEstimation[iCont] = DiscEstimation[iBin] ;
   }

   eopt.binaryEvaluation = binaryEvaluationBefore ;
}



//************************************************************
//
//                        discretizeGreedy
//                        -----------------
//
//     finds best discretization of numeric attribute with
//      greedy algorithm and returns its estimated quality
//
//************************************************************
double estimationReg::discretizeGreedy(int ContAttrIdx, int maxBins, marray<double> &Bounds)
{
	Bounds.setFilled(0) ;

	marray<sortRec> sortedAttr(TrainSize) ;
	int i, j, idx ;
	int OKvalues = 0 ;
	for (j=0 ; j < TrainSize ; j++)
	{
		if (isNAcont(NumValues(j, ContAttrIdx)))
			continue ;
		sortedAttr[OKvalues].key = NumValues(j, ContAttrIdx) ;
		sortedAttr[OKvalues].value = j ;
		OKvalues ++ ;
	}
	if (OKvalues <= 1)    // all the cases have missing value of the attribute or only one OK
	{
		// merror("regressionTree::discretizeGreedy", "all values of the attribute are missing or equal") ;
		return - DBL_MAX ;
	}
	sortedAttr.setFilled(OKvalues) ;
	sortedAttr.qsortAsc() ;

	// eliminate duplicates
	int lastUnique = 0 ;
	for (j=1 ; j < OKvalues ; j ++)
	{
		if (sortedAttr[j].key != sortedAttr[lastUnique].key)
		{
			lastUnique ++ ;
			sortedAttr[lastUnique] = sortedAttr[j] ;
		}
	}
	OKvalues = lastUnique+1 ;
	sortedAttr.setFilled(OKvalues) ;

	if (OKvalues <= 1)    // all the cases have missing value of the attribute or only one OK
	{
		// merror("regressionTree::discretizeGreedy", "all values of the attribute are missing or equal") ;
		return - DBL_MAX ;
	}


	int sampleSize ;
	// we use all the available values only if explicitly demanded
	if (eopt.discretizationSample==0)
		sampleSize = OKvalues -1;
	else
		sampleSize = Mmin(eopt.discretizationSample, OKvalues-1) ;
	marray<int> splits(sampleSize) ;
	randomizedSample(splits, sampleSize, OKvalues-1) ;

	attributeCount bestType ;
	double attrValue ;

	adjustTables(0, noDiscrete + sampleSize) ;
	// greedy search

	marray<double> currentBounds(sampleSize) ;
	int currentIdx ;
	double bestEstimate = - DBL_MAX, bound ;
	int currentLimit=0 ; // number of times the current discretization was worse than the best discretization
	int currentNoValues = 2 ;
	while (currentLimit <= eopt.discretizationLookahead && sampleSize > 0 && (maxBins==0 || currentNoValues <= maxBins))
	{
		// compute data columns
		for (i=0 ; i < TrainSize ; i++)
		{
			attrValue = NumValues(i, ContAttrIdx) ;
			idx = 0 ;
			while (idx < currentBounds.filled()  &&  attrValue > currentBounds[idx])
				idx++ ;
			idx ++ ; // changes idx to discrete value
			for (j=0 ; j < sampleSize ; j++)
			{
				if (isNAcont(attrValue))
					DiscValues.Set(i, noDiscrete + j, NAdisc) ;
				else
					if (attrValue <= sortedAttr[splits[j]].key)
						DiscValues.Set(i, noDiscrete + j, idx) ;
					else
						DiscValues.Set(i, noDiscrete + j, idx+1) ;
			}
		}
		for (j=0 ; j < sampleSize ; j++)
			prepareDiscAttr(noDiscrete + j, currentNoValues) ;
		// estimate and select best
		currentIdx = estimate(eopt.selectionEstimatorReg, 1, 1, noDiscrete, noDiscrete+sampleSize, bestType) ;
		bound = (sortedAttr[splits[currentIdx-noDiscrete]].key	+ sortedAttr[splits[currentIdx-noDiscrete]+1].key)/2.0 ;
		currentBounds.addToAscSorted(bound) ;
		if (DiscEstimation[currentIdx] > bestEstimate)
		{
			bestEstimate = DiscEstimation[currentIdx] ;
			Bounds = currentBounds ;
			currentLimit = 0 ;
		}
		else
			currentLimit ++ ;
		splits[currentIdx-noDiscrete] = splits[--sampleSize] ;
		currentNoValues ++ ;
	}
	return bestEstimate ;
}




//************************************************************
//
//                        discretizeEqualFrequency
//                        -----------------------
//
//     discretize numeric attribute with a fixed number of intervals
//        with approximately the same number of examples in each interval
//
//************************************************************
void estimationReg::discretizeEqualFrequency(int ContAttrIdx, int noIntervals, marray<double> &Bounds)
{
	Bounds.setFilled(0) ;

	marray<sortRec> sortedAttr(TrainSize) ;
	int j ;
	int OKvalues = 0 ;
	for (j=0 ; j < TrainSize ; j++)
	{
		if (isNAcont(NumValues(j, ContAttrIdx)))
			continue ;
		sortedAttr[OKvalues].key = NumValues(j, ContAttrIdx) ;
		sortedAttr[OKvalues].value = 1 ;  // later used as a counter for number of unique values
		OKvalues ++ ;
	}
	if (OKvalues <= 1)    // all the cases have missing value of the attribute or only one OK
	{
		// all values of the attribute are missing
		return  ;
	}
	sortedAttr.setFilled(OKvalues) ;
	sortedAttr.qsortAsc() ;

	// eliminate and count duplicates
	int lastUnique = 0 ;
	for (j=1 ; j < OKvalues ; j++)
	{
		if (sortedAttr[j].key != sortedAttr[lastUnique].key)
		{
			lastUnique ++ ;
			sortedAttr[lastUnique] = sortedAttr[j] ;
		}
		else
			sortedAttr[lastUnique].value ++ ;
	}
	sortedAttr.setFilled(lastUnique+1) ;

	if (lastUnique < 1)
	{
		// all the cases have missing value of the attribute or only one OK
		return  ;
	}
	if (lastUnique < noIntervals)
	{
		// all unique values should form boundaries)

		Bounds.create(lastUnique) ;
		Bounds.setFilled(lastUnique) ;
		for (j=0 ; j < lastUnique ; j++)
			Bounds[j] = (sortedAttr[j].key + sortedAttr[j+1].key)/2.0 ;
		return ;
	}

	Bounds.create(noIntervals-1) ;

	int noDesired = int(ceil(double(OKvalues) / noIntervals)) ;
	double boundry ;

	int grouped = 0 ;
	for (j = 0 ; j < lastUnique ; j++)
	{
		if (grouped + sortedAttr[j].value < noDesired)
			grouped += sortedAttr[j].value ;
		else {
			// form new boundry
			boundry = (sortedAttr[j].key + sortedAttr[j+1].key) / 2.0 ;
			Bounds.addEnd(boundry) ;
			grouped = 0 ;
		}
	}
}

//************************************************************
//
//                        discretizeEqualWidth
//                        -----------------------
//
//     discretize numeric attribute with a fixed number of intervals of equal width
//
//************************************************************
void estimationReg::discretizeEqualWidth(int ContAttrIdx, int noIntervals, marray<double> &Bounds)
{
	Bounds.setFilled(0) ;

	int j=0 ;
	while (j < TrainSize && isNAcont(NumValues(j, ContAttrIdx)))
		j++ ;
	if (j == TrainSize)
		return ; // all values are missing
	double value, minValue, maxValue ;
	minValue = maxValue = NumValues(j, ContAttrIdx) ;
	for (++j ; j < TrainSize ; j++)
	{
		value = NumValues(j, ContAttrIdx) ;
		if (isNAcont(value))
			continue ;
		else if (value < minValue)
			minValue = value ;
		else if (value > maxValue)
			maxValue = value ;
	}
	if (minValue == maxValue)    //  only one non missing value
		return  ;
    double intervalWidth = (maxValue - minValue) / noIntervals ;
	Bounds.create(noIntervals-1) ;

	for (int i = 1 ; i < noIntervals ; i++)
	{
		value = minValue + i * intervalWidth ;
		Bounds.addEnd(value) ;
	}
}

