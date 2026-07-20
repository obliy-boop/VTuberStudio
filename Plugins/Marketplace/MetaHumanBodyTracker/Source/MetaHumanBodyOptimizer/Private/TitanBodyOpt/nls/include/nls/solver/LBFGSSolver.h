// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include <carbon/Common.h>
#include <nls/Context.h>
#include <nls/DiffData.h>

CARBON_NAMESPACE_BEGIN(TITAN_NAMESPACE)

template <class T>
class LBFGSSolver
{
public:
	LBFGSSolver()
	{
		// make sure storage is set up correctly
		s.resize(historySize);
		y.resize(historySize);
		p = Vector<T>::Zero(historySize);
		a = Vector<T>::Zero(historySize);
	}

	bool Solve(const std::function<DiffData<T>(Context<T>*)>& evaluationFunction,
			   int iterations = 1) const
	{
		Context<T> context;
		return Solve(evaluationFunction, context, iterations);
	}

	bool Solve(const std::function<DiffData<T>(Context<T>*)>& evaluationFunction,
			   Context<T>& context,
			   int iterations = 1) const;

	// Setter methods for stopping criteria
	void SetResidualErrorStoppingCriterion(T value) { residualErrorStoppingCriterion = value; }
	void SetPredictionReductionStoppingCriterion(T value) { predictionReductionStoppingCriterion = value; }
	void SetAbsDeltaStoppingCriterion(T value) { absDeltaStoppingCriterion = value; }
	void SetAbsDeltaStoppingWindow(int value) { absDeltaStoppingWindow = value; }

	// Getter methods for stopping criteria
	T GetResidualErrorStoppingCriterion() const { return residualErrorStoppingCriterion; }
	T GetPredictionReductionStoppingCriterion() const { return predictionReductionStoppingCriterion; }
	T GetAbsDeltaStoppingCriterion() const { return absDeltaStoppingCriterion; }
	int GetAbsDeltaStoppingWindow() const { return absDeltaStoppingWindow; }

private:
	// LBFGS storage
	int historySize = 50;
	mutable std::vector<Vector<T>> s;
	mutable std::vector<Vector<T>> y;
	mutable Vector<T> p;
	mutable Vector<T> a;

	// Stopping criteria
	T residualErrorStoppingCriterion = T(1e-8);  
	T predictionReductionStoppingCriterion = T(1e-6);  
	T absDeltaStoppingCriterion = T(0);  // disabled by default
	int absDeltaStoppingWindow = 1;
};

CARBON_NAMESPACE_END(TITAN_NAMESPACE)
