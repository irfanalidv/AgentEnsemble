"""
Metrics

Performance metrics for agent evaluation.
"""

from typing import Any, Dict, List


class Metrics:
    """
    Metrics calculator for agent performance.
    
    Tracks success rate, execution time, cost, etc.
    """
    
    @staticmethod
    def success_rate(results: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        if not results:
            return 0.0
        
        successful = sum(1 for r in results if r.get("success", False))
        return successful / len(results)
    
    @staticmethod
    def average_execution_time(results: List[Dict[str, Any]]) -> float:
        """
        Calculate average execution time.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Average execution time in seconds
        """
        if not results:
            return 0.0
        
        times = [r.get("execution_time", 0) for r in results]
        return sum(times) / len(times)
    
    @staticmethod
    def total_cost(results: List[Dict[str, Any]]) -> float:
        """
        Calculate total cost.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Total cost in dollars
        """
        return sum(r.get("cost", 0) for r in results)
    
    @staticmethod
    def calculate_all(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate all metrics.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with all metrics
        """
        return {
            "success_rate": Metrics.success_rate(results),
            "average_execution_time": Metrics.average_execution_time(results),
            "total_cost": Metrics.total_cost(results),
            "total_tests": len(results),
        }

