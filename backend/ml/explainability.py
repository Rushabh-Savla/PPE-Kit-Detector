"""
Risk Explainability module.

Provides human-readable explanations for why a worker is flagged as at-risk.
"""

from typing import Dict, List, Any


class RiskExplainer:
    """Generate explanations for risk scores"""

    def explain_risk(
        self,
        overall_risk_score: float,
        features: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate list of risk factors explaining the risk score.

        Args:
            overall_risk_score: Computed risk score (0-100)
            features: Extracted features
            predictions: Model predictions

        Returns:
            List of risk factors with impact and description
        """
        risk_factors = []

        # Check violation-related factors
        violations_30d = features.get("violations_last_30d", 0)
        if violations_30d >= 10:
            risk_factors.append({
                "factor": "high_violation_count",
                "impact": 0.9,
                "description": f"{violations_30d} PPE violations in last 30 days"
            })
        elif violations_30d >= 5:
            risk_factors.append({
                "factor": "moderate_violation_count",
                "impact": 0.6,
                "description": f"{violations_30d} PPE violations in last 30 days"
            })
        elif violations_30d >= 2:
            risk_factors.append({
                "factor": "low_violation_count",
                "impact": 0.3,
                "description": f"{violations_30d} PPE violations in last 30 days"
            })

        # Check attendance factors
        attendance_rate = features.get("attendance_rate_30d", 1.0)
        if attendance_rate < 0.6:
            risk_factors.append({
                "factor": "critical_low_attendance",
                "impact": 0.9,
                "description": f"Only {attendance_rate*100:.0f}% attendance rate (critical)"
            })
        elif attendance_rate < 0.75:
            risk_factors.append({
                "factor": "low_attendance",
                "impact": 0.7,
                "description": f"Low attendance rate: {attendance_rate*100:.0f}%"
            })

        # Check consecutive absences
        consecutive_absences = features.get("consecutive_absences_current", 0)
        if consecutive_absences >= 5:
            risk_factors.append({
                "factor": "extended_absence",
                "impact": 0.95,
                "description": f"{consecutive_absences} consecutive days absent"
            })
        elif consecutive_absences >= 3:
            risk_factors.append({
                "factor": "consecutive_absence",
                "impact": 0.7,
                "description": f"{consecutive_absences} consecutive days absent"
            })

        # Check compliance score
        compliance_score = features.get("compliance_score_current", 100)
        if compliance_score < 60:
            risk_factors.append({
                "factor": "very_low_compliance",
                "impact": 0.85,
                "description": f"Compliance score critically low: {compliance_score:.0f}/100"
            })
        elif compliance_score < 70:
            risk_factors.append({
                "factor": "low_compliance",
                "impact": 0.6,
                "description": f"Compliance score below acceptable: {compliance_score:.0f}/100"
            })

        # Check recent violation trend
        violation_trend = features.get("violation_trend", 0)
        if violation_trend > 0.1:
            risk_factors.append({
                "factor": "increasing_violations",
                "impact": 0.6,
                "description": "Violation rate increasing in recent days"
            })

        # Check warnings
        warning_count = features.get("warning_count_30d", 0)
        if warning_count >= 3:
            risk_factors.append({
                "factor": "multiple_warnings",
                "impact": 0.7,
                "description": f"{warning_count} disciplinary warnings in last 30 days"
            })
        elif warning_count >= 1:
            risk_factors.append({
                "factor": "recent_warning",
                "impact": 0.4,
                "description": f"{warning_count} warning(s) in last 30 days"
            })

        # Check specific high-risk PPE items
        high_risk_items = predictions.get("high_risk_items", [])
        if high_risk_items:
            items_str = ", ".join(high_risk_items)
            risk_factors.append({
                "factor": "high_risk_ppe_items",
                "impact": 0.5,
                "description": f"Frequent violations for: {items_str}"
            })

        # Check if new worker (might need training)
        tenure_days = features.get("tenure_days", 999)
        if tenure_days < 30 and violations_30d > 0:
            risk_factors.append({
                "factor": "new_worker_violations",
                "impact": 0.4,
                "description": f"New worker ({tenure_days} days) with violations - may need additional training"
            })

        # Check shift consistency
        shift_consistency = features.get("shift_consistency", 1.0)
        if shift_consistency < 0.7:
            risk_factors.append({
                "factor": "inconsistent_shifts",
                "impact": 0.3,
                "description": f"Irregular shift attendance pattern ({shift_consistency*100:.0f}% consistency)"
            })

        # Sort by impact (highest first)
        risk_factors.sort(key=lambda x: x["impact"], reverse=True)

        # If no specific factors identified but risk is high
        if not risk_factors and overall_risk_score > 60:
            risk_factors.append({
                "factor": "general_risk",
                "impact": 0.5,
                "description": "Multiple moderate risk indicators across various factors"
            })

        # Limit to top 5 risk factors
        return risk_factors[:5]

    def generate_recommendations(self, risk_factors: List[Dict[str, Any]], predictions: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on risk factors.

        Args:
            risk_factors: List of identified risk factors
            predictions: Model predictions

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Extract factor types
        factor_types = {rf["factor"] for rf in risk_factors}

        # Violation-related recommendations
        if any(f in factor_types for f in ["high_violation_count", "moderate_violation_count"]):
            recommendations.append("Schedule mandatory PPE safety training session")

            high_risk_items = predictions.get("high_risk_items", [])
            if high_risk_items:
                items_str = ", ".join(high_risk_items)
                recommendations.append(f"Focus training on proper use of: {items_str}")

        # Attendance-related recommendations
        if "critical_low_attendance" in factor_types or "low_attendance" in factor_types:
            recommendations.append("Conduct attendance counseling session")
            recommendations.append("Review work-life balance and personal issues")

        # Consecutive absence recommendations
        if "extended_absence" in factor_types or "consecutive_absence" in factor_types:
            recommendations.append("Immediate check-in required - verify worker wellbeing")
            recommendations.append("Consider temporary reassignment to less critical zones")

        # Compliance recommendations
        if "very_low_compliance" in factor_types:
            recommendations.append("IMMEDIATE ACTION: Assign safety buddy for supervision")
            recommendations.append("Restrict access to high-risk zones until improvement")
            recommendations.append("Weekly safety compliance reviews with supervisor")

        # Warning-related recommendations
        if "multiple_warnings" in factor_types:
            recommendations.append("Escalate to formal performance improvement plan")
            recommendations.append("Require sign-off from Safety Officer for zone access")

        # New worker recommendations
        if "new_worker_violations" in factor_types:
            recommendations.append("Provide additional onboarding and safety orientation")
            recommendations.append("Assign experienced mentor for first 60 days")

        # General recommendations if critical
        if predictions.get("risk_category") == "critical" and not recommendations:
            recommendations.append("CRITICAL: Immediate intervention required")
            recommendations.append("Schedule meeting with worker and Safety Officer")
            recommendations.append("Develop personalized safety improvement plan")

        return recommendations[:5]  # Limit to top 5 recommendations
