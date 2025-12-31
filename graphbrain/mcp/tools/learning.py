"""Active learning and quality metrics tools for MCP server.

Provides tools for intelligent sample selection and quality monitoring.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_learning_tools(server: FastMCP):
    """Register active learning and metrics tools with the MCP server."""

    @server.tool(
        name="get_learning_candidates",
        description="""
Get candidates for active learning review.

Uses uncertainty sampling to identify the most informative samples
for human labeling. Prioritizes samples where:
- Confidence is near the decision threshold
- High frequency predicates
- Underrepresented classes

Args:
    limit: Maximum candidates to return (default 20)
    strategy: Sampling strategy - "hybrid" (default), "uncertainty",
              "diversity", "margin", "entropy", or "random"
    min_frequency: Minimum predicate frequency (default 1)

Returns:
  - candidates: list of {predicate, confidence, suggested_class, score, ...}
  - strategy: the sampling strategy used
  - total: number of candidates returned
""",
    )
    async def get_learning_candidates(
        limit: int = 20,
        strategy: str = "hybrid",
        min_frequency: int = 1,
    ) -> dict:
        """Get active learning candidates."""
        logger.debug(f"get_learning_candidates: limit={limit}, strategy={strategy}")

        from graphbrain.agents.learning import ActiveLearningSampler, SamplingStrategy

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        backend = lifespan_data.get("backend")

        if not backend:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
                "candidates": [],
            }

        # Parse strategy
        try:
            sampling_strategy = SamplingStrategy(strategy)
        except ValueError:
            sampling_strategy = SamplingStrategy.HYBRID

        sampler = ActiveLearningSampler(
            backend=backend,
            strategy=sampling_strategy,
        )

        candidates = []
        for candidate in sampler.get_candidates(limit=limit, min_frequency=min_frequency):
            candidates.append({
                "predicate": candidate.predicate,
                "current_class": candidate.current_class,
                "suggested_class": candidate.suggested_class,
                "confidence": candidate.confidence,
                "informativeness_score": candidate.informativeness_score,
                "frequency": candidate.frequency,
                "margin": candidate.margin,
                "entropy": candidate.entropy,
            })

        logger.info(f"get_learning_candidates: found {len(candidates)} candidates")
        return {
            "status": "success",
            "candidates": candidates,
            "strategy": strategy,
            "total": len(candidates),
        }

    @server.tool(
        name="get_improvement_suggestions",
        description="""
Get suggestions for improving classification quality.

Analyzes the current classification state and generates
actionable suggestions for improvement, such as:
- Add seed predicates to sparse classes
- Review low-confidence classifications
- Resolve predicate conflicts
- Adjust thresholds

Returns:
  - suggestions: list of improvement suggestions with priority
  - summary: counts by type and priority
""",
    )
    async def get_improvement_suggestions() -> dict:
        """Get improvement suggestions."""
        logger.debug("get_improvement_suggestions")

        from graphbrain.agents.learning import SuggestionEngine

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        backend = lifespan_data.get("backend")

        if not backend:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
                "suggestions": [],
            }

        engine = SuggestionEngine(backend)
        suggestions = engine.analyze()

        result = []
        for s in suggestions:
            result.append(s.to_dict())

        summary = engine.get_summary()

        logger.info(f"get_improvement_suggestions: {len(result)} suggestions")
        return {
            "status": "success",
            "suggestions": result,
            "summary": summary,
        }

    @server.tool(
        name="quality_dashboard",
        description="""
Get classification quality dashboard data.

Returns comprehensive metrics about classification quality:
- Health score (0-100%)
- Classification metrics (confidence, coverage)
- Feedback metrics (pending, approval rate)
- Alerts and recommendations

Args:
    format: Output format - "json" (default) or "text"

Returns:
  - health: {score, status}
  - metrics: comprehensive metrics
  - alerts: list of warnings
  - recommendations: list of suggested actions
""",
    )
    async def quality_dashboard(
        format: str = "json",
    ) -> dict:
        """Get quality dashboard."""
        logger.debug(f"quality_dashboard: format={format}")

        from graphbrain.agents.metrics import QualityDashboard

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        backend = lifespan_data.get("backend")

        if not backend:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
            }

        dashboard = QualityDashboard(backend)

        if format == "text":
            text_output = dashboard.render_text()
            logger.info("quality_dashboard: rendered text format")
            return {
                "status": "success",
                "format": "text",
                "dashboard": text_output,
            }

        data = dashboard.to_dict()

        logger.info(f"quality_dashboard: health={data['health']['score']:.0%}")
        return {
            "status": "success",
            "format": "json",
            **data,
        }

    @server.tool(
        name="suggest_batch_size",
        description="""
Suggest optimal batch size for next labeling round.

Uses heuristics based on dataset size and current accuracy
to recommend how many samples to review.

Args:
    total_predicates: Total number of predicates in the system
    current_accuracy: Estimated current accuracy (0.0-1.0, default 0.7)

Returns:
  - suggested_batch_size: recommended number of samples to review
  - reasoning: explanation of the suggestion
""",
    )
    async def suggest_batch_size(
        total_predicates: int,
        current_accuracy: float = 0.7,
    ) -> dict:
        """Suggest batch size for labeling."""
        logger.debug(f"suggest_batch_size: total={total_predicates}, accuracy={current_accuracy}")

        from graphbrain.agents.learning import ActiveLearningSampler

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        backend = lifespan_data.get("backend")

        if not backend:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
            }

        sampler = ActiveLearningSampler(backend)
        suggested = sampler.suggest_batch_size(total_predicates, current_accuracy)

        reasoning = []
        if current_accuracy < 0.6:
            reasoning.append("Low accuracy suggests larger batches for faster improvement")
        elif current_accuracy > 0.9:
            reasoning.append("High accuracy allows smaller, focused batches")
        else:
            reasoning.append("Moderate accuracy suggests balanced batch size")

        if total_predicates < 100:
            reasoning.append("Small dataset allows proportionally larger batches")
        elif total_predicates > 1000:
            reasoning.append("Large dataset benefits from sampling-based review")

        logger.info(f"suggest_batch_size: {suggested}")
        return {
            "status": "success",
            "suggested_batch_size": suggested,
            "total_predicates": total_predicates,
            "current_accuracy": current_accuracy,
            "reasoning": " ".join(reasoning),
        }

    @server.tool(
        name="get_confidence_distribution",
        description="""
Get distribution of classification confidence scores.

Returns a histogram of confidence scores, useful for
understanding overall classification quality.

Args:
    bins: Number of histogram bins (default 10)

Returns:
  - histogram: list of {range, count} for each bin
  - total: total classifications
  - avg_confidence: average confidence score
""",
    )
    async def get_confidence_distribution(
        bins: int = 10,
    ) -> dict:
        """Get confidence distribution."""
        logger.debug(f"get_confidence_distribution: bins={bins}")

        from graphbrain.agents.metrics import MetricsCollector

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        backend = lifespan_data.get("backend")

        if not backend:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
            }

        collector = MetricsCollector(backend)
        histogram = collector.get_confidence_histogram(bins=bins)

        # Get total and average
        metrics = collector.collect_all()

        logger.info(f"get_confidence_distribution: {len(histogram)} bins")
        return {
            "status": "success",
            "histogram": histogram,
            "total": metrics.classification.total_classifications,
            "avg_confidence": metrics.classification.avg_confidence,
        }
