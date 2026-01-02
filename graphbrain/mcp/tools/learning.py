"""Active learning and quality metrics tools for MCP server.

Provides tools for intelligent sample selection and quality monitoring.
"""

import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from graphbrain.mcp.utils import validate_positive_int, validate_limit, validate_threshold

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
        # Validate inputs
        if error := validate_limit(limit, max_limit=500):
            return error
        if error := validate_positive_int(min_frequency, "min_frequency", allow_zero=True):
            return error

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
        # Validate inputs
        if error := validate_positive_int(total_predicates, "total_predicates"):
            return error
        if error := validate_threshold(current_accuracy, "current_accuracy"):
            return error

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
        # Validate inputs
        if error := validate_positive_int(bins, "bins"):
            return error

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

    @server.tool(
        name="apply_pending_feedback",
        description="""
Apply all pending feedback and update classifications.

This is the key orchestration tool that closes the feedback loop:
1. Gets all pending feedback items
2. Applies each correction (moves predicates to correct classes)
3. Optionally triggers re-evaluation of related edges
4. Returns a summary of what was applied

Use this after reviewing and submitting feedback via submit_feedback
to actually update the classification system.

Args:
    limit: Maximum feedback items to apply (default 50)
    update_frequencies: If True, update predicate frequency counts

Returns:
  - applied: list of applied feedback items
  - failed: list of items that failed to apply
  - total_applied: count of successfully applied items
  - predicates_updated: count of predicates moved to new classes
""",
    )
    async def apply_pending_feedback(
        limit: int = 50,
        update_frequencies: bool = True,
    ) -> dict:
        """Apply pending feedback to update classifications."""
        # Validate inputs
        if error := validate_limit(limit, max_limit=500):
            return error

        logger.debug(f"apply_pending_feedback: limit={limit}")

        from graphbrain.classification.models import PredicateBankEntry

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data.get("repo")

        if not repo:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
            }

        # Get pending feedback
        pending = list(repo.get_pending_feedback(limit=limit))
        if not pending:
            logger.info("apply_pending_feedback: no pending feedback")
            return {
                "status": "success",
                "applied": [],
                "failed": [],
                "total_applied": 0,
                "predicates_updated": 0,
                "message": "No pending feedback to apply",
            }

        applied = []
        failed = []
        predicates_updated = 0

        for feedback in pending:
            try:
                # Get the correct class
                correct_class = repo.get_class_by_name(feedback.correct_class)
                if not correct_class:
                    # Try by ID
                    correct_class = repo.get_class(feedback.correct_class)

                if not correct_class:
                    failed.append({
                        "review_id": feedback.review_id,
                        "predicate": feedback.predicate,
                        "reason": f"Class '{feedback.correct_class}' not found",
                    })
                    continue

                # Create or update predicate entry in correct class
                entry = PredicateBankEntry(
                    class_id=correct_class.id,
                    lemma=feedback.predicate,
                    is_seed=False,
                    similarity_score=feedback.confidence_adjustment if feedback.confidence_adjustment else 0.8,
                )
                repo.save_predicate(entry)
                predicates_updated += 1

                # Mark feedback as applied
                feedback.status = "applied"
                repo.save_feedback(feedback)

                applied.append({
                    "review_id": feedback.review_id,
                    "predicate": feedback.predicate,
                    "new_class": correct_class.name,
                    "new_class_id": correct_class.id,
                })

            except Exception as e:
                logger.warning(f"Failed to apply feedback {feedback.review_id}: {e}")
                failed.append({
                    "review_id": feedback.review_id,
                    "predicate": feedback.predicate,
                    "reason": str(e),
                })

        logger.info(f"apply_pending_feedback: applied {len(applied)}, failed {len(failed)}")
        return {
            "status": "success",
            "applied": applied,
            "failed": failed,
            "total_applied": len(applied),
            "predicates_updated": predicates_updated,
        }

    @server.tool(
        name="create_feedback_from_candidates",
        description="""
Create feedback items from learning candidates with decisions.

Takes learning candidates that have been reviewed and creates
feedback entries for each one. This bridges the gap between
active learning (identifying what to review) and the feedback
system (recording decisions).

Args:
    decisions: List of {predicate, correct_class} decisions
    reviewer_id: Optional identifier for the reviewer

Returns:
  - feedback_created: list of created feedback items
  - total: count of feedback items created
""",
    )
    async def create_feedback_from_candidates(
        decisions: list,
        reviewer_id: Optional[str] = None,
    ) -> dict:
        """Create feedback from reviewed learning candidates."""
        logger.debug(f"create_feedback_from_candidates: {len(decisions)} decisions")

        import uuid
        from datetime import datetime
        from graphbrain.classification.models import ClassificationFeedback

        ctx = server.get_context()
        lifespan_data = ctx.request_context.lifespan_context
        repo = lifespan_data.get("repo")

        if not repo:
            return {
                "status": "error",
                "code": "service_unavailable",
                "message": "Classification backend not available",
            }

        feedback_created = []
        for decision in decisions:
            predicate = decision.get("predicate")
            correct_class = decision.get("correct_class")
            original_class = decision.get("original_class", "unclassified")

            if not predicate or not correct_class:
                continue

            review_id = f"fb_{uuid.uuid4().hex[:8]}"
            feedback = ClassificationFeedback(
                review_id=review_id,
                predicate=predicate,
                original_class=original_class,
                correct_class=correct_class,
                confidence_adjustment=decision.get("confidence"),
                reviewer_id=reviewer_id,
                status="pending",
                created_at=datetime.now(),
            )
            repo.save_feedback(feedback)

            feedback_created.append({
                "review_id": review_id,
                "predicate": predicate,
                "correct_class": correct_class,
            })

        logger.info(f"create_feedback_from_candidates: created {len(feedback_created)} feedback items")
        return {
            "status": "success",
            "feedback_created": feedback_created,
            "total": len(feedback_created),
        }
