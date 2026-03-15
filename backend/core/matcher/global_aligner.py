"""Global alignment optimizer for segment candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.segment import AlignmentStatus, MatchCandidate, Segment, SegmentType


@dataclass
class _State:
    score: float
    prev_idx: Optional[int]
    used_skip: bool


class GlobalAlignmentOptimizer:
    """Pick the best candidate path across the full narration timeline."""

    def __init__(
        self,
        auto_accept_threshold: float = 0.82,
        review_threshold: float = 0.65,
        backtrack_penalty: float = 0.3,
        duplicate_scene_penalty: float = 0.4,
        stagnation_penalty: float = 0.2,
        jump_penalty: float = 0.15,
    ):
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold
        self.backtrack_penalty = backtrack_penalty
        self.duplicate_scene_penalty = duplicate_scene_penalty
        self.stagnation_penalty = stagnation_penalty
        self.jump_penalty = jump_penalty

    def optimize(self, segments: list[Segment], allow_non_sequential: bool = True) -> list[dict]:
        """Return a selected candidate and alignment metadata for each segment."""

        if not segments:
            return []

        option_lists: list[list[Optional[MatchCandidate]]] = []
        for segment in segments:
            candidates = list(segment.match_candidates or [])
            candidates.sort(key=lambda item: item.score, reverse=True)
            option_lists.append(candidates + [None])

        dp: list[list[_State]] = []
        for seg_idx, options in enumerate(option_lists):
            layer: list[_State] = []
            for opt_idx, option in enumerate(options):
                base_score = self._base_score(option, segments[seg_idx])
                if seg_idx == 0:
                    layer.append(_State(score=base_score, prev_idx=None, used_skip=option is None))
                    continue

                best_state: Optional[_State] = None
                for prev_idx, prev_state in enumerate(dp[seg_idx - 1]):
                    transition = self._transition_score(
                        option_lists[seg_idx - 1][prev_idx],
                        option,
                        segments[seg_idx - 1],
                        segments[seg_idx],
                        allow_non_sequential=allow_non_sequential,
                    )
                    total = prev_state.score + base_score + transition
                    if best_state is None or total > best_state.score:
                        best_state = _State(
                            score=total,
                            prev_idx=prev_idx,
                            used_skip=prev_state.used_skip or option is None,
                        )
                layer.append(best_state or _State(score=base_score, prev_idx=None, used_skip=option is None))
            dp.append(layer)

        last_options = dp[-1]
        best_idx = max(range(len(last_options)), key=lambda idx: last_options[idx].score)
        chosen_indices = [best_idx]
        for seg_idx in range(len(segments) - 1, 0, -1):
            best_idx = dp[seg_idx][best_idx].prev_idx or 0
            chosen_indices.append(best_idx)
        chosen_indices.reverse()

        results: list[dict] = []
        for seg_idx, option_idx in enumerate(chosen_indices):
            candidate = option_lists[seg_idx][option_idx]
            results.append(self._build_result(segments[seg_idx], candidate))
        return results

    def _base_score(self, candidate: Optional[MatchCandidate], segment: Segment) -> float:
        if candidate is None:
            if segment.segment_type == SegmentType.NON_MOVIE:
                return 0.05
            return -2.0
        score = candidate.score or candidate.confidence or 0.0
        # 高置信候选额外奖励，强化 DP 对高质量匹配的偏好
        if score >= 0.92:
            score += 0.06 * min(1.0, (score - 0.92) / 0.08)
        elif score >= 0.85:
            score += 0.03 * (score - 0.85) / 0.07
        score += min(0.08, max(0.0, candidate.stability_score) * 0.10)
        score -= min(0.10, max(0.0, candidate.low_info_ratio) * 0.12)
        return score

    def _transition_score(
        self,
        previous: Optional[MatchCandidate],
        current: Optional[MatchCandidate],
        previous_segment: Segment,
        current_segment: Segment,
        allow_non_sequential: bool,
    ) -> float:
        if previous is None or current is None:
            return -0.05 if current is None else 0.0

        score = 0.0
        backtrack = max(0.0, previous.start - current.start)
        if backtrack > 0:
            score -= self.backtrack_penalty * (1.0 + min(backtrack / 45.0, 2.0))

        overlap = max(0.0, min(previous.end, current.end) - max(previous.start, current.start))
        if overlap > 0:
            score -= self.duplicate_scene_penalty * (1.0 + overlap / max(1.0, current.end - current.start))

        narr_delta = current_segment.narration_start - previous_segment.narration_start
        movie_delta = current.start - previous.start
        timeline_error = abs(movie_delta - narr_delta)
        if not allow_non_sequential:
            score -= min(1.5, timeline_error / 50.0)
        else:
            score -= min(0.8, timeline_error / 120.0)

        if narr_delta > 0.75:
            pace_ratio = movie_delta / max(0.25, narr_delta)
            min_ratio = 0.38
            max_ratio = 2.6 if allow_non_sequential else 1.8
            if pace_ratio < min_ratio:
                score -= self.stagnation_penalty * min(1.2, (min_ratio - pace_ratio) / min_ratio * 1.2)
            elif pace_ratio > max_ratio:
                score -= self.jump_penalty * min(1.2, (pace_ratio - max_ratio) / max_ratio * 1.2)

            narr_gap = max(0.0, current_segment.narration_start - previous_segment.narration_end)
            movie_gap = max(0.0, current.start - previous.end)
            gap_error = abs(movie_gap - narr_gap)
            score -= min(0.45, gap_error / max(8.0, narr_delta * 1.8))

        if current.start >= previous.end:
            score += 0.08
        return score

    def _build_result(self, segment: Segment, candidate: Optional[MatchCandidate]) -> dict:
        if segment.skip_matching:
            return {
                "segment_id": segment.id,
                "candidate": None,
                "alignment_status": AlignmentStatus.SKIPPED,
                "review_required": False,
                "match_confidence": 0.0,
            }

        if segment.segment_type == SegmentType.NON_MOVIE:
            return {
                "segment_id": segment.id,
                "candidate": None,
                "alignment_status": AlignmentStatus.NON_MOVIE,
                "review_required": False,
                "match_confidence": 0.0,
            }

        if candidate is None:
            return {
                "segment_id": segment.id,
                "candidate": None,
                "alignment_status": AlignmentStatus.UNMATCHED,
                "review_required": True,
                "match_confidence": 0.0,
            }

        confidence = candidate.confidence or candidate.score
        if segment.is_manual_match:
            status = AlignmentStatus.MANUAL
            review_required = False
        elif confidence >= self.auto_accept_threshold:
            status = AlignmentStatus.AUTO_ACCEPTED
            review_required = False
        elif confidence >= self.review_threshold:
            status = AlignmentStatus.NEEDS_REVIEW
            review_required = True
        else:
            status = AlignmentStatus.NEEDS_REVIEW
            review_required = True

        return {
            "segment_id": segment.id,
            "candidate": candidate,
            "alignment_status": status,
            "review_required": review_required,
            "match_confidence": confidence,
            "estimated_boundary_error": max(candidate.duration_gap / 2.0 + max(0.0, 1.0 - candidate.stability_score), 0.0),
        }
