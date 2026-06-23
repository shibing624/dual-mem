# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Attentional Gate — LLM scores novelty / biographical_relevance / emotional_arousal;
falls back to keyword heuristics when LLM is unavailable or fails. Vector novelty (1 - max_sim)
is fused after text scoring.
"""
import logging
from dataclasses import dataclass

from dual_mem.providers.llm import LLMClient, merge_gate_results
from dual_mem.sdk_models import GateResult

logger = logging.getLogger("dual_mem.agent.gate")


@dataclass
class GateConfig:
    """Attentional gate weights and thresholds."""

    threshold: float = 0.3
    novelty_weight: float = 0.40
    relevance_weight: float = 0.40
    arousal_weight: float = 0.20
    novelty_similarity_cap: float = 0.95
    llm_temperature: float = 0.1
    heuristic_shortcircuit: bool = True
    shortcircuit_novelty: float = 0.8
    shortcircuit_relevance: float = 0.5


class AttentionalGate:
    """Decide whether content is worth deep extraction (L2+).

    Primary path: one LLM call for three semantic dimensions.
    Fallback: keyword/length heuristics when ``llm`` is None or the call fails.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.3,
        llm: LLMClient | None = None,
        config: GateConfig | None = None,
    ):
        self.llm = llm
        self.config = config or GateConfig(threshold=threshold)
        if config is None:
            self.config.threshold = threshold

    async def evaluate(
        self,
        *,
        content: str,
        existing_similarities: list[dict] | list[list[dict]] | None = None,
        agent_context: str | None = None,
    ) -> GateResult:
        """Score ``content`` and decide pass/reject.

        ``existing_similarities`` accepts single-turn ``[{"node_id", "score"}, ...]`` or
        multi-turn ``[turn1_hits, turn2_hits, ...]``; multi-turn novelty uses max across turns.
        """
        text = (content or "").strip()
        if not text:
            return GateResult(
                passed=False,
                gate_score=0.0,
                novelty=0.0,
                biographical_relevance=0.0,
                emotional_arousal=0.0,
                reason="Empty content",
                scoring_method="rule",
            )

        shortcircuited = await self.try_shortcircuit_pass(
            content=text,
            existing_similarities=existing_similarities,
        )
        if shortcircuited is not None:
            return shortcircuited

        llm_scores = await self._llm_score(text, agent_context) if self.llm is not None else None
        if llm_scores is None and self.llm is not None:
            logger.warning("Gate LLM scoring failed, falling back to heuristic")

        return await self.finalize_from_llm(
            content=text,
            llm_scores=llm_scores,
            existing_similarities=existing_similarities,
            scoring_method="llm" if llm_scores is not None else "heuristic",
        )

    async def try_shortcircuit_pass(
        self,
        *,
        content: str,
        existing_similarities: list[dict] | list[list[dict]] | None = None,
    ) -> GateResult | None:
        """If vector novelty + heuristic relevance are clearly high, PASS without gate LLM."""
        if not self.config.heuristic_shortcircuit:
            return None
        text = (content or "").strip()
        if not text:
            return None

        vector_novelty = self._vector_novelty_from_sims(existing_similarities)
        _, h_relevance, h_arousal = self._heuristic_score(text)
        if vector_novelty < self.config.shortcircuit_novelty:
            return None
        if h_relevance < self.config.shortcircuit_relevance:
            return None

        h_novelty, _, _ = self._heuristic_score(text)
        return await self.finalize_from_llm(
            content=text,
            llm_scores={
                "novelty": min(h_novelty, vector_novelty),
                "biographical_relevance": h_relevance,
                "emotional_arousal": h_arousal,
                "reason": "heuristic short-circuit",
            },
            existing_similarities=existing_similarities,
            scoring_method="heuristic_shortcircuit",
        )

    async def finalize_from_llm(
        self,
        *,
        content: str,
        llm_scores: dict | None,
        existing_similarities: list[dict] | list[list[dict]] | None = None,
        scoring_method: str = "llm",
        llm_reason: str = "",
    ) -> GateResult:
        """Fuse LLM/heuristic dimension scores with vector novelty into a GateResult."""
        text = (content or "").strip()
        if llm_scores is None:
            novelty, relevance, arousal = self._heuristic_score(text)
            scoring_method = "heuristic"
            llm_reason = llm_reason or "heuristic fallback"
        else:
            novelty = _clamp(float(llm_scores.get("novelty", 0.5)), 0.0, 1.0)
            relevance = _clamp(
                float(llm_scores.get("biographical_relevance", 0.0)), 0.0, 1.0
            )
            arousal = _clamp(float(llm_scores.get("emotional_arousal", 0.0)), 0.0, 1.0)
            if not llm_reason:
                llm_reason = str(llm_scores.get("reason", ""))

        top_id: str | None = None
        top_score = 0.0
        if existing_similarities:
            first = existing_similarities[0]
            if isinstance(first, list):
                turns: list[list[dict]] = existing_similarities  # type: ignore[assignment]
                best_novelty = 0.0
                for turn_sims in turns:
                    if turn_sims:
                        vec_n, tid, tscore = self._vector_novelty(turn_sims)
                        if vec_n > best_novelty:
                            best_novelty = vec_n
                            top_id, top_score = tid, tscore
                    else:
                        best_novelty = 1.0
                novelty = min(novelty, best_novelty)
            else:
                sims: list[dict] = existing_similarities  # type: ignore[assignment]
                vec_n, top_id, top_score = self._vector_novelty(sims)
                novelty = min(novelty, vec_n)

        cfg = self.config
        gate_score = (
            cfg.novelty_weight * novelty
            + cfg.relevance_weight * relevance
            + cfg.arousal_weight * arousal
        )
        passed = gate_score > cfg.threshold

        if passed:
            top_dims = []
            if novelty > 0.5:
                top_dims.append(f"novelty={novelty:.2f}")
            if relevance > 0.5:
                top_dims.append(f"relevance={relevance:.2f}")
            if arousal > 0.5:
                top_dims.append(f"arousal={arousal:.2f}")
            reason = f"PASS (score={gate_score:.3f}): {', '.join(top_dims) or 'combined'}"
        else:
            reason = f"REJECT (score={gate_score:.3f} < θ={cfg.threshold})"
        if llm_reason:
            reason += f" | {llm_reason}"

        return GateResult(
            passed=passed,
            gate_score=gate_score,
            novelty=novelty,
            biographical_relevance=relevance,
            emotional_arousal=arousal,
            reason=reason,
            scoring_method=scoring_method,
            top_similar_id=top_id,
            top_similar_score=top_score,
        )

    async def _llm_score(
        self,
        content: str,
        agent_context: str | None = None,
    ) -> dict | None:
        """Primary LLM scoring path; returns parsed scores or None on failure."""
        from dual_mem.agent import prompts

        if self.llm is None:
            return None

        if agent_context:
            context_section = prompts.pick(
                prompts.GATE_CONTEXT_ZH,
                prompts.GATE_CONTEXT_EN,
                content,
            ).format(agent_context=agent_context.strip())
        else:
            context_section = ""

        def _build_system(chunk: str) -> str:
            return prompts.pick(prompts.GATE_ZH, prompts.GATE_EN, chunk).format(
                content=chunk,
                context_section=context_section,
            )

        try:
            data = await self.llm.chat_json_for_content(
                content=content,
                build_system=_build_system,
                merge_results=merge_gate_results,
                temperature=self.config.llm_temperature,
            )
        except Exception as exc:
            logger.warning("gate LLM scoring failed: %s", exc)
            return None
        if not isinstance(data, dict):
            return None
        try:
            return {
                "novelty": _clamp(float(data.get("novelty", 0.5)), 0.0, 1.0),
                "biographical_relevance": _clamp(
                    float(data.get("biographical_relevance", 0.0)), 0.0, 1.0
                ),
                "emotional_arousal": _clamp(
                    float(data.get("emotional_arousal", 0.0)), 0.0, 1.0
                ),
                "reason": str(data.get("reason", "")),
            }
        except (TypeError, ValueError):
            return None

    def _heuristic_score(self, content: str) -> tuple[float, float, float]:
        """Fallback scoring when LLM is unavailable or fails."""
        return (
            self._heuristic_novelty(content),
            self._heuristic_relevance(content),
            self._heuristic_arousal(content),
        )

    @staticmethod
    def _heuristic_novelty(content: str) -> float:
        """Length-based proxy for information density."""
        n = len(content)
        if n <= 3:
            return 0.3
        if n <= 8:
            return 0.5
        return min(0.8, 0.5 + n / 200.0)

    @staticmethod
    def _heuristic_relevance(content: str) -> float:
        """Keyword + first-person scoring for biographical relevance."""
        biographical_keywords = [
            "我是", "我叫", "我的名字", "今年", "岁了", "年龄",
            "我在", "我住", "搬到", "搬家", "住在",
            "工作", "公司", "上班", "入职", "离职", "辞职", "跳槽",
            "职业", "岗位", "薪资", "工资", "同事", "老板", "领导",
            "学校", "大学", "毕业", "专业", "学历",
            "家人", "父亲", "母亲", "爸", "妈", "老婆", "老公",
            "女朋友", "男朋友", "对象", "孩子", "儿子", "女儿",
            "兄弟", "姐妹", "爷爷", "奶奶",
            "喜欢", "不喜欢", "讨厌", "偏好", "爱好",
            "最爱", "最喜欢", "受不了", "过敏", "忌口",
            "习惯", "每天", "总是", "从来",
            "生病", "手术", "体检", "身体",
            "结婚", "离婚", "怀孕", "生日", "旅行", "出差",
            "买房", "买车", "考试",
            "觉得", "认为", "价值观", "性格",
            "打算", "计划", "准备", "想要", "下周", "下个月", "明年",
        ]
        haystack = content.lower()
        hit_count = sum(1 for kw in biographical_keywords if kw in haystack)
        first_person = any(m in haystack for m in ["我", "我的", "我们", "自己"])
        base_score = min(1.0, hit_count * 0.18)
        length_bonus = min(0.2, len(content) / 500.0 * 0.2)
        fp_bonus = 0.1 if first_person else 0.0
        return min(1.0, base_score + length_bonus + fp_bonus)

    @staticmethod
    def _heuristic_arousal(content: str) -> float:
        """Emotion-keyword scoring with light punctuation boost."""
        emotional_markers = [
            ("受不了", 0.85), ("崩溃", 0.9), ("气死", 0.9),
            ("想哭", 0.8), ("焦虑", 0.7), ("紧张", 0.6),
            ("害怕", 0.7), ("难过", 0.6), ("失望", 0.6),
            ("郁闷", 0.6), ("烦死了", 0.8), ("压力大", 0.7),
            ("噩梦", 0.8), ("太开心", 0.8), ("太棒了", 0.7),
            ("超兴奋", 0.8), ("好激动", 0.7), ("终于", 0.6),
            ("成功", 0.6), ("感动", 0.7), ("好幸福", 0.7),
            ("开心", 0.4), ("高兴", 0.4), ("无聊", 0.3),
            ("累了", 0.4), ("辛苦", 0.5),
        ]
        haystack = content.lower()
        max_arousal = 0.0
        for keyword, arousal in emotional_markers:
            if keyword in haystack:
                max_arousal = max(max_arousal, arousal)

        excl = content.count("!") + content.count("！")
        ques = content.count("?") + content.count("？")
        punct_boost = min(0.2, (excl + ques * 0.5) * 0.05)
        return min(1.0, max_arousal + punct_boost)

    def _vector_novelty(self, sims: list[dict]) -> tuple[float, str | None, float]:
        """novelty = 1 - min(max_sim, cap); return (novelty, top_id, top_sim)."""
        max_sim = 0.0
        top_id: str | None = None
        for item in sims:
            score = float(item.get("score", 0.0))
            if score > max_sim:
                max_sim = score
                top_id = item.get("node_id")
        cap = self.config.novelty_similarity_cap
        novelty = max(0.0, min(1.0, 1.0 - min(max_sim, cap)))
        return novelty, top_id, max_sim

    def _vector_novelty_from_sims(
        self,
        existing_similarities: list[dict] | list[list[dict]] | None,
    ) -> float:
        """Max vector novelty across single- or multi-turn similarity hits."""
        if not existing_similarities:
            return 1.0
        first = existing_similarities[0]
        if isinstance(first, list):
            turns: list[list[dict]] = existing_similarities  # type: ignore[assignment]
            best = 0.0
            for turn_sims in turns:
                if turn_sims:
                    vec_n, _, _ = self._vector_novelty(turn_sims)
                    best = max(best, vec_n)
                else:
                    best = max(best, 1.0)
            return best
        sims: list[dict] = existing_similarities  # type: ignore[assignment]
        vec_n, _, _ = self._vector_novelty(sims)
        return vec_n


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))
