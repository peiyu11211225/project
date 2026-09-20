import numpy as np
from collections import defaultdict


class AICoach:

    def __init__(self):

        # 關節權重（挑球以手腕手感為主，肩膀轉動重要性降低）
        self.joint_weights = {
            "16": 3.5,  # wrist（挑球核心發力點，權重提高）
            "14": 1.8,  # elbow（略降，非全力伸展型動作）
            "12": 1.0,  # shoulder（挑球不需大幅轉肩，權重降低）
            "11": 1.2,
            "23": 1.0,
            "24": 1.0
        }

        # threshold（挑球動作幅度小，誤差數值通常較小，先收緊判定）
        self.good_threshold = 0.16
        self.bad_threshold = 0.19

        # 教練扣分機制
        self.phase_penalty = {
            "prep": {"bad": 10, "medium": 5, "good": 0},
            "impact": {"bad": 8, "medium": 4, "good": 0},
            "follow": {"bad": 5, "medium": 2, "good": 0}
        }

        # 收拍放寬係數（收拍容易被遮住，誤判率高，故大幅放寬）
        self.relaxed_factor = {
            "prep": 1.1,
            "impact": 1.1,
            "follow": 1.6   # 挑球收拍動作本身較小，放寬係數略降於高遠球版本
        }

    # =========================================================
    # 主入口
    # =========================================================
    def generate_feedback(self, feat_std, feat_usr, path, curve, final_score):

        if len(path) == 0:
            return ["無法分析動作"], "無法評估", 0

        segments = self._split_phases(path, feat_std, feat_usr)

        phase_errors = self._compute_phase_errors(
            feat_std, feat_usr, path, segments
        )

        feedback = []
        total_penalty = 0

        for phase in ["prep", "impact", "follow"]:

            errors = phase_errors.get(phase, {})

            if not errors:
                feedback.append(self._fallback_feedback(phase))
                continue

            msgs, level = self._generate_phase_feedback(phase, errors)

            feedback.append(msgs)

            total_penalty += self.phase_penalty[phase][level]

        overall = self._overall_assessment(final_score)

        return feedback, overall, total_penalty

    # =========================================================
    # fallback
    # =========================================================
    def _fallback_feedback(self, phase):

        if phase == "prep":
            return ["引拍資料不足，請確認動作是否完整"]
        elif phase == "impact":
            return ["擊球階段資料不足，無法完整分析"]
        else:
            return ["收拍資料不足，但建議注意動作收尾"]

    # =========================================================
    # impact detection（用 user）
    # =========================================================
    def _find_impact_index(self, feat_usr, path):
        candidates = [(u, feat_usr[u][1]) for _, u in path]
        return max(candidates, key=lambda x: x[1])[0]

    # =========================================================
    # Phase 切分（用 user timeline）
    # =========================================================
    def _split_phases(self, path, feat_std, feat_usr):

        impact_u = self._find_impact_index(feat_usr, path)

        prep, impact, follow = [], [], []

        for s, u in path:

            if u < impact_u - 5:
                prep.append((s, u))

            elif abs(u - impact_u) <= 5:
                impact.append((s, u))

            else:
                follow.append((s, u))

        if len(follow) == 0 and len(impact) > 0:
            follow.append(impact[-1])

        return {
            "prep": prep,
            "impact": impact,
            "follow": follow
        }

    # =========================================================
    # 誤差計算
    # =========================================================
    def _compute_phase_errors(self, feat_std, feat_usr, path, segments):

        phase_errors = {}

        for phase, pairs in segments.items():

            joint_accumulator = defaultdict(list)

            for s, u in pairs:

                diff = feat_std[s] - feat_usr[u]

                joint_accumulator["16"].append(abs(diff[1]))   # wrist
                joint_accumulator["14"].append(abs(diff[0]))   # elbow
                joint_accumulator["12"].append(abs(diff[2]))   # shoulder

            if len(joint_accumulator) == 0:
                phase_errors[phase] = {}
                continue

            phase_errors[phase] = {
                j: float(np.mean(v)) for j, v in joint_accumulator.items()
            }

        return phase_errors

    # =========================================================
    # 教練語句
    # =========================================================
    def _generate_phase_feedback(self, phase, errors):

        factor = self.relaxed_factor[phase]

        joint_levels = {}
        for j, raw_err in errors.items():
            w = self.joint_weights.get(j, 1.0)
            adjusted = (raw_err * w) / factor

            if adjusted < self.good_threshold:
                joint_levels[j] = "good"
            elif adjusted < self.bad_threshold:
                joint_levels[j] = "medium"
            else:
                joint_levels[j] = "bad"

        if any(v == "bad" for v in joint_levels.values()):
            overall_level = "bad"
        elif any(v == "medium" for v in joint_levels.values()):
            overall_level = "medium"
        else:
            overall_level = "good"

        messages = []

        for j, level in joint_levels.items():
            if level == "good":
                continue
            msg = self._joint_message(phase, j, level)
            if msg:
                messages.append(msg)

        if not messages:
            messages.append(self._phase_good_msg(phase))

        return messages, overall_level

    # =========================================================
    # 單一關節訊息（正拍挑球版）
    # =========================================================
    def _joint_message(self, phase, joint, level):

        # ---------- prep ----------
        if phase == "prep":
            if joint == "16":  # wrist
                if level == "bad":
                    return "準備時手腕過早翻起，建議放鬆手腕，等拍面接近來球再打開迎球"
                else:
                    return "準備時手腕稍微提前動作，可再放鬆延後一點"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "準備時手肘過度彎曲或抬得過高，建議自然前伸，讓拍面順勢對準來球方向"
                else:
                    return "準備手肘角度略嫌僵硬，放鬆前伸會更順手"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "準備時肩膀轉動過大，挑球不需要大幅轉體，建議以手臂前伸迎球為主"
                else:
                    return "準備肩部轉動稍多，挑球動作可以更小、更集中在手臂"

        # ---------- impact ----------
        elif phase == "impact":
            if joint == "16":  # wrist
                if level == "bad":
                    return "擊球瞬間手腕托球動作不足，建議在觸球時用手腕向上帶一下把球送高"
                else:
                    return "擊球時手腕托送力道稍弱，可再多一點向上帶球的動作"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "擊球時手肘過度伸展出力，挑球是控制性擊球，建議減少手肘用力改用手腕托球"
                else:
                    return "擊球時手肘出力稍多，可以再放鬆一點讓手腕主導"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "擊球時肩膀過度參與發力，挑球應以手腕手感為主，肩膀盡量保持穩定"
                else:
                    return "擊球時肩部稍微用力過多，嘗試讓手腕來完成托球動作"

        # ---------- follow ----------
        else:
            if joint == "16":  # wrist
                if level == "bad":
                    return "收拍時手腕動作過大，挑球後應迅速回穩準備下一拍，不需要大幅延伸"
                else:
                    return "收拍手腕稍微多餘動作，試著擊球後就自然停下"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "收拍時手肘延伸過多，建議擊球後立即微收，保持準備下一拍的姿勢"
                else:
                    return "收拍手肘延伸稍多，可以再快一點收回準備位置"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "收拍時肩膀持續轉動過多，挑球後應盡快回正準備下一次來球"
                else:
                    return "收拍肩部跟轉略多，回正速度可以再快一點"

        return None

    # =========================================================
    # 整體階段稱讚（全 good 時使用）
    # =========================================================
    def _phase_good_msg(self, phase):
        if phase == "prep":
            return "準備動作簡潔到位，拍面迎球角度良好"
        elif phase == "impact":
            return "擊球時手腕托球控制精準，出球穩定"
        else:
            return "收拍動作簡潔俐落，已準備好接下一拍"

    # =========================================================
    # 整體評價
    # =========================================================
    def _overall_assessment(self, final_score):

        try:
            score = float(final_score)
        except:
            return "無法計算整體評分"

        if score > 85:
            return "整體動作優秀，已接近標準選手水準"
        elif score > 70:
            return "整體動作良好，但仍有細節可優化"
        elif score > 50:
            return "動作有明顯誤差，建議針對關鍵動作調整"
        else:
            return "動作差異較大，建議從基礎重新建立動作"