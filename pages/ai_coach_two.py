import numpy as np
from collections import defaultdict


class AICoach:

    def __init__(self):

        # 關節權重
        self.joint_weights = {
            "16": 3.0,  # wrist
            "14": 2.0,  # elbow
            "12": 1.5,  # shoulder
            "11": 1.2,
            "23": 1.0,
            "24": 1.0
        }

        # threshold
        self.good_threshold = 0.20
        self.bad_threshold = 0.22

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
            "follow": 1.8   # 原本 1.35，調高放寬收拍判定
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

            feedback.append(msgs)  # msgs 現在是 list，每個關節一條

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

        # 保底（避免 follow 空）
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
    # 教練語句（回傳所有關節問題，level 依最嚴重者決定）
    # =========================================================
    def _generate_phase_feedback(self, phase, errors):

        factor = self.relaxed_factor[phase]

        # 對每個關節套用放寬係數後分級
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

        # 整體 level 依最嚴重者
        if any(v == "bad" for v in joint_levels.values()):
            overall_level = "bad"
        elif any(v == "medium" for v in joint_levels.values()):
            overall_level = "medium"
        else:
            overall_level = "good"

        # 逐關節產生訊息（只輸出 medium / bad；good 的關節不多嘴）
        messages = []

        for j, level in joint_levels.items():
            if level == "good":
                continue
            msg = self._joint_message(phase, j, level)
            if msg:
                messages.append(msg)

        # 若全部關節都 good，給整體稱讚
        if not messages:
            messages.append(self._phase_good_msg(phase))

        return messages, overall_level

    # =========================================================
    # 單一關節訊息
    # =========================================================
    def _joint_message(self, phase, joint, level):

        # ---------- prep ----------
        if phase == "prep":
            if joint == "16":  # wrist
                if level == "bad":
                    return "引拍時手腕過早翻轉出力，建議保持放鬆，手腕放鬆延後至擊球前才發力"
                else:
                    return "引拍手腕稍微提前出力，可再放鬆一點延後收緊時機"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "引拍時手肘偏低，建議抬至與肩同高並保持外開蓄力"
                else:
                    return "引拍手肘高度略低，稍微抬高可增加蓄力空間"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "引拍時肩膀提前轉向擊球方向，建議延後轉肩，先以手肘手腕蓄力為主"
                else:
                    return "引拍肩部轉動時機稍早，可再延後一點讓手臂先完成蓄力"

        # ---------- impact ----------
        elif phase == "impact":
            if joint == "16":  # wrist
                if level == "bad":
                    return "擊球瞬間手腕晃動過大，建議在觸球前穩定鎖定手腕再發力"
                else:
                    return "擊球時手腕控制稍不穩定，觸球點前可再收緊一點"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "擊球時手肘彎曲角度過大、未充分伸展，導致力量無法完整傳遞"
                else:
                    return "擊球時手肘伸展稍不完全，可嘗試多打開一點增加穿透力"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "擊球時肩膀提早停轉，建議保持肩部持續帶動手臂過球"
                else:
                    return "擊球肩部帶動稍提早停止，持續轉肩至擊球後效果更佳"

        # ---------- follow ----------
        else:
            if joint == "16":  # wrist
                if level == "bad":
                    return "收拍時手腕未自然釋放向前翻轉，建議讓手腕順慣性甩出"
                else:
                    return "收拍手腕釋放略顯僵硬，放鬆讓它自然帶出即可"
            elif joint == "14":  # elbow
                if level == "bad":
                    return "收拍時手肘過早彎曲回縮，應讓手臂先隨球延伸後再自然收回"
                else:
                    return "收拍手肘回收稍快，試著讓手臂多延伸一點再收"
            elif joint == "12":  # shoulder
                if level == "bad":
                    return "收拍時肩部過早停止轉動，建議持續跟著動作轉至正面"
                else:
                    return "收拍肩部跟轉略嫌不足，讓身體更完整地轉正會更流暢"

        return None

    # =========================================================
    # 整體階段稱讚（全 good 時使用）
    # =========================================================
    def _phase_good_msg(self, phase):
        if phase == "prep":
            return "引拍階段動作穩定，準備姿勢良好"
        elif phase == "impact":
            return "擊球時機與發力控制良好，動作準確"
        else:
            return "收拍動作完整流暢，結尾姿勢到位"

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