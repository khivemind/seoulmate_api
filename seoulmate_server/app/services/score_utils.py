def score_to_grade(score: float) -> int:
    """
    점수 0~100 기준 등급 변환
    1: 매우 좋음/낮음
    5: 매우 나쁨/높음
    """
    if score <= 20:
        return 1
    elif score <= 40:
        return 2
    elif score <= 60:
        return 3
    elif score <= 80:
        return 4
    else:
        return 5


def normalize_score(value: float, min_value: float, max_value: float) -> float:
    """
    임의 값 범위를 0~100으로 정규화
    """
    if max_value == min_value:
        return 0.0

    score = (value - min_value) / (max_value - min_value) * 100
    return round(max(0, min(100, score)), 2)