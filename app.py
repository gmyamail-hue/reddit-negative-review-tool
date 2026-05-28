import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
import pandas as pd
import streamlit as st
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential


@dataclass
class RuleResult:
    sentiment: str
    aspects: list[str]
    reason: str
    confidence: float


ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
DEFAULT_SUBREDDITS = [
    "technology",
    "gadgets",
    "software",
    "SaaS",
    "productivity",
    "Android",
    "ios",
    "BuyItForLife",
    "Entrepreneur",
    "smallbusiness",
    "reviews",
]
REQUEST_DELAY_SECONDS = 0.4

NEGATIVE_PHRASES: dict[str, float] = {
    "terrible": 2.0,
    "awful": 2.0,
    "horrible": 2.0,
    "worst": 2.0,
    "hate": 1.5,
    "disappointed": 1.5,
    "disappointing": 1.5,
    "not worth": 2.0,
    "waste of money": 2.0,
    "overpriced": 1.5,
    "too expensive": 1.5,
    "refund": 1.5,
    "returned it": 1.5,
    "broken": 1.5,
    "doesn't work": 2.0,
    "do not work": 2.0,
    "didn't work": 2.0,
    "not working": 2.0,
    "constantly crashes": 2.0,
    "keeps crashing": 2.0,
    "buggy": 1.5,
    "full of bugs": 2.0,
    "slow": 1.0,
    "very slow": 1.5,
    "unusable": 2.0,
    "scam": 2.0,
    "misleading": 1.5,
    "false advertising": 2.0,
    "bad experience": 1.5,
    "poor quality": 1.5,
    "cheap quality": 1.5,
    "no support": 1.5,
    "customer support": 0.5,
    "never again": 1.5,
    "avoid": 1.0,
    "don't buy": 2.0,
    "do not buy": 2.0,
    "regret": 1.5,
    "frustrating": 1.5,
    "annoying": 1.0,
    "unreliable": 1.5,
    "glitch": 1.0,
    "issue": 0.5,
    "problem": 0.5,
    "problems": 0.5,
    "issues": 0.5,
    "bad": 1.0,
    "sucks": 2.0,
}

POSITIVE_PHRASES: dict[str, float] = {
    "love it": 1.5,
    "loving it": 1.5,
    "great product": 1.5,
    "highly recommend": 2.0,
    "recommend": 1.0,
    "works great": 1.5,
    "works well": 1.5,
    "worth it": 1.5,
    "good value": 1.0,
    "excellent": 1.5,
    "amazing": 1.5,
    "perfect": 1.5,
    "happy with": 1.0,
    "satisfied": 1.0,
}

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "price": ["expensive", "overpriced", "price", "cost", "subscription", "billing", "waste of money"],
    "performance": ["slow", "lag", "crash", "crashes", "bug", "bugs", "glitch", "freeze"],
    "quality": ["quality", "cheap", "broken", "defect", "poor build"],
    "support": ["support", "customer service", "refund", "no response", "ticket"],
    "ux": ["confusing", "hard to use", "clunky", "ui", "ux", "interface"],
    "reliability": ["unreliable", "outage", "down", "not working", "doesn't work"],
}


def split_keywords(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def resolve_subreddits(subreddit: str, keyword: str) -> list[str]:
    if subreddit.strip().lower() != "all":
        return [subreddit.strip().strip("/")]

    subs = list(DEFAULT_SUBREDDITS)
    if keyword.lower() not in {s.lower() for s in subs}:
        subs.insert(0, keyword)
    return subs


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=1, max=10))
def arctic_get(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    url = f"{ARCTIC_SHIFT_BASE}/{endpoint}?{urlencode(params)}"
    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        payload = resp.json()

    data = payload.get("data", [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        nested = data.get("data", [])
        return nested if isinstance(nested, list) else []
    return []


def normalize_post(post: dict[str, Any], keyword: str) -> dict[str, Any] | None:
    text = f"{post.get('title', '')}\n\n{post.get('selftext', '') or ''}".strip()
    if len(text) < 20:
        return None

    created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
    permalink = post.get("permalink") or ""
    return {
        "source_type": "post",
        "id": post.get("id", ""),
        "keyword": keyword,
        "subreddit": post.get("subreddit", ""),
        "created_utc": created.isoformat(),
        "score": int(post.get("score", 0) or 0),
        "num_comments": int(post.get("num_comments", 0) or 0),
        "author": post.get("author") or "[deleted]",
        "text": text,
        "url": f"https://reddit.com{permalink}" if permalink else "",
    }


def normalize_comment(comment: dict[str, Any], keyword: str) -> dict[str, Any] | None:
    text = (comment.get("body") or "").strip()
    if len(text) < 20:
        return None
    if "[removed]" in text.lower() or "[deleted]" in text.lower():
        return None

    created = datetime.fromtimestamp(comment.get("created_utc", 0), tz=timezone.utc)
    permalink = comment.get("permalink") or ""
    return {
        "source_type": "comment",
        "id": comment.get("id", ""),
        "keyword": keyword,
        "subreddit": comment.get("subreddit", ""),
        "created_utc": created.isoformat(),
        "score": int(comment.get("score", 0) or 0),
        "num_comments": 0,
        "author": comment.get("author") or "[deleted]",
        "text": text,
        "url": f"https://reddit.com{permalink}" if permalink else "",
    }


def fetch_reddit_items(
    keywords: list[str],
    subreddit: str,
    limit_per_keyword: int,
) -> list[dict[str, Any]]:
    cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp())
    collected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    search_jobs = [
        ("posts/search", "title", normalize_post),
        ("posts/search", "selftext", normalize_post),
        ("comments/search", "body", normalize_comment),
    ]

    for kw in keywords:
        keyword_items = 0
        subreddits = resolve_subreddits(subreddit, kw)
        per_query_limit = max(5, min(20, limit_per_keyword))

        for sub in subreddits:
            if keyword_items >= limit_per_keyword:
                break

            for endpoint, field, normalizer in search_jobs:
                if keyword_items >= limit_per_keyword:
                    break

                params = {
                    "subreddit": sub,
                    field: kw,
                    "limit": per_query_limit,
                    "after": cutoff_ts,
                }
                rows = arctic_get(endpoint, params)
                time.sleep(REQUEST_DELAY_SECONDS)

                for row in rows:
                    item = normalizer(row, kw)
                    if not item:
                        continue
                    item_id = item["id"]
                    if not item_id or item_id in seen_ids:
                        continue

                    seen_ids.add(item_id)
                    collected.append(item)
                    keyword_items += 1
                    if keyword_items >= limit_per_keyword:
                        break

    return collected


def find_phrase_matches(text: str, phrases: dict[str, float]) -> list[tuple[str, float]]:
    lowered = text.lower()
    matches: list[tuple[str, float]] = []
    for phrase, weight in phrases.items():
        if phrase in lowered:
            matches.append((phrase, weight))
    return matches


def detect_aspects(text: str) -> list[str]:
    lowered = text.lower()
    aspects: list[str] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            aspects.append(aspect)
    return aspects


def classify_with_rules(text: str) -> RuleResult:
    neg_matches = find_phrase_matches(text, NEGATIVE_PHRASES)
    pos_matches = find_phrase_matches(text, POSITIVE_PHRASES)

    neg_score = sum(weight for _, weight in neg_matches)
    pos_score = sum(weight for _, weight in pos_matches)
    net_score = neg_score - pos_score * 0.7

    aspects = detect_aspects(text)
    matched_phrases = [phrase for phrase, _ in neg_matches[:5]]

    if net_score >= 2.5:
        sentiment = "negative"
        confidence = min(0.95, 0.55 + net_score * 0.08)
    elif net_score >= 1.2:
        sentiment = "negative"
        confidence = min(0.85, 0.45 + net_score * 0.08)
    elif net_score >= 0.6:
        sentiment = "negative"
        confidence = 0.55
    elif pos_score > neg_score + 0.5:
        sentiment = "positive"
        confidence = min(0.9, 0.5 + pos_score * 0.08)
    else:
        sentiment = "neutral"
        confidence = 0.5

    if sentiment == "negative":
        if matched_phrases:
            reason = "Matched complaint phrases: " + ", ".join(matched_phrases)
        else:
            reason = "Rule score indicates likely negative feedback."
    elif sentiment == "positive":
        reason = "Matched positive phrases outweigh complaint signals."
    else:
        reason = "No strong complaint phrases detected."

    return RuleResult(
        sentiment=sentiment,
        aspects=aspects,
        reason=reason,
        confidence=round(confidence, 2),
    )


def format_fetch_error(exc: Exception) -> str:
    if isinstance(exc, RetryError):
        cause = exc.last_attempt.exception()
        if cause is not None:
            return format_fetch_error(cause)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        return f"数据源返回 HTTP {status}，请稍后重试。"
    return str(exc)


def main() -> None:
    st.set_page_config(page_title="Reddit差评查询工具", layout="wide")
    st.title("Reddit 产品差评查询（规则版 · 免费）")
    st.caption("无需 OpenAI，无需 Reddit API Key。通过关键词规则识别可能的差评。")

    with st.sidebar:
        st.header("参数")
        keywords_raw = st.text_area("产品关键词（逗号分隔）", value="Notion, Notion AI")
        subreddit = st.text_input("Subreddit（默认 all）", value="all")
        limit_per_keyword = st.slider("每个关键词最多抓取条数", min_value=5, max_value=30, value=10, step=5)
        min_conf = st.slider("最低置信度", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
        st.info(
            "规则版完全免费，但准确度低于 LLM。"
            "建议先用具体 Subreddit（如 Notion）测试。"
        )

    run_btn = st.button("开始分析", type="primary")

    if not run_btn:
        st.info("配置好参数后，点击“开始分析”。")
        return

    keywords = split_keywords(keywords_raw)
    if not keywords:
        st.warning("请至少输入一个关键词")
        return

    with st.spinner("正在抓取 Reddit 数据..."):
        try:
            items = fetch_reddit_items(keywords, subreddit, limit_per_keyword)
        except Exception as exc:
            st.error(f"抓取 Reddit 失败：{format_fetch_error(exc)}")
            return

    if not items:
        st.warning(
            "未抓到数据。可以尝试："
            "1) 把 Subreddit 改成产品名社区（如 Notion）；"
            "2) 换更常见的英文关键词；"
            "3) 提高抓取条数。"
        )
        return

    results: list[dict[str, Any]] = []
    for row in items:
        rule = classify_with_rules(row["text"])
        out = dict(row)
        out["sentiment"] = rule.sentiment
        out["aspects"] = ", ".join(rule.aspects)
        out["reason"] = rule.reason
        out["confidence"] = rule.confidence
        results.append(out)

    df = pd.DataFrame(results)
    neg_df = df[(df["sentiment"] == "negative") & (df["confidence"] >= min_conf)].copy()
    neg_df.sort_values(by=["score", "created_utc"], ascending=[False, False], inplace=True)

    st.subheader("分析结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("抓取总条数", len(df))
    c2.metric("负面条数", len(neg_df))
    c3.metric("负面占比", f"{(len(neg_df) / max(len(df), 1)) * 100:.1f}%")

    if len(neg_df) == 0:
        st.info("当前条件下没有识别到符合置信度的差评。可以尝试降低“最低置信度”。")
        return

    st.write("### 负面原因分布（Top 10）")
    aspect_series = (
        neg_df["aspects"]
        .fillna("")
        .str.split(",")
        .explode()
        .str.strip()
    )
    aspect_series = aspect_series[aspect_series != ""]
    if len(aspect_series) > 0:
        st.bar_chart(aspect_series.value_counts().head(10))
    else:
        st.write("暂无可用标签")

    st.write("### 负面明细")
    st.dataframe(
        neg_df[
            [
                "source_type",
                "keyword",
                "subreddit",
                "created_utc",
                "score",
                "author",
                "aspects",
                "reason",
                "url",
                "text",
            ]
        ],
        use_container_width=True,
        height=520,
    )

    csv_data = neg_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="下载差评 CSV",
        data=csv_data,
        file_name="reddit_negative_reviews.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
