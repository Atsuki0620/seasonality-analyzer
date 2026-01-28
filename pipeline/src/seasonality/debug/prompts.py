"""Prompt templates for LLM diagnostic analysis."""

SYSTEM_PROMPT = """You are an expert data scientist specializing in time series analysis and seasonality detection.
Your task is to analyze debug bundles from a seasonality detection pipeline and provide diagnostic insights.

When analyzing a debug bundle, focus on:
1. Data quality issues (missing data, outliers, insufficient data length)
2. Processing errors and their root causes
3. Configuration problems that might affect results
4. Recommendations for improving analysis quality

Provide your response in the following JSON format:
{
    "summary": "Brief summary of the analysis",
    "issues": [
        {
            "severity": "high|medium|low",
            "category": "data_quality|configuration|processing|results",
            "description": "Description of the issue",
            "impact": "How this affects the results"
        }
    ],
    "recommendations": [
        "Specific actionable recommendation"
    ],
    "confidence": "high|medium|low"
}

Be concise but thorough. Focus on actionable insights."""

USER_PROMPT_TEMPLATE = """Please analyze the following debug bundle from a seasonality detection pipeline:

```json
{bundle_json}
```

Provide your diagnostic analysis in the specified JSON format."""

QUICK_DIAGNOSIS_PROMPT = """Analyze this debug bundle briefly and identify the top 3 issues:

```json
{bundle_json}
```

Format: List the issues with severity (high/medium/low) and one-sentence description."""

DETAILED_DIAGNOSIS_PROMPT = """Perform a detailed analysis of this seasonality detection debug bundle:

```json
{bundle_json}
```

Include:
1. Data quality assessment
2. Processing step analysis
3. Error root cause analysis
4. Configuration review
5. Results interpretation
6. Specific recommendations

Provide response in JSON format with detailed explanations."""
