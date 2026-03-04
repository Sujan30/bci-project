# Action Priorities & Resume Checklist

**Project**: Sleep-BCI | **Date**: 2026-02-15

---

## Resume-Ready Checklist

### Recruiter-Ready (48h) - Must Have:
- [ ] New README with 60s quickstart
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Test coverage badge (>80%)
- [ ] Example output with metrics
- [ ] Architecture diagram
- [ ] Docker quickstart that works end-to-end

### BCI-Ready (1-2 weeks) - Should Have:
- [ ] Streaming inference endpoint (WebSocket)
- [ ] Redis job storage
- [ ] 20+ tests across pipeline
- [ ] Structured logging
- [ ] API documentation site
- [ ] Notebook with analysis

### Impressive v1 (1 month) - Nice to Have:
- [ ] LSL integration example
- [ ] Live visualization dashboard
- [ ] Model comparison (LDA vs XGBoost)
- [ ] Feature importance analysis
- [ ] Prometheus metrics
- [ ] Load testing results

---

## Quantified Impact Statements (for Resume)

**Current Project** (as-is):
> "Built EEG sleep classifier with 53% balanced accuracy"

**After Improvements**:
> "Engineered production-grade BCI pipeline processing 8K+ EEG epochs across distributed inference API, achieving 80% test coverage and <50ms streaming inference latency"

**Key Numbers to Track**:
- Test coverage: ? → 85%
- API latency: ? → <100ms (p95)
- Throughput: ? → 100 req/s
- Pipeline time: ~3 nights → benchmark on full dataset
- Docker image size: ~1.2GB → <800MB (optimization)

---

## Competitor Comparison (What Recruiters See)

| Feature | Your Project | "Impressive" BCI Project |
|---------|--------------|--------------------------|
| End-to-end pipeline | ✅ | ✅ |
| Docker + compose | ✅ | ✅ |
| CI/CD | ❌ | ✅ |
| Tests (>80%) | ❌ | ✅ |
| Streaming inference | ❌ | ✅ |
| Architecture diagram | ❌ | ✅ |
| Metrics visualization | ❌ | ✅ |
| Production features | ⚠️ | ✅ |
| Live device integration | ❌ | ⚠️ |

**Bottom Line**: You have the hard parts done (pipeline, model, API). The gaps are "polish" items that scream "production-ready" to recruiters.

---

## Priority Ranking (Effort vs Impact)

```
High Impact, Low Effort (DO FIRST):
1. New README (2h)
2. GitHub Actions CI (2h)
3. Basic tests (4h)
4. Architecture diagram (1h)

High Impact, Medium Effort:
5. Streaming endpoint (6h)
6. Structured logging (3h)
7. API tests (4h)
8. Redis job storage (5h)

High Impact, High Effort:
9. Full test suite (12h)
10. Dashboard UI (16h)
11. LSL integration (20h)

Low Priority:
- Authentication (unless public-facing)
- Kubernetes (overkill for resume)
- Multiple model comparison (nice but not critical)
```

---

## Final Recommendation

**Action**: Follow the 3-phase PLAN.md roadmap.

**Timeline**:
- **Week 1**: Recruiter-ready (new README, CI, tests, docs)
- **Week 2**: BCI-ready (streaming, logging, persistence)
- **Week 3-4**: Impressive v1 (dashboard, advanced features)

**Expected Outcome**:
- 10x more recruiter attention (visual proof of engineering skills)
- Interview talking points (architecture decisions, trade-offs)
- Demonstrates: Testing, CI/CD, Docker, API design, BCI domain knowledge
- Competitive with senior SWE projects

---

**Next Step**: Review ROADMAP_OVERVIEW.md in FUTURE_FEATURES/ for detailed implementation roadmap.
