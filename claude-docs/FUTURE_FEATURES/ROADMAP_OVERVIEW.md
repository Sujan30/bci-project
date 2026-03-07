# Sleep-BCI: Resume-Grade Roadmap

**Goal**: Transform from "solid prototype" → "production-ready showcase"

**Phases**:
1. **Recruiter-Ready** (48 hours) - First impression wins
2. **BCI-Ready** (1-2 weeks) - Domain credibility
3. **Impressive v1** (1 month) - Senior-level signals

---

## Dependencies & Tools

### Add to pyproject.toml:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pytest-asyncio",
    "httpx",
    "ruff",
    "mypy",
]
streaming = [
    "websockets",
    "redis",
    "pylsl",  # LSL integration
]
analysis = [
    "jupyter",
    "matplotlib",
    "seaborn",
    "pandas",
]
```

---

## Time Budget (Realistic)

| Phase | Tasks | Hours | Cumulative |
|-------|-------|-------|------------|
| Phase 1 | README, CI, tests, diagram | 10h | 10h |
| Phase 2 | Streaming, Redis, logging, config | 20h | 30h |
| Phase 3 | LSL, dashboard, metrics, notebooks | 38h | 68h |

**Total**: ~68 hours (~2 weeks full-time, ~1 month part-time)

---

## Success Metrics

### Before:
- Lines of code: ~1500
- Test coverage: <10%
- CI/CD: None
- Documentation: Basic README
- Production features: Docker only

### After:
- Lines of code: ~3500
- Test coverage: >85%
- CI/CD: ✅ (GitHub Actions)
- Documentation: README, API docs, notebooks, diagrams
- Production features: Streaming, Redis, logging, metrics, tests

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope creep | Schedule slip | Stick to phases, cut Phase 3 if needed |
| LSL complexity | Blocks streaming | Use simulation first, LSL optional |
| Test data generation | Blocks testing | Use synthetic data, not real EDF |
| Performance issues | Bad demo | Profile early, optimize hot paths |

---

## Resume Impact Projection

**Before** (current):
> "Built a sleep stage classifier using EEG data"

**After Phase 1**:
> "Engineered production-ready EEG classifier with CI/CD, 85% test coverage, and Docker deployment"

**After Phase 2**:
> "Developed real-time BCI inference API serving 100 req/s with <50ms latency via WebSocket streaming"

**After Phase 3**:
> "Architected end-to-end neurotechnology pipeline processing live EEG streams (LSL) with monitoring dashboard, achieving 75% sleep stage classification accuracy across 20+ subjects"

---

## Next Steps

1. **Review this plan** - Adjust priorities based on your timeline
2. **Start with Phase 1** - High impact, low effort
3. **Commit after each task** - Build a good git history
4. **Use the templates** - README_NEW.md, API_SPEC.md, etc.
5. **Track progress** - Check off items as you go

**Questions? Decision Points:**
- Skip Phase 3 if 2 weeks is max time?
- Focus on streaming OR dashboard (not both)?
- Need help with any specific implementation?
