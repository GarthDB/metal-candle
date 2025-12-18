# metal-candle Visual Roadmap

**Last Updated**: December 18, 2024

---

## 📊 Release Timeline

```
2024 Q4              2025 Q1                    2025 Q2                    2025 Q3
   │                    │                          │                          │
   ├─ v1.3.0 ✅        ├─ v1.3.1                  ├─ v1.7.0                  ├─ v2.0.0
   │  Dec 18           │  Late Jan                │  Late May                │  Jul-Sep
   │  • Streaming      │  • ApplyAdapter          │  • Flash Attention       │  • Multi-GPU
   │  • Adapters       │  • Benchmarks            │  • 32k context           │  • 70B+ models
   │                   │                          │                          │
   │                   ├─ v1.4.0                  │                          │
   │                   │  Late Feb                │                          │
   │                   │  • GGUF support          │                          │
   │                   │  • Quantized inference   │                          │
   │                   │                          │                          │
   │                   ├─ v1.5.0                  │                          │
   │                   │  Late Mar                │                          │
   │                   │  • LLaMA/Mistral         │                          │
   │                   │  • Multi-arch            │                          │
   │                   │                          │                          │
   │                   └─ v1.6.0                  │                          │
   │                      Late Apr                │                          │
   │                      • Quantization          │                          │
   │                      • GPTQ/AWQ              │                          │
   │                                              │                          │
```

---

## 🎯 Feature Matrix

| Feature | v1.3.0 | v1.3.1 | v1.4.0 | v1.5.0 | v1.6.0 | v1.7.0 | v2.0.0 |
|---------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| **Streaming Inference** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Adapter Registry** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Hot-Swap Adapters** | 🔸 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming Benchmarks** | ⏱️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **GGUF Loading** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Quantized Inference** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **LLaMA Architecture** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Mistral Architecture** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **In-Memory Quantization** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **GPTQ/AWQ** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Flash Attention** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **32k+ Context** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Multi-GPU** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

**Legend**: ✅ Complete | 🔸 Partial | ⏱️ Expected | ❌ Not Available

---

## 🏗️ Architecture Evolution

### Current (v1.3.0)
```
┌─────────────────────────────────────┐
│         metal-candle v1.3.0         │
├─────────────────────────────────────┤
│ • Streaming Inference (sync/async)  │
│ • LoRA Adapter Registry             │
│ • Qwen + BERT Models                │
│ • Safetensors Format                │
│ • Single GPU (Metal)                │
│ • fp16 Only                         │
└─────────────────────────────────────┘
```

### v1.4.0 (February 2025)
```
┌─────────────────────────────────────┐
│         metal-candle v1.4.0         │
├─────────────────────────────────────┤
│ • Streaming Inference ✅            │
│ • Hot-Swap Adapters ✅              │
│ • Qwen + BERT Models                │
│ • Safetensors + GGUF ⭐             │
│ • Single GPU (Metal)                │
│ • fp16 + Quantized (4/8-bit) ⭐     │
└─────────────────────────────────────┘
```

### v1.5.0 (March 2025)
```
┌─────────────────────────────────────┐
│         metal-candle v1.5.0         │
├─────────────────────────────────────┤
│ • Streaming Inference ✅            │
│ • Hot-Swap Adapters ✅              │
│ • Qwen + BERT + LLaMA + Mistral ⭐  │
│ • Safetensors + GGUF ✅             │
│ • Single GPU (Metal)                │
│ • fp16 + Quantized (4/8-bit) ✅     │
└─────────────────────────────────────┘
```

### v1.7.0 (May 2025)
```
┌─────────────────────────────────────┐
│         metal-candle v1.7.0         │
├─────────────────────────────────────┤
│ • Streaming Inference ✅            │
│ • Hot-Swap Adapters ✅              │
│ • Multi-Architecture ✅             │
│ • Multiple Formats ✅               │
│ • Flash Attention ⭐                │
│ • 32k+ Context ⭐                   │
│ • Single GPU (Metal)                │
│ • fp16 + Quantized ✅               │
└─────────────────────────────────────┘
```

### v2.0.0 (Q3 2025)
```
┌─────────────────────────────────────┐
│         metal-candle v2.0.0         │
├─────────────────────────────────────┤
│ • Streaming Inference ✅            │
│ • Hot-Swap Adapters ✅              │
│ • Multi-Architecture ✅             │
│ • Multiple Formats ✅               │
│ • Flash Attention ✅                │
│ • 32k+ Context ✅                   │
│ • Multi-GPU (2-4 GPUs) ⭐           │
│ • 70B+ Model Support ⭐             │
│ • fp16 + Quantized ✅               │
│ • Production Deployment ⭐          │
└─────────────────────────────────────┘
```

---

## 📈 Performance Evolution

### Inference Speed (Tokens/Second)

```
500 tok/s ┤
          │                                           ╭─ v2.0.0 (Multi-GPU)
400 tok/s ┤                       ╭───────────────────╯
          │                   ╭───╯ v1.7.0 (Flash Attention)
300 tok/s ┤               ╭───╯
          │           ╭───╯ v1.4.0 (Quantized)
200 tok/s ┤       ╭───╯
          │   ╭───╯ v1.3.0 (Streaming)
100 tok/s ┤───╯
          │
          └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───>
          v1.3.0 v1.3.1 v1.4.0 v1.5.0 v1.6.0 v1.7.0 v2.0.0
```

### Memory Efficiency (GB for 7B Model)

```
14 GB ┤───╮
      │   │ v1.3.0 (fp16)
12 GB ┤   │
      │   │
10 GB ┤   ╰───╮
      │       │ v1.4.0 (int8)
 8 GB ┤       │
      │       ╰───────╮
 6 GB ┤               │
      │               ╰───────────────────╮ v1.4.0 (int4)
 4 GB ┤                                   │
      │                                   ╰─────────────────────>
      └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───>
      v1.3.0     v1.4.0         v1.5.0         v1.6.0         v2.0.0
```

---

## 🎯 Development Focus by Quarter

### Q4 2024 (✅ Complete)
- **Foundation**: Core features and quality
- **v1.0-v1.2**: Training, inference, embeddings
- **v1.3.0**: Streaming and adapter management

### Q1 2025 (🚧 In Progress)
- **Ecosystem**: Format compatibility and architecture support
- **v1.3.1**: Hot-swapping implementation
- **v1.4.0**: GGUF and quantization
- **v1.5.0**: Multi-architecture support

### Q2 2025 (📋 Planned)
- **Performance**: Advanced optimizations
- **v1.6.0**: Advanced quantization methods
- **v1.7.0**: Flash Attention

### Q3 2025 (📋 Planned)
- **Scale**: Multi-GPU and production features
- **v2.0.0**: Multi-GPU training and inference

---

## 🔄 Feature Dependency Graph

```
                            ┌──────────────┐
                            │  v1.3.0 Core │
                            │   Features   │
                            └──────┬───────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
              ┌─────────┐    ┌─────────┐    ┌─────────┐
              │ v1.3.1  │    │ v1.4.0  │    │ v1.5.0  │
              │  Adapt  │    │  GGUF   │    │ LLaMA/  │
              │  Swap   │    │         │    │ Mistral │
              └────┬────┘    └────┬────┘    └────┬────┘
                   │              │              │
                   └──────┬───────┴──────────────┘
                          ▼
                    ┌─────────┐
                    │ v1.6.0  │
                    │  Quant  │
                    │ Methods │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    ▼         ▼
              ┌─────────┐ ┌─────────┐
              │ v1.7.0  │ │ v2.0.0  │
              │  Flash  │ │  Multi  │
              │  Attn   │ │   GPU   │
              └─────────┘ └─────────┘
```

---

## 🎓 Complexity & Effort Estimation

| Release | Complexity | Effort (Weeks) | Risk | Community Impact |
|---------|:----------:|:--------------:|:----:|:----------------:|
| v1.3.1  | 🟢 Low | 2-3 | Low | Medium |
| v1.4.0  | 🟡 Medium | 3-4 | Medium | 🔥 High |
| v1.5.0  | 🟡 Medium | 3-4 | Medium | 🔥 High |
| v1.6.0  | 🟠 High | 3-4 | Medium | Medium |
| v1.7.0  | 🔴 Very High | 4-5 | High | 🔥 High |
| v2.0.0  | 🔴 Very High | 8-12 | High | 🔥🔥 Very High |

**Complexity Factors**: API changes, new dependencies, Metal kernel development, testing requirements

---

## 🏆 Priority Matrix

```
High Impact │
           │      v1.4.0 ●         ● v2.0.0
           │      (GGUF)          (Multi-GPU)
           │                  
           │                ● v1.7.0
           │              (Flash Attn)
           │         
           │  v1.3.1 ●           ● v1.5.0
Low Impact │  (Apply)         (LLaMA)
           │              
           │                  ● v1.6.0
           │                  (Quant)
           └────────────────────────────────>
           Low Effort      High Effort
```

**Strategy**: 
- Start with high-impact, low-effort (v1.3.1, v1.4.0)
- Build to high-impact, high-effort (v1.7.0, v2.0.0)
- Schedule medium-impact features around infrastructure needs

---

## 📊 Community Engagement Plan

### Issue Tracking
- Weekly triage of new issues
- Monthly roadmap reviews
- Quarterly feature voting

### Communication
- Release announcements (GitHub, Twitter, Reddit)
- Progress updates in discussions
- Benchmark results published
- Development blogs for major features

### Contribution
- Good first issues labeled
- Detailed contribution guides
- Code review within 1 week
- Regular contributor recognition

---

## 🎯 Success Metrics Dashboard

### Code Quality Targets

```
Tests:       [██████████████████] 195+  → 300+ (v2.0)
Coverage:    [████████████████  ] 80%+  → 85%+ (v2.0)
Clippy:      [██████████████████] 0 warnings (maintained)
Docs:        [██████████████████] 100% (maintained)
```

### Performance Targets (7B Model, M4 Max)

```
Adapter Swap:     [██] 2.5ms        → <2ms (v1.3.1)
GGUF Loading:     [████████████████] TBD (v1.4.0)
Quantized Speed:  [████████████████] TBD (v1.4.0)
Flash Attention:  [████████████████] TBD (v1.7.0)
Multi-GPU Scale:  [████████████████] TBD (v2.0.0)
```

### Community Metrics

```
GitHub Stars:     ⭐ Track growth
Contributors:     👥 Welcome new contributors
Issues Closed:    ✅ <48h response time
Downloads:        📦 Monitor crates.io (post-publish)
```

---

## 🚀 Getting Started as Contributor

### For v1.3.1 (Immediate)
1. Review Issue #49 (ApplyAdapter)
2. Check CONTRIBUTING.md
3. Set up development environment
4. Pick a task from the issue

### For v1.4.0+ (Future)
1. Comment on GitHub issues with interest
2. Research GGUF/quantization/etc.
3. Propose implementation approach
4. Start with smaller PRs first

---

## 📚 Resources

- **Main Roadmap**: [ROADMAP.md](../ROADMAP.md)
- **Next Steps**: [NEXT_STEPS.md](../NEXT_STEPS.md)
- **Project Board**: https://github.com/users/GarthDB/projects/3
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*This visual roadmap is updated quarterly. Last update: December 18, 2024*

