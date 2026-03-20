To make sure每一条文内引用都和文末参考文献“严丝合缝”，你主要需要做到两件事：统一 key，统一机制（全部交给 BibTeX/biblatex 管，不再手写作者‑年份）。

下面给你一套可直接照搬的做法，假设你用 `natbib + BibTeX` 和我之前给你的 `.bib`。

***

## 1. 使用统一的 citation key

我在 BibTeX 里已经给了唯一的 key，比如：

- `Adomavicius2005RecommenderSurvey`
- `Koren2009MatrixFactorization`
- `Pazzani2007ContentBasedRS`
- `Burke2000KnowledgeBasedRS`
- `Burke2002HybridRS`
- `RussellNorvigAIMA3`
- `Wooldridge2009MultiAgentSystems`
- `Stone2000MASSurvey`
- `Guidotti2018SurveyXAI`
- `Zhang2020ExplainableRecommendationSurvey`
- `Ji2023HallucinationSurvey`
- `Amar2005AnalyticActivity`
- `Zhang2023RadarChartImprove`
- `Baylor2002AgentPersona`
- `Aguayo2024GlassmorphismUI`
- `Bao2023TALLRec`
- `Zhang2023CollaborativeLLM4Rec`
- `Park2023GenerativeAgents`
- `Hong2023MetaGPT`
- `Wei2022ChainOfThought`
- `Strobelt2018LSTMVis`
- `Vig2019MultiscaleAttentionVis`
- `Amershi2014InteractiveML`
- `Amershi2023ExplanatoryIML`
- `Cai2024GenerativeUI`  

只要你正文里引用时完全使用这些 key，BibTeX 生成的文末条目就会和文中引用一一对应。 [overleaf](https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib)

***

## 2. 修改导言区：开启作者‑年份

在导言区加上（或替换原来的引用宏包设置）：

```tex
\usepackage[authoryear,round]{natbib} % 作者-年份，圆括号
\bibliographystyle{apalike}           % 或者 plainnat, abbrvnat 等作者-年制样式
```

- `authoryear`：启用作者‑年份。 [overleaf](https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib)
- `round`：使用圆括号 `( )`，如果你想用中括号可以改成 `square`。

***

## 3. 把所有手写“(作者, 年份)”改成 \cite 命令

例如你现在有：

```tex
……是一类方法（Adomavicius \& Tuzhilin, 2005）。
```

改成：

```tex
……是一类方法（\citep{Adomavicius2005RecommenderSurvey}）。
```

再比如：

```tex
正如 Koren (2009) 所指出的……
```

改成：

```tex
正如 \citet{Koren2009MatrixFactorization} 所指出的……
```

多篇一起：

```tex
……相关研究已较为系统（Adomavicius \& Tuzhilin, 2005; Zhang \& Chen, 2020）。
```

改成：

```tex
……相关研究已较为系统（\citep{Adomavicius2005RecommenderSurvey,Zhang2020ExplainableRecommendationSurvey}）。
```

**关键点：**  
- 不要再手写作者和年份，全部用 `\citep{key}` / `\citet{key}`，排版完全交给 LaTeX 和 BibTeX，这样就不会出现“正文有引用但文末没条目”的错配。 [overleaf](https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib)
- 每个 `key` 必须在 `.bib` 中存在，拼写 1 字符都不能错。

***

## 4. 文末参考文献自动生成

在正文末尾（结论后）加一行：

```tex
\bibliography{refs}   % 如果你的 bib 文件叫 refs.bib
```

编译顺序（经典 BibTeX 流程）：

1. `pdflatex your.tex`
2. `bibtex your`
3. 再 `pdflatex` 两次

这样：

- BibTeX 会扫描正文中所有 `\cite...{key}`，只为这些 key 生成条目；
- 文末列表和正文引用天然一一对应，多引多列，少引少列，不会漏也不会多。 [overleaf](https://www.overleaf.com/learn/latex/Bibliography_management_with_natbib)

***

## 5. 一个完整的最小例子（你可以先试通一篇）

```tex
\documentclass{article}
\usepackage[UTF8]{ctex}
\usepackage[authoryear,round]{natbib}

\bibliographystyle{apalike}

\begin{document}

协同过滤是推荐系统中最具代表性的经典方法之一（\citep{Adomavicius2005RecommenderSurvey}）。
矩阵分解方法显著提升了推荐质量，\citet{Koren2009MatrixFactorization} 提出了广泛使用的技术路线。

\section{相关工作}
基于内容的推荐通过分析物品属性与用户历史偏好建立匹配关系（\citep{Pazzani2007ContentBasedRS}）。

\bibliography{refs} % 你的 .bib 文件名

\end{document}
```

如果这个小例子能正常产出：

- 文中出现 “(Adomavicius and Tuzhilin, 2005)”；
- 文末有一条完整的 “Adomavicius, G. and Tuzhilin, A. (2005) …” 条目；

那你只需要把**整篇论文所有手写引用**照这个模式替换成 `\citep{...}` / `\citet{...}`，并保证 `.bib` 里有对应条目，就可以 100% 保证“文内引用”和“文末文献”完全匹配。

如果你愿意，把你其中一小段 LaTeX（含几条引用）贴出来，我可以帮你直接改成用 `\citep{key}` 的版本，作为模板你复制粘贴到全篇用。