# 当前定制分支启动说明

这份文档是此定制分支的 Quick Start。

## 1. 拉取并安装

```bash
git clone https://github.com/zhangyuqi-1/AutoResearchClaw.git
cd AutoResearchClaw
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
pip install "httpx[socks]" torch scikit-learn pandas seaborn tqdm google-genai Pillow
```

## 2. 初始化环境

```bash
# `researchclaw setup` 会处理 OpenCode 安装，并检查 Docker / LaTeX
researchclaw setup
# 但当前真实运行链路还依赖 `acpx`
npm install -g acpx
# Stage 24 的 `docx` 导出依赖 `pandoc`
sudo apt update && sudo apt install -y \
  pandoc \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-fonts-extra \
  texlive-xetex \
  texlive-lang-chinese \
  latexmk \
  wkhtmltopdf \
  libreoffice
```

## 3. 启动运行

```bash
# 先准备配置：参考 `config2.arc.yaml`

# 启动运行 ，主 LLM 走 `ACP + codex` ，代码生成 / repair 走 `OpenCode`
set -a && source .env && set +a && researchclaw run --config config2.arc.yaml --auto-approve
# 从 Stage 22 重跑命令：
set -a && source .env && set +a && researchclaw run --config config2.arc.yaml --output artifacts/rc-20260409-184727-b50802 --from-stage EXPORT_PUBLISH --auto-approve
# 从 Stage 24 重跑命令：
set -a && source .env && set +a && researchclaw run --config config2.arc.yaml --output artifacts/rc-20260409-184727-b50802 --from-stage FINAL_EDITORIAL_REPAIR --auto-approve
```
