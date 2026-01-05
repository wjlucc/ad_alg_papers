#!/usr/bin/env python3
"""
论文批量下载工具
支持从 arXiv 和 Semantic Scholar 搜索并下载论文PDF

使用方法:
    python paper_downloader.py                    # 交互式单篇搜索下载
    python paper_downloader.py --batch papers.txt # 批量下载（从文件读取标题列表）
    python paper_downloader.py --from-readme      # 从readme.md解析论文标题并下载
"""

import os
import re
import html
import json
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional
import ssl

# 创建不验证SSL的上下文（某些环境可能需要）
SSL_CONTEXT = ssl.create_default_context()
SSL_CONTEXT.check_hostname = False
SSL_CONTEXT.verify_mode = ssl.CERT_NONE

# 目录映射：板块编号 -> 目录名
CATEGORY_MAP = {
    "1": "1_竞价策略",
    "2": "2_拍卖机制设计", 
    "3": "3_LLM与经济代理",
    "4": "4_博弈论基础",
    "5": "5_基准与综述",
}

# 默认输出目录
DEFAULT_OUTPUT_DIR = "./Ad_Bidding_Auction_Mechanisms"


@dataclass
class Paper:
    """论文信息结构"""
    title: str
    authors: list[str]
    year: Optional[int]
    pdf_url: Optional[str]
    arxiv_id: Optional[str]
    source: str  # 'arxiv' or 'semantic_scholar'
    abstract: Optional[str] = None


class SemanticScholarAPI:
    """Semantic Scholar API 封装"""
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {"Accept": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key
    
    def search(self, query: str, limit: int = 5) -> list[Paper]:
        """搜索论文"""
        params = urllib.parse.urlencode({
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,openAccessPdf,externalIds,abstract"
        })
        url = f"{self.BASE_URL}/paper/search?{params}"

        last_error: Optional[Exception] = None
        for attempt in range(1, 5):
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
                    data = json.loads(response.read().decode('utf-8'))
                last_error = None
                break
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429 and attempt < 5:
                    retry_after = e.headers.get("Retry-After") if getattr(e, "headers", None) else None
                    wait_s = int(retry_after) if retry_after and str(retry_after).isdigit() else min(2**attempt, 30)
                    print(f"⚠️  Semantic Scholar 触发限流(429)，等待 {wait_s}s 后重试... ({attempt}/5)")
                    time.sleep(wait_s)
                    continue
                print(f"⚠️  Semantic Scholar API 错误: {e.code}")
                return []
            except Exception as e:
                last_error = e
                break

        if last_error is not None:
            print(f"⚠️  请求失败: {last_error}")
            return []
        
        papers = []
        for item in data.get("data", []):
            arxiv_id = None
            if item.get("externalIds"):
                arxiv_id = item["externalIds"].get("ArXiv")
            
            pdf_url = None
            if item.get("openAccessPdf"):
                pdf_url = item["openAccessPdf"].get("url")
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            papers.append(Paper(
                title=item.get("title", ""),
                authors=[a.get("name", "") for a in item.get("authors", [])],
                year=item.get("year"),
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                source="semantic_scholar",
                abstract=item.get("abstract")
            ))
        
        return papers


class ArxivAPI:
    """arXiv API 封装"""
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        """搜索论文"""
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance"
        })
        url = f"{self.BASE_URL}?{params}"
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8')
        except Exception as e:
            print(f"⚠️  arXiv API 请求失败: {e}")
            return []
        
        # 简单的XML解析（避免依赖额外库）
        papers = []
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        
        for entry in entries:
            # 提取标题
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_match.group(1).strip().replace('\n', ' ') if title_match else ""
            
            # 提取作者
            authors = re.findall(r'<name>(.*?)</name>', entry)
            
            # 提取ID和PDF链接
            id_match = re.search(r'<id>http://arxiv.org/abs/(.*?)</id>', entry)
            arxiv_id = id_match.group(1) if id_match else None
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
            
            # 提取年份
            published_match = re.search(r'<published>(\d{4})', entry)
            year = int(published_match.group(1)) if published_match else None
            
            # 提取摘要
            abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else None
            
            if title:
                papers.append(Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    pdf_url=pdf_url,
                    arxiv_id=arxiv_id,
                    source="arxiv",
                    abstract=abstract
                ))
        
        return papers


class PaperDownloader:
    """论文下载器"""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.arxiv_api = ArxivAPI()
        self.semantic_api = SemanticScholarAPI()
        self.downloaded = []
        self.failed = []
    
    def sanitize_filename(self, name: str) -> str:
        """清理文件名，移除非法字符"""
        name = html.unescape(name)
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = re.sub(r'\s+', '_', name)
        if len(name) > 150:
            name = name[:150]
        return name

    @staticmethod
    def _normalize_title(text: str) -> str:
        text = html.unescape(text or "")
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-z0-9 ]+', ' ', text)
        return ' '.join(text.split())

    @classmethod
    def _title_similarity(cls, a: str, b: str) -> float:
        return SequenceMatcher(None, cls._normalize_title(a), cls._normalize_title(b)).ratio()

    @classmethod
    def _token_coverage(cls, query: str, candidate: str) -> float:
        query_tokens = {t for t in cls._normalize_title(query).split() if len(t) > 2}
        if not query_tokens:
            return 0.0
        candidate_tokens = set(cls._normalize_title(candidate).split())
        return len(query_tokens & candidate_tokens) / len(query_tokens)

    def _find_existing_pdf(self, title: str, subfolder: str = "") -> Optional[Path]:
        target_dir = self.output_dir / subfolder if subfolder else self.output_dir
        if not target_dir.exists():
            return None

        best_path: Optional[Path] = None
        best_score = 0.0
        for pdf_path in target_dir.glob("*.pdf"):
            cand_title = pdf_path.stem.replace("_", " ")
            sim = self._title_similarity(title, cand_title)
            cov = self._token_coverage(title, cand_title)
            score = (sim + cov) / 2
            if score > best_score:
                best_score = score
                best_path = pdf_path

        if best_path and best_score >= 0.7:
            return best_path
        return None
    
    def search_paper(self, title: str) -> Optional[Paper]:
        """搜索论文，优先使用Semantic Scholar，失败则用arXiv"""
        print(f"\n🔍 搜索: {title[:60]}...")
        
        # 先尝试 Semantic Scholar（优先）
        papers = self.semantic_api.search(title, limit=5)

        # 如果没找到或没有可用PDF，再尝试 arXiv（仅作为兜底）
        if not papers or not any(p.pdf_url for p in papers):
            arxiv_papers = self.arxiv_api.search(title, max_results=5)
            papers.extend(arxiv_papers)
        
        if not papers:
            print(f"  ❌ 未找到匹配的论文")
            return None
        
        # 基于“标题相似度 + token覆盖率”选最佳候选，避免下载到无关论文
        with_pdf = [p for p in papers if p.pdf_url]
        candidates = with_pdf if with_pdf else papers
        best = max(
            candidates,
            key=lambda p: (self._title_similarity(title, p.title) + self._token_coverage(title, p.title)) / 2,
        )
        best_sim = self._title_similarity(title, best.title)
        best_cov = self._token_coverage(title, best.title)
        best_score = (best_sim + best_cov) / 2

        if best_score < 0.72 or best_cov < 0.7:
            print(
                f"  ❌ 候选标题匹配度过低(sim={best_sim:.2f}, cov={best_cov:.2f})，跳过: {best.title}"
            )
            return None

        if not best.pdf_url:
            print(f"  ⚠️  找到论文但无开放PDF: {best.title}")
            return best

        return best
    
    def download_pdf(self, paper: Paper, subfolder: str = "") -> bool:
        """下载论文PDF"""
        if not paper.pdf_url:
            print(f"  ⚠️  无可用PDF链接")
            return False
        
        # 准备保存路径
        target_dir = self.output_dir / subfolder if subfolder else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = self.sanitize_filename(paper.title) + ".pdf"
        filepath = target_dir / filename
        
        if filepath.exists():
            print(f"  ✅ 已存在: {filename}")
            return True
        
        print(f"  ⬇️  下载中: {paper.pdf_url}")
        
        try:
            req = urllib.request.Request(
                paper.pdf_url,
                headers={"User-Agent": "Mozilla/5.0 (Academic Paper Downloader)"}
            )
            with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
                content = response.read()
            
            # 验证是否为PDF
            if not content.startswith(b'%PDF'):
                print(f"  ❌ 下载内容不是有效的PDF")
                return False
            
            with open(filepath, 'wb') as f:
                f.write(content)
            
            size_mb = len(content) / (1024 * 1024)
            print(f"  ✅ 下载成功: {filename} ({size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return False
    
    def process_single(self, title: str, subfolder: str = "") -> bool:
        """处理单篇论文"""
        existing = self._find_existing_pdf(title, subfolder=subfolder)
        if existing:
            print(f"\n✅ 已存在(匹配): {existing.name}")
            self.downloaded.append(title)
            return True

        paper = self.search_paper(title)
        if paper:
            success = self.download_pdf(paper, subfolder)
            if success:
                self.downloaded.append(title)
            else:
                self.failed.append((title, "下载失败"))
            return success
        else:
            self.failed.append((title, "未找到"))
            return False
    
    def process_batch(self, titles: list[str], delay: float = 1.0):
        """批量处理"""
        print(f"\n📚 开始批量下载 {len(titles)} 篇论文...")
        
        for i, title in enumerate(titles, 1):
            print(f"\n[{i}/{len(titles)}]")
            self.process_single(title)
            time.sleep(delay)
        
        self.print_summary()
    
    def print_summary(self):
        """打印下载摘要"""
        print("\n" + "=" * 60)
        print("📊 下载摘要")
        print("=" * 60)
        print(f"✅ 成功下载: {len(self.downloaded)} 篇")
        print(f"❌ 下载失败: {len(self.failed)} 篇")
        
        if self.failed:
            print("\n失败列表:")
            for title, reason in self.failed:
                print(f"  - {title[:50]}... ({reason})")


def parse_readme_for_papers(readme_path: str) -> list[tuple[str, str]]:
    """
    从readme.md解析论文标题和对应的分类
    返回: [(论文标题, 目录名), ...]
    """
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    def extract_title(markdown_line: str) -> Optional[str]:
        # 去掉开头的 "- "
        item = markdown_line[2:].strip()
        # 去掉形如 [待下载] 的标记
        item = re.sub(r'\s*\[[^\]]+\]\s*', ' ', item).strip()
        # 仅保留标题部分（去掉后面的描述）
        item = item.split(' - ')[0].strip()
        # 去掉末尾包含年份的括号，例如 "(2019)"、"(Meta, 2024)"、"(Akbarpour & Li, 2020)"
        item = re.sub(r'\s*\([^)]*\d{4}[^)]*\)\s*$', '', item).strip()
        if len(item) <= 5:
            return None
        return item

    papers: list[tuple[str, str]] = []
    current_section = ""
    
    lines = content.split('\n')
    for line in lines:
        # 检测一级标题 (## 1. 竞价策略)
        section_match = re.match(r'^## (\d+)\.\s+', line)
        if section_match:
            section_num = section_match.group(1)
            current_section = CATEGORY_MAP.get(section_num, "")
            continue
        
        # 解析论文行: - 论文标题 (年份) - 描述
        # 允许包含 [待下载] 的条目（用于重试下载/补全）
        if line.startswith('- ') and current_section:
            title = extract_title(line)
            if title:
                papers.append((title, current_section))
    
    return papers


def interactive_mode(downloader: PaperDownloader):
    """交互式模式"""
    print("\n📖 论文下载工具 - 交互模式")
    print(f"输出目录: {downloader.output_dir}")
    print("输入论文标题进行搜索下载，输入 'q' 退出\n")
    
    # 显示可用分类
    print("可用分类:")
    for num, name in CATEGORY_MAP.items():
        print(f"  {num}: {name}")
    print()
    
    while True:
        title = input("🔎 请输入论文标题: ").strip()
        if title.lower() == 'q':
            break
        if not title:
            continue
        
        category = input("📁 选择分类 (1-5, 回车跳过): ").strip()
        subfolder = CATEGORY_MAP.get(category, "")
        
        downloader.process_single(title, subfolder=subfolder)
    
    if downloader.downloaded:
        downloader.print_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="论文批量下载工具")
    parser.add_argument("--batch", type=str, help="包含论文标题的文件（每行一个标题）")
    parser.add_argument("--from-readme", action="store_true", help="从readme.md解析论文标题")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--delay", type=float, default=1.5, help="请求间隔（秒）")
    parser.add_argument("-y", "--yes", action="store_true", help="自动确认下载，跳过确认提示")
    
    args = parser.parse_args()
    
    downloader = PaperDownloader(output_dir=args.output)
    
    if args.from_readme:
        readme_path = Path(__file__).parent / "readme.md"
        if not readme_path.exists():
            print(f"❌ 未找到 {readme_path}")
            return
        
        papers = parse_readme_for_papers(str(readme_path))
        print(f"\n📋 从 readme.md 解析出 {len(papers)} 篇论文")
        
        for title, category in papers:
            print(f"  - [{category}] {title[:50]}...")
        
        confirm = 'y' if args.yes else input("\n确认下载这些论文? (y/n): ").strip().lower()
        if confirm == 'y':
            for title, category in papers:
                downloader.process_single(title, subfolder=category)
                time.sleep(args.delay)
            downloader.print_summary()
    
    elif args.batch:
        with open(args.batch, 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f if line.strip()]
        downloader.process_batch(titles, delay=args.delay)
    
    else:
        interactive_mode(downloader)


if __name__ == "__main__":
    main()
