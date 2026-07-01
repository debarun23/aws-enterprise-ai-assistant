"""
src/data/scraper.py — AWS documentation scraper.
Fetches pages from AWS docs and extracts clean text content.
"""

import time
import logging
import requests
from pathlib import Path
from typing import Iterator
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# AWS docs URLs — one guide per major service (User Guides are most Q&A-rich)
AWS_DOC_URLS: dict[str, list[str]] = {
    "ec2": [
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instance-types.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSVolumes.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html",
        "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring_ec2.html",
    ],
    "s3": [
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-overview.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lifecycle-mgmt.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/serv-side-encryption.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/storage-class-intro.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/cors.html",
        "https://docs.aws.amazon.com/AmazonS3/latest/userguide/replication.html",
    ],
    "lambda": [
        "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/lambda-security.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/lambda-monitoring.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html",
    ],
    "iam": [
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_groups.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_managed-vs-inline.html",
        "https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_mfa.html",
    ],
    "rds": [
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Welcome.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_Storage.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_CreateSnapshot.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Overview.Encryption.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/multi-az-db-instances-concepts.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ReadRepl.html",
        "https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Overview.BackingUpAndRestoringAmazonRDSInstances.html",
    ],
    "vpc": [
        "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/configure-subnets.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/security-groups.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/vpc-network-acls.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/vpc-peering.html",
        "https://docs.aws.amazon.com/vpc/latest/userguide/internet-gateways.html",
    ],
    "cloudwatch": [
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/WhatIsCloudWatch.html",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/working_with_metrics.html",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Dashboards.html",
        "https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/WhatIsCloudWatchLogs.html",
    ],
    "dynamodb": [
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/HowItWorks.CoreComponents.html",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-partition-key-design.html",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ProvisionedThroughput.html",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GSI.html",
        "https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/backuprestore_HowItWorks.html",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_page(url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """
    Fetch raw HTML from a URL with retry logic.
    Returns None on permanent failure.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    logger.error(f"All retries exhausted for: {url}")
    return None


def parse_aws_page(html: str, url: str) -> dict | None:
    """
    Parse an AWS documentation HTML page.
    Extracts: title, main content text, service name from URL.
    Returns None if no meaningful content found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup.find_all(["nav", "footer", "script", "style",
                               "header", "aside", ".feedback-panel"]):
        tag.decompose()

    # Remove AWS-specific nav divs
    for div in soup.find_all("div", {"id": ["left-column", "right-column",
                                             "breadcrumbs", "page-toc"]}):
        div.decompose()

    # Extract title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    # Extract main content — AWS docs use div#main-content or div.awsdocs-container
    main = (
        soup.find("div", {"id": "main-content"})
        or soup.find("div", {"class": "awsdocs-container"})
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "doc-content"})
        or soup.body
    )

    if not main:
        return None

    # Get clean text
    text = main.get_text(separator=" ", strip=True)

    # Collapse whitespace
    import re
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 200:
        logger.debug(f"Skipping short content ({len(text)} chars): {url}")
        return None

    return {
        "url": url,
        "title": title,
        "text": text,
        "char_count": len(text),
    }


def scrape_all(
    output_dir: str = "data/raw",
    delay_between_requests: float = 1.5,
) -> list[dict]:
    """
    Scrape all AWS doc URLs, save per-service JSONL files.
    Returns list of all scraped documents.
    """
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_docs: list[dict] = []

    for service, urls in AWS_DOC_URLS.items():
        service_docs: list[dict] = []
        logger.info(f"Scraping service: {service.upper()} ({len(urls)} pages)")

        for url in urls:
            html = fetch_page(url)
            if html is None:
                continue

            parsed = parse_aws_page(html, url)
            if parsed is None:
                continue

            parsed["service"] = service
            service_docs.append(parsed)
            logger.info(f"  ✓ {parsed['title'][:60]} ({parsed['char_count']} chars)")

            time.sleep(delay_between_requests)

        # Save per-service file
        service_file = out / f"{service}.jsonl"
        with open(service_file, "w", encoding="utf-8") as f:
            for doc in service_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        logger.info(f"  → Saved {len(service_docs)} docs to {service_file}")
        all_docs.extend(service_docs)

    logger.info(f"Scraping complete. Total docs: {len(all_docs)}")
    return all_docs