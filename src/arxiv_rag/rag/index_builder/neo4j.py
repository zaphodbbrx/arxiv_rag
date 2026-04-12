
from __future__ import annotations

import logging
import os

from neo4j import GraphDatabase
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from openai import APIConnectionError, AuthenticationError

from src.arxiv_rag.data_ingestion.models import ArxivPaper, SectionType
from src.arxiv_rag.rag.index_builder.base import BaseIndexBuilder
from src.arxiv_rag.utils.setup import setup_openai_api

logger = logging.getLogger(__name__)



class Neo4jIndexBuilder(BaseIndexBuilder):
    def __init__(self, url: str | None = None, username: str | None = None, password: str | None = None):
        def _check_input(variable: str | None, env_name: str) -> bool:
            if not variable:
                logger.error("'%s' does not set.", env_name)
                return False
            return True
        
        super().__init__()

        neo4j_url = os.environ.get("NEO4J_URI", url)
        neo4j_user = os.environ.get("NEO4J_USERNAME", username)
        neo4j_pass = os.environ.get("NEO4J_PASSWORD", password)

        if not _check_input(neo4j_url, "NEO4J_URI"):
            return

        try:
        
            self._driver = GraphDatabase.driver(neo4j_url, #auth=(neo4j_user, neo4j_pass)
                                            )
            self._client = setup_openai_api()
        except ValueError as e:
            logger.exception(f"Неверный формат URL Neo4j")
        except ServiceUnavailable as e:
            logger.exception(f"Neo4j сервер недоступен")
        except ConnectionRefusedError as e:
            # Не удалось установить соединение
            logger.exception(f"Соединение с Neo4j отклонено")
        except TimeoutError as e:
            logger.exception(f"Таймаут подключения к Neo4j")
        except APIConnectionError as e:
            logger.exception(f"Ошибка подключения к OpenAI API")
        except AuthenticationError as e:
            logger.exception(f"Ошибка аутентификации OpenAI")
        except KeyError as e:
            logger.exception(f"Отсутствует необходимая переменная окружения")


    def build_index(self, papers: list[ArxivPaper], **kwargs):
        with self._driver.session() as session:
            session.execute_write(self.insert_papers, papers)

    def load_index(self, **kwargs):
        if not self._driver or not self._client:
            raise RuntimeError("Neo4j Index was not initialized correctly."
                              "Please, set 'NEO4J_URI', 'OPENAI_API_KEY', ''OPENAI_API_BASE to continue")

    def extract_keywords(self, text, top_k=25):
        prompt = f"Extract {top_k} important keywords from this text:\n\n{text}"
        resp = self._client.chat.completions.create(
            model="Qwen3-Coder-30B-A3B-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.split(",")


    def insert_papers(self, tx, papers: list[ArxivPaper]):
        for paper in papers:
            if not paper.is_parsed or not paper.sections:
                continue

            for section in paper.sections:
                stype = (
                    section.section_type.value
                    if isinstance(section.section_type, SectionType)
                    else str(section.section_type)
                )

                # Skip non-content sections
                if stype == "skip":
                    skipped += 1
                    continue

                text = section.content.strip()
                if not text or len(text) < 20:
                    continue

                # insert news node
                tx.run("""
                    MERGE (a:Articles {arxiv_id: $arxiv_id, title: $title, authors: $authors, first_author: $first_author, published: $published, year: $year, url: $url, section: $section, section_type: $section_type, page_numbers: $page_numbers
                    })
                    SET a.text = $text
                    """, 
                    text=text,
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    authors=", ".join(paper.authors),
                    first_author=paper.authors[0].split()[-1] if paper.authors else "Unknown",
                    published=paper.published,
                    year=paper.published[:4] if paper.published and len(paper.published) >= 4 else None,
                    url=paper.entry_url or f"https://arxiv.org/abs/{paper.arxiv_id}",
                    section=section.heading,
                    section_type=stype,
                    page_numbers=section.page_numbers,
                )

            # insert keyword nodes and edges (<=5)
            for key_word in self.extract_keywords(text, top_k=25):
                tx.run("""
                    MERGE (k:Keyword {name: $key_word})
                    WITH k
                    MATCH (a:Articles {arxiv_id: $arxiv_id, title: $title})
                    MERGE (n)-[:MENTIONS]->(k)
                    """, 
                    key_word=key_word.strip().lower(), 
                    title=paper.title,
                    arxiv_id=paper.arxiv_id
                )
