#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Documents, in a sqlite database."""

import sqlite3
from typing import List
import unicodedata



def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def get_multiple_docs_text(self, doc_ids):
        """Fetch the raw text of the docs in 'doc_ids'."""
        cursor = self.connection.cursor()
        doc_ids = [normalize(doc_id) for doc_id in doc_ids]
        sql = "SELECT text FROM documents WHERE id IN ({0})".format(
            ", ".join("?" for _ in doc_ids))
        cursor.execute(sql, doc_ids)
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results        
    
    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
    
    def add_lines_column(self):
        """Creates a new column 'lines' in the documents table"""
        cursor = self.connection.cursor()
        sql = """
            ALTER TABLE documents
            ADD lines TEXT
        """
        cursor.execute(sql)
        cursor.close()
    
    def store_doc_lines(self, doc_id: str, lines: str):
        """Stores the Wiki article lines in the DB"""
        cursor = self.connection.cursor()
        sql = """
            UPDATE documents
            SET lines = ?
            WHERE id = ?
        """
        cursor.execute(sql, (lines, normalize(doc_id)))
        self.connection.commit()
        cursor.close()

    # TODO: This does not work 
    def store_multiple_doc_lines(self, doc_ids: List[str], lines: List[str]):
        """Stores the Wiki article lines in the DB"""
        doc_ids = [normalize(doc_id) for doc_id in doc_ids]
        doc_id_lines_tuples = list(zip(doc_ids, lines))
        cursor = self.connection.cursor()
        sql = """
            UPDATE documents
            SET lines = ?
            WHERE id = ?
        """
        cursor.executemany(sql, doc_id_lines_tuples)
        self.connection.commit()
        cursor.close()

    def count_docs_with_lines(self):
        cursor = self.connection.cursor()
        sql = "SELECT COUNT(*) FROM documents WHERE lines NOTNULL"
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def create_documents_table(self):
        cursor = self.connection.cursor()
        sql = """
            CREATE TABLE IF NOT EXISTS documents
            (id PRIMARY KEY, text, lines)
        """
        cursor.execute(sql)
        self.connection.commit()
        cursor.close()
        
    def drop_documents_table(self):
        cursor = self.connection.cursor()
        sql = """
            DROP TABLE IF EXISTS documents
        """
        cursor.execute(sql)
        self.connection.commit()
        print("'documents' table dropped")
        cursor.close()
        
    
    def insert_docs_with_lines(self, ids, texts, lines):
        """Stores Wiki articles with lines in the DB"""
        ids = [normalize(id) for id in ids]
        doc_tuples = list(zip(ids, texts, lines))
        cursor = self.connection.cursor()
        sql = """
            INSERT OR IGNORE INTO documents
            VALUES (?, ?, ?)
        """
        cursor.executemany(sql, doc_tuples)
        self.connection.commit()
        cursor.close()
        
    def insert_doc_with_lines(self, id, text, lines):
        """Stores the Wiki article with lines in the DB"""
        cursor = self.connection.cursor()
        sql = """
            INSERT OR IGNORE INTO documents
            VALUES (?, ?, ?)
        """
        cursor.execute(sql, (normalize(id), text, lines))
        self.connection.commit()
        cursor.close()
        