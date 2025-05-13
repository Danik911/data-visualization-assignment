#!/usr/bin/env python
import sqlite3
import os
import datetime

# Define the database file path
DB_PATH = "data/project_db.db"
FULL_PATH = os.path.join(os.path.dirname(__file__), DB_PATH)

# Create a connection to the database
# This will create the file if it doesn't exist
conn = sqlite3.connect(FULL_PATH)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON")

print(f"Creating database at {FULL_PATH}...")

# Create utility tables
cursor.execute('''
CREATE TABLE status_types (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT
)
''')

# Create thought tracking tables
cursor.execute('''
CREATE TABLE thought_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
)
''')

cursor.execute('''
CREATE TABLE thoughts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    thought_number INTEGER NOT NULL,
    total_thoughts INTEGER NOT NULL,
    thought_content TEXT NOT NULL,
    is_revision INTEGER DEFAULT 0,
    revises_thought_id INTEGER,
    branch_id TEXT,
    branch_from_thought_id INTEGER,
    needs_more_thoughts INTEGER DEFAULT 0,
    next_thought_needed INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (session_id) REFERENCES thought_sessions(id),
    FOREIGN KEY (revises_thought_id) REFERENCES thoughts(id),
    FOREIGN KEY (branch_from_thought_id) REFERENCES thoughts(id)
)
''')

# Create issue tracking tables
cursor.execute('''
CREATE TABLE issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    priority INTEGER DEFAULT 3, -- 1=highest, 5=lowest
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    updated_at TEXT NOT NULL DEFAULT (DATETIME('now'))
)
''')

cursor.execute('''
CREATE TABLE solutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL,
    solution_content TEXT NOT NULL,
    implemented INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (issue_id) REFERENCES issues(id)
)
''')

# Create search tracking tables
cursor.execute('''
CREATE TABLE searches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
)
''')

cursor.execute('''
CREATE TABLE search_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    relevance_score REAL,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (search_id) REFERENCES searches(id)
)
''')

# Create task and feature tracking tables
cursor.execute('''
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    updated_at TEXT NOT NULL DEFAULT (DATETIME('now'))
)
''')

cursor.execute('''
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    sequence_number INTEGER,
    depends_on_task_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
    FOREIGN KEY (feature_id) REFERENCES features(id),
    FOREIGN KEY (depends_on_task_id) REFERENCES tasks(id)
)
''')

# Create indexes
cursor.execute('''CREATE INDEX idx_thoughts_session_id ON thoughts(session_id)''')
cursor.execute('''CREATE INDEX idx_thoughts_branch_id ON thoughts(branch_id)''')
cursor.execute('''CREATE INDEX idx_solutions_issue_id ON solutions(issue_id)''')
cursor.execute('''CREATE INDEX idx_search_results_search_id ON search_results(search_id)''')
cursor.execute('''CREATE INDEX idx_searches_query ON searches(query)''')
cursor.execute('''CREATE INDEX idx_tasks_feature_id ON tasks(feature_id)''')
cursor.execute('''CREATE INDEX idx_tasks_depends_on ON tasks(depends_on_task_id)''')

# Insert sample data
print("Adding sample data...")

# Insert sample data for status types
cursor.execute('''
INSERT INTO status_types (name, description) VALUES 
('new', 'Newly created item'),
('in_progress', 'Work has begun'),
('completed', 'Work is finished'),
('blocked', 'Progress is blocked'),
('deferred', 'Postponed for later')
''')

# Insert sample thought session
cursor.execute('''
INSERT INTO thought_sessions (name, description) 
VALUES ('Database Design Session', 'Thinking through the database design for storing thoughts and tasks')
''')

# Insert sample thoughts
cursor.execute('''
INSERT INTO thoughts (session_id, thought_number, total_thoughts, thought_content, next_thought_needed) 
VALUES (1, 1, 3, 'We need to design a database schema to store sequential thoughts, issues, solutions, search results, and tasks', 1)
''')

cursor.execute('''
INSERT INTO thoughts (session_id, thought_number, total_thoughts, thought_content, next_thought_needed) 
VALUES (1, 2, 3, 'The schema should include tables for thought_sessions, thoughts, issues, solutions, searches, search_results, features, and tasks', 1)
''')

cursor.execute('''
INSERT INTO thoughts (session_id, thought_number, total_thoughts, thought_content, next_thought_needed) 
VALUES (1, 3, 3, 'Finally, we need to create indexes for performance optimization on foreign keys and frequently queried fields', 0)
''')

# Insert sample issue
cursor.execute('''
INSERT INTO issues (title, description, status, priority) 
VALUES ('Database Design', 'Need to design a database for tracking thoughts and tasks', 'completed', 1)
''')

# Insert sample solution
cursor.execute('''
INSERT INTO solutions (issue_id, solution_content, implemented) 
VALUES (1, 'Created SQLite database with tables for thoughts, issues, solutions, searches, and tasks', 1)
''')

# Insert sample search
cursor.execute('''
INSERT INTO searches (query) 
VALUES ('SQLite database design best practices')
''')

# Insert sample search result
cursor.execute('''
INSERT INTO search_results (search_id, content, source) 
VALUES (1, 'SQLite is best for embedded applications and prototypes. Use INTEGER PRIMARY KEY for auto-incrementing IDs', 'database-guide.com')
''')

# Insert sample feature
cursor.execute('''
INSERT INTO features (name, description, status) 
VALUES ('Thought Tracking', 'System for recording sequential thoughts during problem solving', 'in_progress')
''')

# Insert sample tasks
cursor.execute('''
INSERT INTO tasks (feature_id, description, status, sequence_number) 
VALUES (1, 'Design database schema', 'completed', 1)
''')

cursor.execute('''
INSERT INTO tasks (feature_id, description, status, sequence_number, depends_on_task_id) 
VALUES (1, 'Implement database creation script', 'in_progress', 2, 1)
''')

# Commit changes and close connection
conn.commit()
print("Database created successfully!")
conn.close()