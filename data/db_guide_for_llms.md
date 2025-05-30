# SQLite Database Guide for LLMs
**Database Path**: `data/sample.db`

This guide provides a comprehensive overview of the SQLite database structure used in the data visualization assignment project. Follow this guide to efficiently interact with the database without the need to re-explore its structure.

## Database Purpose

This database serves as a knowledge repository for the data analysis workflow, storing:
- Issues encountered during analysis and their solutions
- Thought processes and reasoning chains
- Search queries and results
- Task tracking
- Root causes of problems

## Table Structure and Relationships

### Core Tables

#### 1. `issues`
Stores problems encountered during data analysis.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
title TEXT                    -- Short issue description
description TEXT              -- Detailed issue description
status TEXT                   -- Current status (e.g., 'completed', 'pending')
priority INTEGER              -- Issue priority (lower number = higher priority)
created_at DATETIME           -- Creation timestamp
```

**Example query**: `SELECT * FROM issues WHERE status='completed' ORDER BY priority ASC;`

#### 2. `solutions`
Contains solutions to issues recorded in the `issues` table.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
issue_id INTEGER              -- References issues.id
description TEXT              -- Detailed solution description
implemented INTEGER           -- Boolean (1=implemented, 0=proposed)
created_at DATETIME           -- Creation timestamp
```

**Relationship**: Many solutions can link to one issue (one-to-many)
**Example query**: `SELECT i.title, s.description FROM issues i JOIN solutions s ON i.id = s.issue_id;`

#### 3. `thought_sessions`
Represents thinking sessions or analysis sessions.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
name TEXT                     -- Session name/topic
description TEXT              -- Session purpose
created_at DATETIME           -- Creation timestamp
```

**Example query**: `SELECT * FROM thought_sessions ORDER BY created_at DESC LIMIT 5;`

#### 4. `thoughts`
Individual thoughts or reasoning steps within a thinking session.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
session_id INTEGER            -- References thought_sessions.id
thought_number INTEGER        -- Position in the thought sequence
total_thoughts INTEGER        -- Total expected thoughts in session
thought_content TEXT          -- The actual thought content
is_revision INTEGER           -- Boolean (1=revises previous thought)
revises_thought_id INTEGER    -- References thoughts.id (if revision)
branch_id TEXT                -- Branch identifier for thought branches
branch_from_thought_id INTEGER -- References thoughts.id (branching point)
needs_more_thoughts INTEGER   -- Boolean (1=needs additional thoughts)
next_thought_needed INTEGER   -- Boolean (1=next thought required)
created_at DATETIME           -- Creation timestamp
```

**Relationship**: Many thoughts belong to one session (one-to-many)
**Example query**: `SELECT * FROM thoughts WHERE session_id=1 ORDER BY thought_number;`

#### 5. `searches` and `search_results`
Track searches performed and their results.

```sql
-- searches structure
id INTEGER PRIMARY KEY AUTOINCREMENT
query TEXT                    -- Search query text
source TEXT                   -- Search source (e.g., 'tavily', 'web')
created_at DATETIME           -- Creation timestamp

-- search_results structure
id INTEGER PRIMARY KEY AUTOINCREMENT
search_id INTEGER             -- References searches.id
title TEXT                    -- Result title
content TEXT                  -- Result content
url TEXT                      -- Source URL (if applicable)
relevance_score REAL          -- Relevance score (0.0-1.0)
created_at DATETIME           -- Creation timestamp
```

**Example query**: `SELECT s.query, sr.title, sr.relevance_score FROM searches s JOIN search_results sr ON s.id = sr.search_id ORDER BY sr.relevance_score DESC;`

#### 6. `root_causes`
Documents root causes of issues.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
issue_id INTEGER              -- References issues.id
cause_description TEXT        -- Description of the root cause
severity TEXT                 -- Severity level (e.g., 'high', 'medium', 'low')
created_at DATETIME           -- Creation timestamp
```

**Example query**: `SELECT i.title, rc.cause_description FROM issues i JOIN root_causes rc ON i.id = rc.issue_id;`

#### 7. `tasks`
Tracks tasks to be performed.

```sql
-- Structure
id INTEGER PRIMARY KEY AUTOINCREMENT
title TEXT                    -- Task title
description TEXT              -- Task description
status TEXT                   -- Task status
priority INTEGER              -- Task priority
due_date DATETIME             -- Due date (optional)
created_at DATETIME           -- Creation timestamp
updated_at DATETIME           -- Last update timestamp
```

## Common Operations

### Adding a New Issue
```sql
INSERT INTO issues (title, description, status, priority, created_at) 
VALUES ('Issue Title', 'Detailed issue description', 'pending', 2, datetime('now'));
```

### Adding a Solution to an Issue
```sql
-- First get the issue ID
SELECT id FROM issues WHERE title='Issue Title';

-- Then add the solution
INSERT INTO solutions (issue_id, description, implemented, created_at) 
VALUES (issue_id, 'Detailed solution description', 1, datetime('now'));
```

### Recording a Thought Process
```sql
-- First create a thought session
INSERT INTO thought_sessions (name, description, created_at) 
VALUES ('Session Name', 'Session Description', datetime('now'));

-- Get the session ID
SELECT id FROM thought_sessions ORDER BY id DESC LIMIT 1;

-- Then add thoughts
INSERT INTO thoughts (session_id, thought_number, total_thoughts, thought_content, is_revision, next_thought_needed, created_at) 
VALUES (session_id, 1, 3, 'First thought content', 0, 1, datetime('now'));
```

### Recording a Root Cause
```sql
-- Given an issue ID
INSERT INTO root_causes (issue_id, cause_description, severity, created_at) 
VALUES (issue_id, 'Root cause description', 'medium', datetime('now'));
```

### Finding Related Solutions for an Issue Type
```sql
SELECT i.title as issue_title, s.description as solution_description
FROM issues i
JOIN solutions s ON i.id = s.issue_id
WHERE i.title LIKE '%keyword%'
ORDER BY i.created_at DESC;
```

## Best Practices

1. **Always check for existing issues** before creating new ones to prevent duplication.
2. **Link solutions to issues** through the `issue_id` foreign key.
3. **Structure thought processes** with clear thought numbers and session organization.
4. **Use timestamps** consistently with `datetime('now')` function.
5. **Use transactions** for complex operations involving multiple tables.

## Example Workflow

When analyzing a new issue:

1. **Check for similar issues**:
   ```sql
   SELECT id, title FROM issues WHERE title LIKE '%keyword%' OR description LIKE '%keyword%';
   ```

2. **If no similar issue exists, create one**:
   ```sql
   INSERT INTO issues (title, description, status, priority, created_at) 
   VALUES ('New Issue', 'Issue description', 'pending', 2, datetime('now'));
   ```

3. **Start a thought session**:
   ```sql
   INSERT INTO thought_sessions (name, description, created_at) 
   VALUES ('Analysis of New Issue', 'Thinking through the new issue', datetime('now'));
   ```

4. **Record your thoughts**:
   ```sql
   INSERT INTO thoughts (session_id, thought_number, total_thoughts, thought_content, is_revision, next_thought_needed, created_at) 
   VALUES (session_id, 1, 3, 'First thought about the issue', 0, 1, datetime('now'));
   ```

5. **Add a solution**:
   ```sql
   INSERT INTO solutions (issue_id, description, implemented, created_at) 
   VALUES (issue_id, 'Solution description', 1, datetime('now'));
   ```

6. **Document root cause**:
   ```sql
   INSERT INTO root_causes (issue_id, cause_description, severity, created_at) 
   VALUES (issue_id, 'Root cause identified', 'medium', datetime('now'));
   ```

This database is designed to maintain a knowledge repository of analysis processes, problems, and solutions encountered during the data visualization assignment. By following this guide, future LLMs can efficiently interact with the database structure without spending time re-analyzing its schema.